use candle_core::{Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder, Linear, RmsNorm};
use serde::{Deserialize, Serialize};

// --- Simplified replacements for mistralrs ---

#[derive(Clone, Debug)]
pub enum AttentionMask {
    None,
    CausalFlash,
    Custom(Tensor),
}

pub struct SdpaParams {
    pub n_kv_groups: usize,
    pub softcap: Option<f32>,
    pub softmax_scale: f32,
    pub sliding_window: Option<usize>,
    pub sinks: Option<Tensor>,
}

pub struct Sdpa;

impl Sdpa {
    pub fn run_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _mask: &AttentionMask,
        _causal: Option<()>,
        params: &SdpaParams,
    ) -> Result<Tensor> {
        let (_b, h_q, _s, _d) = q.dims4()?;
        let h_kv = k.dim(1)?;
        
        let k = self.repeat_kv(k.clone(), h_q / h_kv)?;
        let v = self.repeat_kv(v.clone(), h_q / h_kv)?;

        let q = q.contiguous()?;
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let att = (q.matmul(&k_t)? * (params.softmax_scale as f64))?;
        let att = candle_nn::ops::softmax(&att, D::Minus1)?;
        att.matmul(&v.contiguous()?)
    }

    fn repeat_kv(&self, x: Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            x.contiguous()
        } else {
            let (b, h, s, d) = x.dims4()?;
            x.unsqueeze(2)?
                .expand((b, h, n_rep, s, d))?
                .reshape((b, h * n_rep, s, d))?
                .contiguous()
        }
    }
}

pub struct TopKOutput {
    pub values: Tensor,
    pub indices: Tensor,
}

pub trait TopKLastDimOp {
    fn topk(&self, topk: usize) -> Result<TopKOutput>;
}

impl TopKLastDimOp for Tensor {
    fn topk(&self, topk: usize) -> Result<TopKOutput> {
        let (values, sorted_indices) = self.sort_last_dim(false)?;
        let topk_indices = sorted_indices.narrow(D::Minus1, 0, topk)?.contiguous()?;
        let topk_values = values.narrow(D::Minus1, 0, topk)?.contiguous()?;
        Ok(TopKOutput {
            values: topk_values,
            indices: topk_indices,
        })
    }
}

// ---------------------------------------------

// Simplified Config to match granite.rs and config.json
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub shared_intermediate_size: Option<usize>,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_position_embeddings: usize,
    pub tie_word_embeddings: bool,
    pub layer_types: Vec<String>,
    pub attention_multiplier: f32,
    pub embedding_multiplier: f32,
    pub residual_multiplier: f32,
    pub logits_scaling: f32,
    pub attention_bias: bool,
    // Mamba configuration
    pub mamba_n_heads: Option<usize>,
    pub mamba_n_groups: usize,
    pub mamba_d_state: usize,
    pub mamba_d_head: Option<usize>,
    pub mamba_d_conv: usize,
    pub mamba_expand: usize,
    pub mamba_chunk_size: usize,
    pub mamba_conv_bias: bool,
    pub mamba_proj_bias: bool,
    // MoE configuration
    pub num_local_experts: usize,
    pub num_experts_per_tok: usize,
    pub position_embedding_type: String,
    #[serde(default)]
    pub use_fp4_kv: bool,
}

impl Config {
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }
    pub fn shared_intermediate_size(&self) -> usize {
        self.shared_intermediate_size.unwrap_or(self.intermediate_size)
    }
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    pub fn mamba_intermediate_size(&self) -> usize {
        self.mamba_expand * self.hidden_size
    }
    pub fn mamba_n_heads(&self) -> usize {
        self.mamba_n_heads.unwrap_or(128)
    }
    pub fn mamba_d_head(&self) -> usize {
        self.mamba_d_head.unwrap_or(self.mamba_intermediate_size() / self.mamba_n_heads())
    }
    pub fn mamba_conv_dim(&self) -> usize {
        self.mamba_intermediate_size() + 2 * self.mamba_n_groups * self.mamba_d_state
    }
}

// Model Forward Context for basic inference
pub struct ModelForwardContext {
    pub seqlen_offsets: Vec<usize>,
}

pub struct GraniteMlp {
    input_linear: Linear,
    output_linear: Linear,
}

impl GraniteMlp {
    pub fn new(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let size = cfg.shared_intermediate_size();
        let input_linear = candle_nn::linear_no_bias(cfg.hidden_size, size * 2, vb.pp("shared_mlp").pp("input_linear"))?;
        let output_linear = candle_nn::linear_no_bias(size, cfg.hidden_size, vb.pp("shared_mlp").pp("output_linear"))?;
        Ok(Self { input_linear, output_linear })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let projected = self.input_linear.forward(x)?;
        let chunks = projected.chunk(2, D::Minus1)?;
        let gated = (candle_nn::ops::silu(&chunks[0])? * &chunks[1])?;
        self.output_linear.forward(&gated)
    }
}

// MoE parts
struct GraniteTopKGating {
    layer: candle_nn::Linear,
    num_experts: usize,
    top_k: usize,
}

impl GraniteTopKGating {
    fn new(input_size: usize, num_experts: usize, top_k: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.pp("layer").get((num_experts, input_size), "weight")?;
        Ok(Self { layer: candle_nn::Linear::new(weight, None), num_experts, top_k })
    }
    fn forward(&self, x: &Tensor) -> Result<(Tensor, Tensor, Vec<usize>)> {
        let logits = self.layer.forward(x)?;
        // Simplified Top-K for CPU/Inference
        let topk = logits.topk(self.top_k)?;
        let indices = topk.indices;
        let values = candle_nn::ops::softmax(&topk.values, D::Minus1)?;
        
        let selected_experts = indices.to_vec2::<u32>()?;
        let routing_weights = values.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?;
        
        let mut expert_token_gates = Vec::new();
        let mut expert_counts = vec![0usize; self.num_experts];
        for (token_idx, (experts, weights)) in selected_experts.iter().zip(routing_weights.iter()).enumerate() {
            for (&expert_idx, &gate) in experts.iter().zip(weights.iter()) {
                let expert_idx = expert_idx as usize;
                expert_token_gates.push((expert_idx, token_idx, gate));
                expert_counts[expert_idx] += 1;
            }
        }
        expert_token_gates.sort_by_key(|(idx, _, _)| *idx);
        let batch_index = Tensor::from_vec(expert_token_gates.iter().map(|(_, t, _)| *t as u32).collect::<Vec<u32>>(), (expert_token_gates.len(),), x.device())?;
        let batch_gates = Tensor::from_vec(expert_token_gates.iter().map(|(_, _, g)| *g).collect::<Vec<f32>>(), (expert_token_gates.len(),), x.device())?.to_dtype(x.dtype())?;
        Ok((batch_index, batch_gates, expert_counts))
    }
}

struct GraniteParallelExperts {
    weights: Vec<Tensor>,
    output_size: usize,
}

impl GraniteParallelExperts {
    fn new(num_experts: usize, input_size: usize, output_size: usize, vb: VarBuilder) -> Result<Self> {
        let all_weights = vb.get((num_experts, output_size, input_size), "weight")?;
        let weights = (0..num_experts).map(|i| all_weights.i(i)).collect::<Result<Vec<_>>>()?;
        Ok(Self { weights, output_size })
    }
    fn forward(&self, x: &Tensor, expert_size: &[usize]) -> Result<Tensor> {
        let mut outputs = Vec::new();
        let mut offset = 0;
        for (idx, &size) in expert_size.iter().enumerate() {
            if size == 0 { continue; }
            let input = x.narrow(0, offset, size)?;
            outputs.push(input.matmul(&self.weights[idx].t()?)?);
            offset += size;
        }
        if outputs.is_empty() {
            Tensor::zeros((0, self.output_size), x.dtype(), x.device())
        } else {
            Tensor::cat(&outputs, 0)
        }
    }
}

struct GraniteMoE {
    input_linear: GraniteParallelExperts,
    output_linear: GraniteParallelExperts,
    router: GraniteTopKGating,
}

impl GraniteMoE {
    fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            input_linear: GraniteParallelExperts::new(cfg.num_local_experts, cfg.hidden_size, cfg.intermediate_size * 2, vb.pp("input_linear"))?,
            output_linear: GraniteParallelExperts::new(cfg.num_local_experts, cfg.intermediate_size, cfg.hidden_size, vb.pp("output_linear"))?,
            router: GraniteTopKGating::new(cfg.hidden_size, cfg.num_local_experts, cfg.num_experts_per_tok, vb.pp("router"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, h) = x.dims3()?;
        let x_flat = x.reshape((b * s, h))?;
        let (indices, gates, counts) = self.router.forward(&x_flat)?;
        if indices.dim(0)? == 0 { return Tensor::zeros_like(x); }
        let inputs = x_flat.index_select(&indices, 0)?;
        let hidden = self.input_linear.forward(&inputs, &counts)?;
        let chunks = hidden.chunk(2, D::Minus1)?;
        let hidden = (candle_nn::ops::silu(&chunks[0])? * &chunks[1])?;
        let outputs = self.output_linear.forward(&hidden, &counts)?;
        let outputs = outputs.broadcast_mul(&gates.unsqueeze(1)?)?;
        
        // Scatter-add logic (simplified for CPU)
        let indices_vec = indices.to_vec1::<u32>()?;
        let outputs_f32 = outputs.to_dtype(candle_core::DType::F32)?.to_vec2::<f32>()?;
        let mut flat_res = vec![vec![0.0f32; h]; b * s];
        for (i, &idx) in indices_vec.iter().enumerate() {
            for (j, &val) in outputs_f32[i].iter().enumerate() {
                flat_res[idx as usize][j] += val;
            }
        }
        Tensor::from_vec(flat_res.into_iter().flatten().collect::<Vec<f32>>(), (b * s, h), x.device())?.to_dtype(x.dtype())?.reshape((b, s, h))
    }
}

// Mamba parts
#[derive(Clone)]
pub struct MambaLayerCache {
    pub conv_state: Tensor,
    pub ssm_state: Tensor,
}

impl MambaLayerCache {
    pub fn new(batch: usize, cfg: &Config, device: &Device, dtype: candle_core::DType) -> Result<Self> {
        let conv_dim = cfg.mamba_conv_dim();
        Ok(Self {
            conv_state: Tensor::zeros((batch, conv_dim, cfg.mamba_d_conv), dtype, device)?,
            ssm_state: Tensor::zeros((batch, cfg.mamba_n_heads(), cfg.mamba_d_head(), cfg.mamba_d_state), dtype, device)?,
        })
    }
    pub fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.ssm_state = self.ssm_state.zeros_like()?;
        Ok(())
    }
}

struct RmsNormGated {
    weight: Tensor,
    eps: f64,
}

impl RmsNormGated {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        Ok(Self { weight: vb.get(size, "weight")?, eps })
    }
    fn forward(&self, x: &Tensor, gate: Option<&Tensor>) -> Result<Tensor> {
        let mut x = x.to_dtype(candle_core::DType::F32)?;
        if let Some(g) = gate {
            x = (x * candle_nn::ops::silu(&g.to_dtype(candle_core::DType::F32)?)?)?;
        }
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let norm = x.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        norm.to_dtype(self.weight.dtype())?.broadcast_mul(&self.weight)
    }
}

struct MambaLayer {
    in_proj: candle_nn::Linear,
    conv1d_weight: Tensor,
    conv1d_bias: Option<Tensor>,
    dt_bias: Tensor,
    a_log: Tensor,
    d: Tensor,
    norm: RmsNormGated,
    out_proj: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    ssm_state_size: usize,
    conv_kernel_size: usize,
    n_groups: usize,
}

impl MambaLayer {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let intermediate = cfg.mamba_intermediate_size();
        let conv_dim = cfg.mamba_conv_dim();
        let num_heads = cfg.mamba_n_heads();
        let head_dim = cfg.mamba_d_head();
        let ssm_state_size = cfg.mamba_d_state;
        let conv_kernel_size = cfg.mamba_d_conv;
        let n_groups = cfg.mamba_n_groups;
        let proj_size = intermediate + conv_dim + num_heads;
        
        let in_proj = candle_nn::Linear::new(vb.pp("in_proj").get((proj_size, cfg.hidden_size), "weight")?, cfg.mamba_proj_bias.then(|| vb.pp("in_proj").get(proj_size, "bias")).transpose()?);
        let conv1d_weight = vb.pp("conv1d").get((conv_dim, 1, conv_kernel_size), "weight")?;
        let conv1d_bias = cfg.mamba_conv_bias.then(|| vb.pp("conv1d").get(conv_dim, "bias")).transpose()?;
        let dt_bias = vb.get(num_heads, "dt_bias")?;
        let a_log = vb.get(num_heads, "A_log")?;
        let d = vb.get(num_heads, "D")?;
        let norm = RmsNormGated::new(intermediate, cfg.rms_norm_eps, vb.pp("norm"))?;
        let out_proj = candle_nn::Linear::new(vb.pp("out_proj").get((cfg.hidden_size, intermediate), "weight")?, cfg.mamba_proj_bias.then(|| vb.pp("out_proj").get(cfg.hidden_size, "bias")).transpose()?);
        
        Ok(Self { in_proj, conv1d_weight, conv1d_bias, dt_bias, a_log, d, norm, out_proj, num_heads, head_dim, intermediate_size: intermediate, ssm_state_size, conv_kernel_size, n_groups })
    }

    fn forward(&self, x: &Tensor, cache: &mut MambaLayerCache) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let proj = self.in_proj.forward(x)?;
        
        let groups_time_state_size = self.n_groups * self.ssm_state_size;

        let gate = proj.narrow(D::Minus1, 0, self.intermediate_size)?;
        let h_b_c = proj.narrow(D::Minus1, self.intermediate_size, self.intermediate_size + 2 * groups_time_state_size)?;
        let dt = proj.narrow(D::Minus1, self.intermediate_size + self.intermediate_size + 2 * groups_time_state_size, self.num_heads)?;

        // 속도 개선: 루프 내부에서 반복 계산될 필요가 없는 텐서 연산을 루프 외부로 분리 (Prefill 속도 향상)
        let a_expanded = self.a_log.to_dtype(candle_core::DType::F32)?.exp()?.neg()?
            .unsqueeze(1)?.unsqueeze(2)?.expand((self.num_heads, self.head_dim, self.ssm_state_size))?.unsqueeze(0)?;
        let dt_bias_f32 = self.dt_bias.to_dtype(candle_core::DType::F32)?.unsqueeze(0)?;
        let d_f32 = self.d.to_dtype(candle_core::DType::F32)?.unsqueeze(0)?.unsqueeze(2)?;
        let conv1d_weight_ext = self.conv1d_weight.squeeze(1)?.unsqueeze(0)?;

        // Simplified Mamba loop (CPU fallback style)
        let mut outputs = Vec::new();
        for t in 0..s {
            let h_b_c_t = h_b_c.i((.., t, ..))?;
            let dt_t = dt.i((.., t, ..))?;
            
            // Update conv state
            let next_conv = cache.conv_state.narrow(2, 1, self.conv_kernel_size - 1)?;
            cache.conv_state = Tensor::cat(&[next_conv, h_b_c_t.unsqueeze(2)?], 2)?;
            
            let mut conv_out = (cache.conv_state.clone() * conv1d_weight_ext.clone())?.sum(D::Minus1)?;
            if let Some(ref b) = self.conv1d_bias { conv_out = conv_out.broadcast_add(b)?; }
            let conv_out = candle_nn::ops::silu(&conv_out)?;
            
            let hs = conv_out.narrow(D::Minus1, 0, self.intermediate_size)?;
            let b_val = conv_out.narrow(D::Minus1, self.intermediate_size, groups_time_state_size)?;
            let c_val = conv_out.narrow(D::Minus1, self.intermediate_size + groups_time_state_size, groups_time_state_size)?;
            
            let dt_t_f32 = dt_t.to_dtype(candle_core::DType::F32)?.broadcast_add(&dt_bias_f32)?;
            let dt_t_f32 = (dt_t_f32.exp()? + 1.0)?.log()?; // softplus
            
            let dt_t_expanded = dt_t_f32.unsqueeze(2)?.expand((b, self.num_heads, self.head_dim))?;
            let da = dt_t_expanded.unsqueeze(3)?.broadcast_mul(&a_expanded)?.exp()?;

            let b_val = b_val.reshape((b, self.n_groups, self.ssm_state_size))?.to_dtype(candle_core::DType::F32)?;
            let b_val_expanded = b_val.unsqueeze(2)?
                .expand((b, self.n_groups, self.num_heads / self.n_groups, self.ssm_state_size))?
                .reshape((b, self.num_heads, self.ssm_state_size))?;
            let db = dt_t_expanded.unsqueeze(3)?.broadcast_mul(&b_val_expanded.unsqueeze(2)?)?;
            
            let hs = hs.reshape((b, self.num_heads, self.head_dim))?.to_dtype(candle_core::DType::F32)?;
            let dbx = db.broadcast_mul(&hs.unsqueeze(3)?)?;
            
            let new_ssm = (cache.ssm_state.to_dtype(candle_core::DType::F32)?.broadcast_mul(&da)?).broadcast_add(&dbx)?;
            cache.ssm_state = new_ssm.to_dtype(cache.ssm_state.dtype())?;
            
            let c_val = c_val.reshape((b, self.n_groups, self.ssm_state_size))?.to_dtype(candle_core::DType::F32)?;
            let c_val_expanded = c_val.unsqueeze(2)?
                .expand((b, self.n_groups, self.num_heads / self.n_groups, self.ssm_state_size))?
                .reshape((b, self.num_heads, self.ssm_state_size))?;
            
            let y_t = cache.ssm_state.to_dtype(candle_core::DType::F32)?.matmul(&c_val_expanded.unsqueeze(3)?)?.squeeze(3)?;
            let y_t = y_t.broadcast_add(&hs.broadcast_mul(&d_f32)?)?;
            outputs.push(y_t.reshape((b, self.intermediate_size))?);
        }
        let y = Tensor::stack(&outputs, 1)?;
        let y = self.norm.forward(&y, Some(&gate))?;
        self.out_proj.forward(&y)
    }
}

pub struct MambaBlock {
    rms_1: RmsNorm,
    mamba: MambaLayer,
    rms_2: RmsNorm,
    mlp: GraniteMlp,
    moe: Option<GraniteMoE>,
    res_mult: f32,
}

impl MambaBlock {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        Ok(Self {
            rms_1: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            mamba: MambaLayer::load(vb.pp("mamba"), cfg)?,
            rms_2: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
            mlp: GraniteMlp::new(vb.clone(), cfg)?,
            moe: (cfg.num_local_experts > 0).then(|| GraniteMoE::new(cfg, vb.pp("block_sparse_moe"))).transpose()?,
            res_mult: cfg.residual_multiplier,
        })
    }
    fn forward(&self, x: &Tensor, cache: &mut MambaLayerCache) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let mamba_out = self.mamba.forward(&x, cache)?;
        let x = (residual + mamba_out.affine(self.res_mult as f64, 0.0)?)?;
        
        let residual = &x;
        let x = self.rms_2.forward(&x)?;
        let mlp_out = self.mlp.forward(&x)?;
        let ffn_out = if let Some(ref moe) = self.moe { (mlp_out + moe.forward(&x)?)? } else { mlp_out };
        residual + ffn_out.affine(self.res_mult as f64, 0.0)?
    }
}

pub struct AttentionBlock {
    rms_1: RmsNorm,
    attn: Sdpa,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rms_2: RmsNorm,
    mlp: GraniteMlp,
    moe: Option<GraniteMoE>,
    cfg: Config,
}

impl AttentionBlock {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let size_q = cfg.head_dim() * cfg.num_attention_heads;
        let size_kv = cfg.head_dim() * cfg.num_key_value_heads();
        let vb_attn = vb.pp("self_attn");
        Ok(Self {
            rms_1: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            attn: Sdpa,
            q_proj: if cfg.attention_bias { candle_nn::linear(cfg.hidden_size, size_q, vb_attn.pp("q_proj"))? } else { candle_nn::linear_no_bias(cfg.hidden_size, size_q, vb_attn.pp("q_proj"))? },
            k_proj: if cfg.attention_bias { candle_nn::linear(cfg.hidden_size, size_kv, vb_attn.pp("k_proj"))? } else { candle_nn::linear_no_bias(cfg.hidden_size, size_kv, vb_attn.pp("k_proj"))? },
            v_proj: if cfg.attention_bias { candle_nn::linear(cfg.hidden_size, size_kv, vb_attn.pp("v_proj"))? } else { candle_nn::linear_no_bias(cfg.hidden_size, size_kv, vb_attn.pp("v_proj"))? },
            o_proj: if cfg.attention_bias { candle_nn::linear(size_q, cfg.hidden_size, vb_attn.pp("o_proj"))? } else { candle_nn::linear_no_bias(size_q, cfg.hidden_size, vb_attn.pp("o_proj"))? },
            rms_2: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("post_attention_layernorm"))?,
            mlp: GraniteMlp::new(vb.clone(), cfg)?,
            moe: (cfg.num_local_experts > 0).then(|| GraniteMoE::new(cfg, vb.pp("block_sparse_moe"))).transpose()?,
            cfg: cfg.clone(),
        })
    }
    fn forward(&self, x: &Tensor, kv_cache: &mut (Tensor, Tensor)) -> Result<Tensor> {
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let q = self.q_proj.forward(&x)?;
        let k = self.k_proj.forward(&x)?;
        let v = self.v_proj.forward(&x)?;
        
        let (b, s, _) = x.dims3()?;
        let q = q.reshape((b, s, self.cfg.num_attention_heads, self.cfg.head_dim()))?.transpose(1, 2)?;
        let k = k.reshape((b, s, self.cfg.num_key_value_heads(), self.cfg.head_dim()))?.transpose(1, 2)?;
        let v = v.reshape((b, s, self.cfg.num_key_value_heads(), self.cfg.head_dim()))?.transpose(1, 2)?;
        
        // 🌟 [FP4 Compression] Update simple KV cache with On-the-fly F4 Quantization
        let target_dtype = k.dtype(); 
        
        let (k, v) = if kv_cache.0.dim(2)? == 0 {
            // 최초 저장 시 FP4로 압축하여 보관
            *kv_cache = (
                k.to_dtype(candle_core::DType::F4).unwrap_or_else(|_| k.clone()), 
                v.to_dtype(candle_core::DType::F4).unwrap_or_else(|_| v.clone())
            );
            (k, v) // 현재 연산에는 원본 사용
        } else {
            // 1. 기존 압축된 FP4 캐시를 연산 및 결합을 위해 원래 타입(BF16/F32)으로 임시 복원
            let k_prev = kv_cache.0.to_dtype(target_dtype).unwrap_or_else(|_| kv_cache.0.clone());
            let v_prev = kv_cache.1.to_dtype(target_dtype).unwrap_or_else(|_| kv_cache.1.clone());
            
            // 2. 새로운 토큰 결합
            let k_new = Tensor::cat(&[&k_prev, &k], 2)?;
            let v_new = Tensor::cat(&[&v_prev, &v], 2)?;
            
            // 3. 업데이트된 전체 캐시를 다시 FP4로 압축하여 VRAM 장부에 저장
            *kv_cache = (
                k_new.to_dtype(candle_core::DType::F4).unwrap_or_else(|_| k_new.clone()), 
                v_new.to_dtype(candle_core::DType::F4).unwrap_or_else(|_| v_new.clone())
            );
            
            (k_new, v_new) // 현재 연산에는 복원된 텐서 사용
        };

        let mut attn_out = self.attn.run_attention(&q, &k, &v, &AttentionMask::None, None, &SdpaParams {
            n_kv_groups: self.cfg.num_attention_heads / self.cfg.num_key_value_heads(),
            softmax_scale: self.cfg.attention_multiplier,
            softcap: None, sliding_window: None, sinks: None,
        })?;
        attn_out = attn_out.transpose(1, 2)?.reshape((b, s, ()))?;
        let attn_out = self.o_proj.forward(&attn_out)?;
        
        let x = (residual + attn_out.affine(self.cfg.residual_multiplier as f64, 0.0)?)?;
        let residual = &x;
        let x = self.rms_2.forward(&x)?;
        let mlp_out = self.mlp.forward(&x)?;
        let ffn_out = if let Some(ref moe) = self.moe { (mlp_out + moe.forward(&x)?)? } else { mlp_out };
        residual + ffn_out.affine(self.cfg.residual_multiplier as f64, 0.0)?
    }
}

pub enum DecoderLayer {
    Attention(AttentionBlock),
    Mamba(MambaBlock),
}

#[derive(Clone)]
pub struct GraniteHybridCache {
    pub attention_caches: Vec<(Tensor, Tensor)>,
    pub mamba_caches: Vec<MambaLayerCache>,
}

pub struct GraniteMoeHybrid {
    pub wte: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub ln_f: RmsNorm,
    pub lm_head: Linear,
    pub cfg: Config,
}

impl GraniteMoeHybrid {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let vb_m = vb.pp("model");
        let wte = Embedding::new(vb_m.pp("embed_tokens").get((cfg.vocab_size, cfg.hidden_size), "weight")?, cfg.hidden_size);
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            let vb_l = vb_m.pp("layers").pp(i);
            let layer = if cfg.layer_types[i] == "attention" {
                DecoderLayer::Attention(AttentionBlock::load(vb_l, cfg)?)
            } else {
                DecoderLayer::Mamba(MambaBlock::load(vb_l, cfg)?)
            };
            layers.push(layer);
        }
        let ln_f = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            candle_nn::Linear::new(wte.embeddings().clone(), None)
        } else {
            candle_nn::linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self { wte, layers, ln_f, lm_head, cfg: cfg.clone() })
    }

    pub fn forward(&self, input_ids: &Tensor, cache: &mut GraniteHybridCache) -> Result<Tensor> {
        let mut x = self.wte.forward(input_ids)?;
        x = x.affine(self.cfg.embedding_multiplier as f64, 0.0)?;
        
        let mut att_idx = 0;
        let mut mam_idx = 0;
        for layer in &self.layers {
            match layer {
                DecoderLayer::Attention(block) => {
                    x = block.forward(&x, &mut cache.attention_caches[att_idx])?;
                    att_idx += 1;
                }
                DecoderLayer::Mamba(block) => {
                    x = block.forward(&x, &mut cache.mamba_caches[mam_idx])?;
                    mam_idx += 1;
                }
            }
        }
        let x = self.ln_f.forward(&x)?;
        let logits = self.lm_head.forward(&x)?;
        logits.affine(1.0 / self.cfg.logits_scaling as f64, 0.0)
    }
}


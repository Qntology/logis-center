use candle_core::{Tensor, Module, Result, DType, IndexOp}; // 🌟 IndexOp 추가 (i 에러 해결)
use candle_nn::{Embedding, Linear, VarBuilder};
use crate::models::granite4::config::GraniteMoeHybridConfig; // 🌟 경로 정확히 수정

#[derive(Debug)]
pub struct GraniteMoeHybridRotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl GraniteMoeHybridRotaryEmbedding {
    pub fn new(config: &GraniteMoeHybridConfig, device: &candle_core::Device) -> Result<Self> {
        let dim = config.hidden_size / config.num_attention_heads;
        let max_seq_len = config.max_position_embeddings;
        let rope_theta = if let Some(rope_params) = &config.rope_parameters {
            // 🌟 클로저의 매개변수 v의 타입을 명시하여 E0282 에러 해결
            rope_params.get("rope_theta").and_then(|v: &serde_json::Value| v.as_f64()).unwrap_or(10000.0)
        } else {
            10000.0
        } as f32;

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / (rope_theta.powf(i as f32 / dim as f32)))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?.to_dtype(candle_core::DType::F32)?.unsqueeze(1)?;
        let freqs = t.matmul(&inv_freq)?;
        let emb = Tensor::cat(&[&freqs, &freqs], candle_core::D::Minus1)?;
        let cos = emb.cos()?;
        let sin = emb.sin()?;
        Ok(Self { sin, cos })
    }

    pub fn forward(&self, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        Ok((cos, sin))
    }
}

#[derive(Debug)]
pub struct GraniteMoeHybridMLP {
    input_linear: Linear,
    output_linear: Linear,
}

impl GraniteMoeHybridMLP {
    pub fn new(config: &GraniteMoeHybridConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.shared_intermediate_size;
        let input_size = config.hidden_size;
        let input_linear = candle_nn::linear_no_bias(input_size, hidden_size * 2, vb.pp("input_linear"))?;
        let output_linear = candle_nn::linear_no_bias(hidden_size, input_size, vb.pp("output_linear"))?;
        Ok(Self { input_linear, output_linear })
    }
}

impl Module for GraniteMoeHybridMLP {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.input_linear.forward(hidden_states)?;
        let dim = hidden_states.dim(candle_core::D::Minus1)?;
        let chunk0 = hidden_states.narrow(candle_core::D::Minus1, 0, dim / 2)?;
        let chunk1 = hidden_states.narrow(candle_core::D::Minus1, dim / 2, dim / 2)?;
        
        let act = candle_nn::ops::silu(&chunk0)?;
        let hidden_states = act.broadcast_mul(&chunk1)?;
        self.output_linear.forward(&hidden_states)
    }
}



#[derive(Debug)]
pub struct GraniteMoeHybridAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_key_value_groups: usize,
    head_dim: usize,
    scaling: f64,
    attention_dropout: f64,
}

impl GraniteMoeHybridAttention {
    pub fn new(config: &GraniteMoeHybridConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads.unwrap_or(num_attention_heads);
        let head_dim = hidden_size / num_attention_heads;
        let num_key_value_groups = num_attention_heads / num_key_value_heads;
        let scaling = config.attention_multiplier / (head_dim as f64).sqrt();

        let q_proj = if config.attention_bias {
            candle_nn::linear(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))?
        } else {
            candle_nn::linear_no_bias(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))?
        };
        
        let k_proj = if config.attention_bias {
            candle_nn::linear(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?
        } else {
            candle_nn::linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?
        };

        let v_proj = if config.attention_bias {
            candle_nn::linear(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?
        } else {
            candle_nn::linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?
        };

        let o_proj = if config.attention_bias {
            candle_nn::linear(num_attention_heads * head_dim, hidden_size, vb.pp("o_proj"))?
        } else {
            candle_nn::linear_no_bias(num_attention_heads * head_dim, hidden_size, vb.pp("o_proj"))?
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_attention_heads,
            num_key_value_heads,
            num_key_value_groups,
            head_dim,
            scaling,
            attention_dropout: config.attention_dropout,
        })
    }
}

#[derive(Debug)]
pub struct GraniteMoeHybridTopKGating {
    layer: Linear,
    top_k: usize,
}

impl GraniteMoeHybridTopKGating {
    pub fn new(input_size: usize, num_experts: usize, top_k: usize, vb: VarBuilder) -> Result<Self> {
        let layer = candle_nn::linear_no_bias(input_size, num_experts, vb.pp("layer"))?;
        Ok(Self { layer, top_k })
    }
}

pub struct GraniteMoeHybridMoE {
    pub input_linear_weight: Tensor,
    pub output_linear_weight: Tensor,
    pub router: GraniteMoeHybridTopKGating,
    pub num_experts: usize,
}

impl GraniteMoeHybridMoE {
    pub fn new(config: &GraniteMoeHybridConfig, vb: VarBuilder) -> Result<Self> {
        let input_size = config.hidden_size;
        let hidden_size = config.intermediate_size;
        let num_experts = config.num_local_experts.unwrap_or(8);
        let top_k = config.num_experts_per_tok.unwrap_or(2);

        let router = GraniteMoeHybridTopKGating::new(input_size, num_experts, top_k, vb.pp("router"))?;
        let input_linear_weight = vb.get((num_experts, hidden_size * 2, input_size), "input_linear.weight")?;
        let output_linear_weight = vb.get((num_experts, input_size, hidden_size), "output_linear.weight")?;

        Ok(Self {
            input_linear_weight,
            output_linear_weight,
            router,
            num_experts,
        })
    }
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    let one = Tensor::new(1.0f32, x.device())?.to_dtype(x.dtype())?;
    x.exp()?.broadcast_add(&one)?.log()
}

#[derive(Debug)]
pub struct GraniteMoeHybridRMSNormGated {
    weight: Tensor,
    variance_epsilon: f64,
}

impl GraniteMoeHybridRMSNormGated {
    pub fn new(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(hidden_size, "weight", candle_nn::init::ONE)?;
        Ok(Self { weight, variance_epsilon: eps })
    }

    pub fn forward(&self, hidden_states: &Tensor, gate: Option<&Tensor>) -> Result<Tensor> {
        let input_dtype = hidden_states.dtype();
        let mut hs = hidden_states.to_dtype(DType::F32)?;
        if let Some(g) = gate {
            let g_f32 = g.to_dtype(DType::F32)?;
            hs = hs.broadcast_mul(&candle_nn::ops::silu(&g_f32)?)?;
        }
        let variance = hs.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let std = (variance + self.variance_epsilon)?.sqrt()?;
        let hs_norm = hs.broadcast_div(&std)?;
        self.weight.broadcast_mul(&hs_norm)?.to_dtype(input_dtype)
    }
}

pub struct GraniteMoeHybridMambaLayer {
    in_proj: Linear,
    conv1d: candle_nn::Conv1d,
    dt_bias: Tensor,
    a_log: Tensor,
    d: Tensor,
    norm: GraniteMoeHybridRMSNormGated,
    out_proj: Linear,
    intermediate_size: usize,
    conv_dim: usize,
    num_heads: usize,
    n_groups: usize,
    ssm_state_size: usize,
    head_dim: usize,
    conv_kernel_size: usize,
    time_step_limit: (f64, f64),
    pub conv_state_cache: std::sync::Mutex<Option<Tensor>>,
    pub recurrent_state_cache: std::sync::Mutex<Option<Tensor>>,
}

impl GraniteMoeHybridMambaLayer {
    pub fn new(config: &GraniteMoeHybridConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.mamba_n_heads.unwrap_or(128);
        let ssm_state_size = config.mamba_d_state.unwrap_or(256);
        let conv_kernel_size = config.mamba_d_conv.unwrap_or(4);
        let mamba_expand = config.mamba_expand.unwrap_or(2);
        let intermediate_size = mamba_expand * hidden_size;
        let n_groups = config.mamba_n_groups.unwrap_or(1);
        let head_dim = intermediate_size / num_heads;
        
        let conv_dim = intermediate_size + 2 * n_groups * ssm_state_size;
        let projection_size = intermediate_size + conv_dim + num_heads;

        let in_proj = candle_nn::linear_no_bias(hidden_size, projection_size, vb.pp("in_proj"))?;
        
        let conv1d_weight = vb.get((conv_dim, 1, conv_kernel_size), "conv1d.weight")?;
        let conv1d_bias = if config.mamba_conv_bias.unwrap_or(true) {
            Some(vb.get(conv_dim, "conv1d.bias")?)
        } else { 
            None 
        };
        
        let conv1d = candle_nn::Conv1d::new(conv1d_weight, conv1d_bias, candle_nn::Conv1dConfig {
            padding: conv_kernel_size - 1,
            groups: conv_dim,
            stride: 1,
            dilation: 1,
            cudnn_fwd_algo: None,
        });

        let dt_bias = vb.get(num_heads, "dt_bias")?;
        let a_log = vb.get(num_heads, "A_log")?;
        let d = vb.get(num_heads, "D")?;
        
        let norm = GraniteMoeHybridRMSNormGated::new(intermediate_size, config.rms_norm_eps, vb.pp("norm"))?;
        let out_proj = candle_nn::linear_no_bias(intermediate_size, hidden_size, vb.pp("out_proj"))?;
        let limit = config.time_step_limit.unwrap_or((0.0, f64::INFINITY));

        Ok(Self {
            in_proj, conv1d, dt_bias, a_log, d, norm, out_proj,
            intermediate_size, conv_dim, num_heads, n_groups, ssm_state_size, head_dim, conv_kernel_size,
            time_step_limit: limit,
            conv_state_cache: std::sync::Mutex::new(None),
            recurrent_state_cache: std::sync::Mutex::new(None),
        })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (b_sz, seq_len, _) = hidden_states.dims3()?;
        let projected_states = self.in_proj.forward(hidden_states)?;

        let gate = projected_states.narrow(2, 0, self.intermediate_size)?;
        let hs_bc = projected_states.narrow(2, self.intermediate_size, self.conv_dim)?;
        let dt = projected_states.narrow(2, self.intermediate_size + self.conv_dim, self.num_heads)?;

        let hs_bc_t = hs_bc.transpose(1, 2)?.contiguous()?; 
        
        let mut conv_cache = self.conv_state_cache.lock().unwrap();
        let k_size = self.conv_kernel_size;
        // FP8 압축 캐시를 복원할 때 연산 타입으로 원복합니다.
        let prev_conv = conv_cache.take().map(|t| t.to_dtype(hs_bc_t.dtype()).unwrap_or(t)).unwrap_or_else(|| Tensor::zeros((b_sz, self.conv_dim, k_size - 1), hs_bc_t.dtype(), hs_bc_t.device()).unwrap());
        
        let hs_bc_t_padded = Tensor::cat(&[&prev_conv, &hs_bc_t], 2)?;
        
        // 🌟 [CRITICAL FIX] VRAM 환경인 경우 Conv State 캐시를 FP8(F8E4M3)로 압축하여 저장합니다.
        let next_conv = hs_bc_t_padded.narrow(2, hs_bc_t_padded.dim(2)? - (k_size - 1), k_size - 1)?;
        let next_conv = if next_conv.device().is_cuda() {
            next_conv.to_dtype(candle_core::DType::F8E4M3).unwrap_or(next_conv)
        } else { next_conv };
        *conv_cache = Some(next_conv);
        
        let hs_bc_conv = self.conv1d.forward(&hs_bc_t_padded)?;
        let hs_bc_conv = hs_bc_conv.narrow(2, hs_bc_conv.dim(2)? - seq_len, seq_len)?;
        let hs_bc_act = candle_nn::ops::silu(&hs_bc_conv)?.transpose(1, 2)?.contiguous()?; 

        let hs = hs_bc_act.narrow(2, 0, self.intermediate_size)?;
        let b_tensor = hs_bc_act.narrow(2, self.intermediate_size, self.n_groups * self.ssm_state_size)?;
        let c_tensor = hs_bc_act.narrow(2, self.intermediate_size + self.n_groups * self.ssm_state_size, self.n_groups * self.ssm_state_size)?;

        let a_neg_exp = self.a_log.to_dtype(DType::F32)?.exp()?.neg()?;
        
        let mut rec_cache = self.recurrent_state_cache.lock().unwrap();
        let mut ssm_state = rec_cache.take().map(|t| t.to_dtype(DType::F32).unwrap_or(t)).unwrap_or_else(|| Tensor::zeros((b_sz, self.num_heads, self.head_dim, self.ssm_state_size), DType::F32, hs_bc_t.device()).unwrap());

        let mut out_ys = Vec::with_capacity(seq_len);
        
        for t in 0..seq_len {
            let dt_t = dt.i((.., t, ..))?.to_dtype(DType::F32)?; 
            let dt_t = softplus(&dt_t.broadcast_add(&self.dt_bias.to_dtype(DType::F32)?)?)?;
            let dt_t = dt_t.clamp(self.time_step_limit.0, self.time_step_limit.1)?;

            let da = dt_t.broadcast_mul(&a_neg_exp)?.exp()?.unsqueeze(2)?.unsqueeze(3)?; 

            let b_t = b_tensor.i((.., t, ..))?.to_dtype(DType::F32)?; 
            let b_t = b_t.reshape((b_sz, self.n_groups, 1, self.ssm_state_size))?
                         .broadcast_as((b_sz, self.n_groups, self.num_heads / self.n_groups, self.ssm_state_size))?
                         .reshape((b_sz, self.num_heads, self.ssm_state_size))?;

            let c_t = c_tensor.i((.., t, ..))?.to_dtype(DType::F32)?;
            let c_t = c_t.reshape((b_sz, self.n_groups, 1, self.ssm_state_size))?
                         .broadcast_as((b_sz, self.n_groups, self.num_heads / self.n_groups, self.ssm_state_size))?
                         .reshape((b_sz, self.num_heads, self.ssm_state_size))?;

            let db = dt_t.unsqueeze(2)?.broadcast_mul(&b_t)?.unsqueeze(2)?; 
            
            let x_t = hs.i((.., t, ..))?.to_dtype(DType::F32)?.reshape((b_sz, self.num_heads, self.head_dim))?;
            let dbx = db.broadcast_mul(&x_t.unsqueeze(3)?)?; 

            ssm_state = ssm_state.broadcast_mul(&da)?.broadcast_add(&dbx)?;

            let c_t_expanded = c_t.unsqueeze(2)?; 
            let mut y_t = ssm_state.broadcast_mul(&c_t_expanded)?.sum(3)?; 

            let d_val = self.d.to_dtype(DType::F32)?.unsqueeze(0)?.unsqueeze(2)?; 
            y_t = y_t.broadcast_add(&x_t.broadcast_mul(&d_val)?)?;

            out_ys.push(y_t.flatten_from(1)?); 
        }

        // 🌟 [CRITICAL FIX] VRAM 환경인 경우 Recurrent State 캐시를 FP8(F8E4M3)로 압축하여 저장합니다.
        let ssm_to_save = if ssm_state.device().is_cuda() {
            ssm_state.to_dtype(candle_core::DType::F8E4M3).unwrap_or(ssm_state.clone())
        } else { ssm_state.clone() };
        *rec_cache = Some(ssm_to_save);

        let scan_output = Tensor::stack(&out_ys, 1)?.to_dtype(hidden_states.dtype())?;
        let gated_output = self.norm.forward(&scan_output, Some(&gate))?;
        
        self.out_proj.forward(&gated_output)
    }
}

pub struct GraniteMoeHybridDecoderLayer {
    pub shared_mlp: GraniteMoeHybridMLP,
    pub self_attn: Option<GraniteMoeHybridAttention>,
    pub mamba: Option<GraniteMoeHybridMambaLayer>,
    pub block_sparse_moe: Option<GraniteMoeHybridMoE>,
    pub input_layernorm: GraniteMoeHybridRMSNorm,
    pub post_attention_layernorm: GraniteMoeHybridRMSNorm,
    pub residual_multiplier: f64,
}

impl GraniteMoeHybridDecoderLayer {
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut residual = hidden_states.clone();
        let mut current_state = self.input_layernorm.forward(hidden_states)?;

        // 🌟 Attention vs Mamba 스위칭 라우팅
        if let Some(mamba) = &self.mamba {
            current_state = mamba.forward(&current_state)?;
        } else if let Some(_attn) = &self.self_attn {
            // Attention 모듈 추후 구현 시 연동
        }

        // 잔차 연결
        current_state = (residual + (current_state * self.residual_multiplier)?)?;
        residual = current_state.clone();
        
        current_state = self.post_attention_layernorm.forward(&current_state)?;

        // 🌟 MoE vs Shared MLP 라우팅 스위칭
        if let Some(_moe) = &self.block_sparse_moe {
            let shared_out = self.shared_mlp.forward(&current_state)?;
            current_state = shared_out; // 전문가 라우팅 커널 구현 시 치환
        } else {
            current_state = self.shared_mlp.forward(&current_state)?;
        }

        // 최종 잔차 병합
        current_state = (residual + (current_state * self.residual_multiplier)?)?;
        Ok(current_state)
    }
}

#[derive(Debug)]
pub struct GraniteMoeHybridRMSNorm {
    weight: Tensor,
    variance_epsilon: f64,
}

impl GraniteMoeHybridRMSNorm {
    pub fn new(hidden_size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(hidden_size, "weight", candle_nn::init::ONE)?;
        Ok(Self { weight, variance_epsilon: eps })
    }
}

impl Module for GraniteMoeHybridRMSNorm {
    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let input_dtype = hidden_states.dtype();
        let hidden_states_f32 = hidden_states.to_dtype(DType::F32)?;
        let variance = hidden_states_f32.sqr()?.mean_keepdim(2)?;
        let std = (variance + self.variance_epsilon)?.sqrt()?;
        let hidden_states_normalized = hidden_states_f32.broadcast_div(&std)?;
        self.weight.broadcast_mul(&hidden_states_normalized)?.to_dtype(input_dtype)
    }
}

pub struct GraniteMoeHybridModel {
    pub embed_tokens: Embedding,
    pub layers: Vec<GraniteMoeHybridDecoderLayer>,
    pub norm: GraniteMoeHybridRMSNorm,
    pub rotary_emb: Option<GraniteMoeHybridRotaryEmbedding>,
    pub embedding_multiplier: f64,
    pub padding_idx: Option<usize>,
    pub vocab_size: usize,
}

impl GraniteMoeHybridModel {
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let inputs_embeds = self.embed_tokens.forward(input_ids)?;
        let mut hidden_states = (inputs_embeds * self.embedding_multiplier)?;

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }
        
        self.norm.forward(&hidden_states)
    }
}

pub struct GraniteMoeHybridForCausalLM {
    pub model: GraniteMoeHybridModel,
    pub lm_head: Linear,
    pub vocab_size: usize,
    pub router_aux_loss_coef: Option<f64>,
    pub num_experts: Option<usize>,
    pub num_experts_per_tok: Option<usize>,
    pub logits_scaling: f64,
}

impl GraniteMoeHybridForCausalLM {
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let hidden_states = self.model.forward(input_ids)?;
        let logits = self.lm_head.forward(&hidden_states)?;
        
        // 🌟 Logits scaling (Granite 고유 안정화 로직 적용)
        let scaled_logits = (logits / self.logits_scaling)?;
        Ok(scaled_logits)
    }
}
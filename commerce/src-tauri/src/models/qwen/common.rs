use anyhow::Result;
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{
    Activation, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig,
    Linear, Module, RmsNorm, VarBuilder, batch_norm, conv2d, conv2d_no_bias, layer_norm, linear,
    linear_no_bias, rms_norm,
};

use crate::{
    position_embed::rope::apply_rotary_pos_emb,
};

#[derive(Debug, Clone)]
pub struct GateUpDownMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl GateUpDownMLP {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        intermediate_size: usize,
        act_fn: Activation,
        bias: bool,
    ) -> Result<Self> {
        let (gate_proj, up_proj, down_proj) = if bias {
            (
                linear(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
                linear(hidden_size, intermediate_size, vb.pp("up_proj"))?,
                linear(intermediate_size, hidden_size, vb.pp("down_proj"))?,
            )
        } else {
            (
                linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
                linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
                linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
            )
        };
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
        })
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let g_w = self.gate_proj.weight().to_device(device)?;
        let g_b = self.gate_proj.bias().map(|b| b.to_device(device)).transpose()?;
        self.gate_proj = Linear::new(g_w, g_b);

        let u_w = self.up_proj.weight().to_device(device)?;
        let u_b = self.up_proj.bias().map(|b| b.to_device(device)).transpose()?;
        self.up_proj = Linear::new(u_w, u_b);

        let d_w = self.down_proj.weight().to_device(device)?;
        let d_b = self.down_proj.bias().map(|b| b.to_device(device)).transpose()?;
        self.down_proj = Linear::new(d_w, d_b);
        Ok(())
    }
}

impl Module for GateUpDownMLP {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
pub struct TwoLinearMLP {
    linear1: Linear,
    linear2: Linear,
    act: Activation,
}

impl TwoLinearMLP {
    pub fn new(
        vb: VarBuilder,
        embedding_dim: usize,
        mlp_dim: usize,
        act: Activation,
        bias: bool,
        linear1_pp_name: &str,
        linear2_pp_name: &str,
    ) -> Result<Self> {
        let (linear1, linear2) = if bias {
            (
                linear(embedding_dim, mlp_dim, vb.pp(linear1_pp_name))?,
                linear(mlp_dim, embedding_dim, vb.pp(linear2_pp_name))?,
            )
        } else {
            (
                linear_no_bias(embedding_dim, mlp_dim, vb.pp(linear1_pp_name))?,
                linear_no_bias(mlp_dim, embedding_dim, vb.pp(linear2_pp_name))?,
            )
        };
        Ok(Self {
            linear1,
            linear2,
            act,
        })
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let l1_w = self.linear1.weight().to_device(device)?;
        let l1_b = self.linear1.bias().map(|b| b.to_device(device)).transpose()?;
        self.linear1 = Linear::new(l1_w, l1_b);

        let l2_w = self.linear2.weight().to_device(device)?;
        let l2_b = self.linear2.bias().map(|b| b.to_device(device)).transpose()?;
        self.linear2 = Linear::new(l2_w, l2_b);
        Ok(())
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs
            .apply(&self.linear1)?
            .apply(&self.act)?
            .apply(&self.linear2)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
// pub struct AttentionNobias {
pub struct NaiveAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    middle_size: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl NaiveAttention {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: Option<usize>,
        bias: bool,
        o_proj_pp_name: Option<&str>,
    ) -> Result<Self> {
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let head_dim = match head_dim {
            None => hidden_size / num_attention_heads,
            Some(dim) => dim,
        };
        let o_proj_pp_name = o_proj_pp_name.unwrap_or("o_proj");
        let (q_proj, k_proj, v_proj, o_proj) = if bias {
            (
                linear(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))?,
                linear(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?,
                linear(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?,
                linear(
                    num_attention_heads * head_dim,
                    hidden_size,
                    vb.pp(o_proj_pp_name),
                )?,
            )
        } else {
            (
                linear_no_bias(hidden_size, num_attention_heads * head_dim, vb.pp("q_proj"))?,
                linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("k_proj"))?,
                linear_no_bias(hidden_size, num_key_value_heads * head_dim, vb.pp("v_proj"))?,
                linear_no_bias(
                    num_attention_heads * head_dim,
                    hidden_size,
                    vb.pp(o_proj_pp_name),
                )?,
            )
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: num_attention_heads,
            num_kv_heads: num_key_value_heads,
            num_kv_groups,
            head_dim,
            middle_size: num_attention_heads * head_dim,
            kv_cache: None,
        })
    }

    pub fn to_device(&mut self, device: &Device) -> Result<()> {
        let q_w = self.q_proj.weight().to_device(device)?;
        let q_b = self.q_proj.bias().map(|b| b.to_device(device)).transpose()?;
        self.q_proj = Linear::new(q_w, q_b);

        let k_w = self.k_proj.weight().to_device(device)?;
        let k_b = self.k_proj.bias().map(|b| b.to_device(device)).transpose()?;
        self.k_proj = Linear::new(k_w, k_b);

        let v_w = self.v_proj.weight().to_device(device)?;
        let v_b = self.v_proj.bias().map(|b| b.to_device(device)).transpose()?;
        self.v_proj = Linear::new(v_w, v_b);

        let o_w = self.o_proj.weight().to_device(device)?;
        let o_b = self.o_proj.bias().map(|b| b.to_device(device)).transpose()?;
        self.o_proj = Linear::new(o_w, o_b);

        if let Some((k, v)) = &self.kv_cache {
            self.kv_cache = Some((k.to_device(device)?, v.to_device(device)?));
        }
        Ok(())
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        tof32: bool,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;
        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) = if let Some(cos) = cos {
            if let Some(sin) = sin {
                apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?
            } else {
                (query_states, key_states)
            }
        } else {
            (query_states, key_states)
        };

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            scale,
        )?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.middle_size))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn forward_with_cache(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        tof32: bool,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?;
        let key_states = self.k_proj.forward(xs)?;
        let value_states = self.v_proj.forward(xs)?;
        let query_states = query_states
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key_states = key_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, tof32)?;
        let (key_states, value_states) = match &self.kv_cache {
            None => (key_states, value_states),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &key_states], 2)?;
                let value_states = Tensor::cat(&[prev_v, &value_states], 2)?;
                (key_states, value_states)
            }
        };

        self.kv_cache = Some((key_states.clone(), value_states.clone()));
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            scale,
        )?;
        let attn_output = attn_output.reshape((b_sz, q_len, self.middle_size))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

pub struct NaiveAttnTwoLinearMLPBlock {
    self_attn: NaiveAttention,
    mlp: TwoLinearMLP,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl NaiveAttnTwoLinearMLPBlock {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: Option<usize>,
        head_dim: Option<usize>,
        attn_bias: bool,
        attn_pp_name: &str,
        o_proj_pp_name: Option<&str>,
        intermediate_size: usize,
        hidden_act: Activation,
        mlp_bias: bool,
        mlp_pp_name: &str,
        linear1_pp_name: &str,
        linear2_pp_name: &str,
        norm_eps: f64,
        input_norm_pp_name: &str,
        post_norm_pp_name: &str,
    ) -> Result<Self> {
        let num_key_value_heads = match num_key_value_heads {
            Some(heads) => heads,
            None => num_attention_heads,
        };
        let self_attn = NaiveAttention::new(
            vb.pp(attn_pp_name),
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            attn_bias,
            o_proj_pp_name,
        )?;
        let mlp = TwoLinearMLP::new(
            vb.pp(mlp_pp_name),
            hidden_size,
            intermediate_size,
            hidden_act,
            mlp_bias,
            linear1_pp_name,
            linear2_pp_name,
        )?;

        let input_layernorm = get_layer_norm(vb.pp(input_norm_pp_name), norm_eps, hidden_size)?;
        let post_attention_layernorm =
            get_layer_norm(vb.pp(post_norm_pp_name), norm_eps, hidden_size)?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        cos: Option<&Tensor>,
        sin: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        tof32: bool,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, cos, sin, attention_mask, tof32)?;
        let residual = residual.add(&xs)?;
        let xs = self.post_attention_layernorm.forward(&residual)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = residual.add(&xs)?;
        Ok(xs)
    }
}

pub struct NaiveAttnGateUpDownMLPBlock {
    self_attn: NaiveAttention,
    mlp: GateUpDownMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl NaiveAttnGateUpDownMLPBlock {
    pub fn new(
        vb: VarBuilder,
        hidden_size: usize,
        num_attention_heads: usize,
        num_key_value_heads: Option<usize>,
        head_dim: Option<usize>,
        attn_bias: bool,
        attn_pp_name: &str,
        o_proj_pp_name: Option<&str>,
        intermediate_size: usize,
        hidden_act: Activation,
        mlp_bias: bool,
        mlp_pp_name: &str,
        norm_eps: f64,
        input_norm_pp_name: &str,
        post_norm_pp_name: &str,
    ) -> Result<Self> {
        let num_key_value_heads = match num_key_value_heads {
            Some(heads) => heads,
            None => num_attention_heads,
        };
        let self_attn = NaiveAttention::new(
            vb.pp(attn_pp_name),
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            attn_bias,
            o_proj_pp_name,
        )?;
        let mlp = GateUpDownMLP::new(
            vb.pp(mlp_pp_name),
            hidden_size,
            intermediate_size,
            hidden_act,
            mlp_bias,
        )?;
        let input_layernorm = rms_norm(hidden_size, norm_eps, vb.pp(input_norm_pp_name))?;
        let post_attention_layernorm = rms_norm(hidden_size, norm_eps, vb.pp(post_norm_pp_name))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward_with_cache(&xs, cos, sin, attention_mask, false)?;
        let residual = residual.add(&xs)?;
        let xs = self.post_attention_layernorm.forward(&residual)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = residual.add(&xs)?;
        Ok(xs)
    }
    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

pub fn decoding_attention_parallel(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    _num_key_value_groups: Option<usize>, // <-- Add this argument here!
    scaling: f64,
) -> Result<Tensor> {
    // Flash-Decoding style optimization for seq_len = 1 (Decoding)
    // Splits KV into chunks and parallelizes attention calculation
    let (_b_sz, _n_heads, q_len, _head_dim) = query_states.dims4()?;
    
    // [FIX] Early exit if not a decoding step (q_len > 1)
    if q_len != 1 {
        return eager_attention_forward(query_states, key_states, value_states, None, None, scaling);
    }

    // [FIX] GQA Support: Repeat KV heads if they are fewer than query heads
    let _n_kv_heads = key_states.dim(1)?;

    let kv_seq_len = key_states.dim(2)?;
    let chunk_size = 128; // Optimal chunk size for parallel reduction
    
    if kv_seq_len <= chunk_size {
        // [FIX] Pass by reference to resolve E0308
        return eager_attention_forward(query_states, &key_states, &value_states, None, None, scaling);
    }

    let num_chunks = (kv_seq_len + chunk_size - 1) / chunk_size;
    let mut chunk_outputs = Vec::with_capacity(num_chunks);
    let mut chunk_logsumexp = Vec::with_capacity(num_chunks);

    // [CRITICAL FIX] 루프 내부에서 매번 발생하던 query_states의 형변환을 루프 밖으로 빼내 GPU 스톨을 제거합니다!
    let q_aligned = query_states.to_dtype(key_states.dtype())?;

    for i in 0..num_chunks {
        let start = i * chunk_size;
        let end = (start + chunk_size).min(kv_seq_len);
        let k_chunk = key_states.narrow(2, start, end - start)?;
        let v_chunk = value_states.narrow(2, start, end - start)?;

        let attn_weights = (q_aligned.matmul(&k_chunk.transpose(2, 3)?)? * scaling)?;
        let max_logits = attn_weights.max_keepdim(D::Minus1)?;
        let exp_weights = attn_weights.broadcast_sub(&max_logits)?.exp()?;
        let sum_exp = exp_weights.sum_keepdim(D::Minus1)?;
        
        let out_chunk = exp_weights.to_dtype(v_chunk.dtype())?.matmul(&v_chunk)?;
        // [OPTIMIZATION] log()를 취하지 않고 가중치 합(sum_exp)과 max_logits만 보관
        let exp_sum_val = sum_exp; 

        chunk_outputs.push(out_chunk);
        chunk_logsumexp.push((exp_sum_val, max_logits)); // 튜플 형태로 저장
    }

    // Parallel Reduction 시작 부분 수정
    let (mut current_exp_sum, mut current_max_logit) = chunk_logsumexp[0].clone();
    let mut final_output = chunk_outputs[0].clone();

    for i in 1..num_chunks {
        let (next_exp_sum, next_max_logit) = &chunk_logsumexp[i];
        let out_i = &chunk_outputs[i];

        let new_max_logit = current_max_logit.broadcast_maximum(next_max_logit)?;
        
        let alpha = current_max_logit.broadcast_sub(&new_max_logit)?.exp()?;
        let beta = next_max_logit.broadcast_sub(&new_max_logit)?.exp()?;

        final_output = (final_output.broadcast_mul(&alpha)? + out_i.broadcast_mul(&beta)?)?;
        current_exp_sum = (current_exp_sum.broadcast_mul(&alpha)? + next_exp_sum.broadcast_mul(&beta)?)?;
        current_max_logit = new_max_logit;
    }

    // 루프가 다 끝난 후 마지막에 딱 한 번 정규화
    Ok(final_output.broadcast_div(&current_exp_sum)?)
}

pub fn block_wise_attention(
    query_states: &Tensor,
    k_blocks: &[Tensor],
    v_blocks: &[Tensor],
    num_kv_groups: usize,
    scaling: f64,
    attention_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (_b_sz, _n_heads, _q_len, _d_head) = query_states.dims4()?;
    let mut final_out: Option<Tensor> = None;
    let mut max_logits_acc: Option<Tensor> = None;
    let mut sum_exp_acc: Option<Tensor> = None;
    let mut current_kv_offset = 0;

    // [CRITICAL FIX] 루프 진입 전에 딱 한 번만 캐스팅합니다! 
    let q_aligned = if !k_blocks.is_empty() {
        query_states.to_dtype(k_blocks[0].dtype())?
    } else {
        query_states.clone()
    };

    for (k_block, v_block) in k_blocks.iter().zip(v_blocks.iter()) {
        let block_len = k_block.dim(2)?;
        // GQA support: repeat KV heads if needed
        let (mut k, mut v) = (k_block.clone(), v_block.clone());
        if num_kv_groups > 1 {
            let (b, h, s, d) = k.dims4()?;
            k = k.unsqueeze(2)?.expand((b, h, num_kv_groups, s, d))?.reshape((b, h * num_kv_groups, s, d))?;
            v = v.unsqueeze(2)?.expand((b, h, num_kv_groups, s, d))?.reshape((b, h * num_kv_groups, s, d))?;
        }

        // Attn Scores: Q @ K^T
        let mut attn_weights = (q_aligned.matmul(&k.transpose(2, 3)?)? * scaling)?;

        // Apply Mask if present
        if let Some(mask) = attention_mask {
            let m_len = mask.dim(D::Minus1)?;
            if current_kv_offset < m_len {
                let take = (m_len - current_kv_offset).min(block_len);
                let m_sub = mask.narrow(D::Minus1, current_kv_offset, take)?;
                
                if take < block_len {
                    let left_masked = attn_weights.narrow(D::Minus1, 0, take)?
                        .broadcast_add(&m_sub.to_dtype(attn_weights.dtype())?)?;
                    let right_unmasked = attn_weights.narrow(D::Minus1, take, block_len - take)?;
                    
                    attn_weights = Tensor::cat(&[&left_masked, &right_unmasked], D::Minus1)?;
                } else {
                    attn_weights = attn_weights.broadcast_add(&m_sub.to_dtype(attn_weights.dtype())?)?;
                }
            }
        }

        // Online Softmax Logic (Safe Softmax)
        let attn_weights_f32 = attn_weights.to_dtype(DType::F32)?;
        let max_logits = attn_weights_f32.max_keepdim(D::Minus1)?;
        let exp_weights = attn_weights_f32.broadcast_sub(&max_logits)?.exp()?;
        let sum_exp = exp_weights.sum_keepdim(D::Minus1)?;
        
        let out_block = exp_weights.to_dtype(v.dtype())?.matmul(&v)?;

        if let (Some(prev_out), Some(prev_max), Some(prev_sum)) = (final_out, max_logits_acc, sum_exp_acc) {
            let new_max = prev_max.broadcast_maximum(&max_logits)?;
            let exp_p = prev_max.broadcast_sub(&new_max)?.exp()?;
            let exp_n = max_logits.broadcast_sub(&new_max)?.exp()?;
            
            let new_sum = (prev_sum.broadcast_mul(&exp_p)? + sum_exp.broadcast_mul(&exp_n)?)?;
            let new_out = (prev_out.broadcast_mul(&exp_p.to_dtype(prev_out.dtype())?)? 
                         + out_block.broadcast_mul(&exp_n.to_dtype(out_block.dtype())?)?)?;
            
            final_out = Some(new_out);
            max_logits_acc = Some(new_max);
            sum_exp_acc = Some(new_sum);
        } else {
            final_out = Some(out_block);
            max_logits_acc = Some(max_logits);
            sum_exp_acc = Some(sum_exp);
        }
        current_kv_offset += block_len;
    }

    // Final normalization
    let res = final_out.as_ref().unwrap().broadcast_div(&sum_exp_acc.unwrap().to_dtype(final_out.as_ref().unwrap().dtype())?)?;
    Ok(res)
}

pub fn eager_attention_forward(
    query_states: &Tensor,
    key_states: &Tensor,
    value_states: &Tensor,
    _num_key_value_groups: Option<usize>,
    attention_mask: Option<&Tensor>,
    scaling: f64,
) -> Result<Tensor> {
    // [FLASH-DECODING] 5만 토큰 이상의 초장문 문맥을 위한 블록 단위 어텐션 최적화
    let (_b_sz, _n_heads, _q_len, _d_head) = query_states.dims4()?;
    let kv_seq_len = key_states.dim(2)?;
    
    // [CRITICAL FIX] 메모리 폭발을 막기 위해 여기서 전체 시퀀스에 대해 repeat_kv를 수행하던 로직을 통째로 삭제합니다!
    
    // 블록 크기 설정 (GPU SM 효율 및 VRAM 고려)
    let block_size = 4096;
    
    // 일반적인 짧은 문장이나 Flash-Attn 지원 시 기존 방식 사용
    #[cfg(feature = "flash-attn")]
    {
        // [CRITICAL FIX] Flash Attention은 F32 및 FP8을 네이티브로 지원하지 않습니다.
        // KV 캐시가 FP8이거나 모델이 F32로 동작 중일 경우 강제로 BF16으로 캐스팅하여 하드웨어 크래시를 방지합니다.
        let target_dtype = if query_states.device().is_cuda() { candle_core::DType::BF16 } else { query_states.dtype() };
        
        let q_aligned = query_states.to_dtype(target_dtype)?.transpose(1, 2)?.contiguous()?;
        let k_aligned = key_states.to_dtype(target_dtype)?.transpose(1, 2)?.contiguous()?;
        let v_aligned = value_states.to_dtype(target_dtype)?.transpose(1, 2)?.contiguous()?;

        let attn_output = candle_flash_attn::flash_attn(
            &q_aligned,
            &k_aligned,
            &v_aligned,
            scaling as f32,
            attention_mask.is_some(),
        )?
        .transpose(1, 2)?;
        
        // 연산 완료 후 다시 원래 데이터 타입(FP8/F32 등)으로 복구하여 이후 파이프라인의 타입 불일치 방지
        return Ok(attn_output.to_dtype(query_states.dtype())?.transpose(1, 2)?.contiguous()?);
    }

    if kv_seq_len <= block_size {
        let q_aligned = query_states.to_dtype(key_states.dtype())?;
        let attn_weights = q_aligned.matmul(&key_states.transpose(D::Minus2, D::Minus1)?)?;
        let attn_weights = (attn_weights * scaling)?;
        
        let attn_weights = match attention_mask {
            None => attn_weights,
            Some(mask) => {
                // [CRITICAL FIX] 거대한 어텐션 행렬을 F32로 복사하는 끔찍한 병목을 제거하고 즉시 덧셈!
                let mask_aligned = mask.to_dtype(attn_weights.dtype())?;
                attn_weights.broadcast_add(&mask_aligned)?
            }
        };
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.to_dtype(value_states.dtype())?.matmul(&value_states)?;
        return Ok(attn_output.transpose(1, 2)?.contiguous()?);
    }

    // --- Flash-Decoding 병렬 연산 구간 ---
    let num_blocks = (kv_seq_len + block_size - 1) / block_size;
    let mut block_outputs = Vec::with_capacity(num_blocks); 
    let mut block_lse = Vec::with_capacity(num_blocks); 

    // [CRITICAL FIX] 블록을 순회하기 전에 미리 1번만 캐스팅해둡니다!
    let q_aligned = query_states.to_dtype(key_states.dtype())?;

    for i in 0..num_blocks {
        let start = i * block_size;
        let end = (start + block_size).min(kv_seq_len); 
        let k_block = key_states.narrow(2, start, end - start)?;
        let v_block = value_states.narrow(2, start, end - start)?; 

        let attn_weights = (q_aligned.matmul(&k_block.transpose(2, 3)?)? * scaling)?; 
        
        // 마스크 처리
        let attn_weights = if let Some(mask) = attention_mask {
            let m_len = mask.dim(D::Minus1)?;
            if start < m_len {
                let m_end = end.min(m_len);
                let sub_mask = mask.narrow(D::Minus1, start, m_end - start)?;
                
                // [CRITICAL FIX] F32 캐스팅 역주행을 지우고 BF16 상태 그대로 초고속 덧셈!
                attn_weights.broadcast_add(&sub_mask)?
            } else {
                attn_weights
            }
        } else {
            attn_weights
        };

        // 수치적 안정성을 위한 Max 로짓 추출 및 통합 준비
        let max_logits = attn_weights.max_keepdim(D::Minus1)?;
        
        
        let safe_floor = Tensor::new(-10000.0_f32, max_logits.device())?
            .to_dtype(max_logits.dtype())?
            .broadcast_as(max_logits.shape())?;
        let max_logits_safe = max_logits.maximum(&safe_floor)?;

        let exp_weights = attn_weights.broadcast_sub(&max_logits_safe)?.exp()?;
        let sum_exp = exp_weights.sum_keepdim(D::Minus1)?;
        
        let out_block = exp_weights.to_dtype(v_block.dtype())?.matmul(&v_block)?;

        block_outputs.push(out_block);
        block_lse.push((sum_exp, max_logits));
    }

    // 블록 결과 통합 (Merging)
    let (mut current_exp_sum, mut current_max_logit) = block_lse[0].clone();
    let mut final_output = block_outputs[0].clone();

    for i in 1..num_blocks {
        let (next_exp_sum, next_max_logit) = &block_lse[i];
        let out_i = &block_outputs[i];

        let new_max_logit = current_max_logit.broadcast_maximum(next_max_logit)?;
        let exp_a = current_max_logit.broadcast_sub(&new_max_logit)?.exp()?;
        let exp_b = next_max_logit.broadcast_sub(&new_max_logit)?.exp()?;

        // [CRITICAL FIX] F32로 계산된 exp_a와 exp_b를 원본 BF16 타입으로 되돌린 후 곱셈(Mul) 수행!
        let exp_a_cast = exp_a.to_dtype(final_output.dtype())?;
        let exp_b_cast = exp_b.to_dtype(out_i.dtype())?;

        final_output = (final_output.broadcast_mul(&exp_a_cast)? + out_i.broadcast_mul(&exp_b_cast)?)?;
        current_exp_sum = (current_exp_sum.broadcast_mul(&exp_a)? + next_exp_sum.broadcast_mul(&exp_b)?)?;
        current_max_logit = new_max_logit;
    }

    // 3. 루프가 다 끝난 후 마지막에 딱 한 번 정규화!
    let final_output = final_output.broadcast_div(&current_exp_sum)?;
    let attn_output = final_output.transpose(1, 2)?.contiguous()?;

    Ok(attn_output)
}

pub fn get_conv2d(
    vb: VarBuilder,
    in_c: usize,
    out_c: usize,
    kernel_size: usize,
    padding: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
    bias: bool,
) -> Result<Conv2d> {
    let cfg = Conv2dConfig {
        padding,
        stride,
        dilation,
        groups,
        cudnn_fwd_algo: None,
    };
    let conv2d = if bias {
        conv2d(in_c, out_c, kernel_size, cfg, vb)?
    } else {
        conv2d_no_bias(in_c, out_c, kernel_size, cfg, vb)?
    };
    Ok(conv2d)
}

pub fn get_layer_norm(vb: VarBuilder, eps: f64, dim: usize) -> Result<LayerNorm> {
    let ln_config = LayerNormConfig {
        eps,
        remove_mean: true, // true for layernorm, false for RMSNorm
        affine: true,      // true for with bias, false for without bias
    };
    let norm = layer_norm(dim, ln_config, vb)?;
    Ok(norm)
}

pub fn get_batch_norm(vb: VarBuilder, eps: f64, dim: usize) -> Result<BatchNorm> {
    let bn_config = BatchNormConfig {
        eps,
        remove_mean: true,
        affine: true,
        momentum: 0.1,
    };
    let norm = batch_norm(dim, bn_config, vb)?;
    Ok(norm)
}

pub fn deform_conv2d_kernel(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    offset: &Tensor,
    mask: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    // 不考虑空洞卷积, bs = 1
    let (_, in_c, in_h, in_w) = input.dims4()?;
    let (out_channel, _, ker_h, ker_w) = weight.dims4()?;
    let out_h = ((in_h + 2 * padding - ker_h) / stride) + 1;
    let out_w = ((in_w + 2 * padding - ker_w) / stride) + 1;

    let num_kernels = in_c * out_h * out_w;
    let mask_vec = if let Some(mask) = mask {
        Some(mask.squeeze(0)?.to_vec3::<f32>()?)
    } else {
        None
    };
    let offset_vec = offset.squeeze(0)?.to_vec3::<f32>()?;
    let input_vec = input.squeeze(0)?.to_vec3::<f32>()?;
    let mut columns_vec = vec![vec![0.0f32; out_h * out_w]; in_c * ker_h * ker_w];
    for index in 0..num_kernels {
        let out_x = index % out_w;
        let out_y = (index / out_w) % out_h;
        let in_c = index / (out_w * out_h);
        let out_c = in_c * ker_h * ker_w;

        for i in 0..ker_h {
            for j in 0..ker_w {
                let mask_idx = i * ker_w + j;
                let offset_idx = 2 * mask_idx;
                let mask_value = if mask.is_some() {
                    mask_vec.as_ref().unwrap()[mask_idx][out_y][out_x]
                } else {
                    1.0
                };
                let offset_h = offset_vec[offset_idx][out_y][out_x];
                let offset_w = offset_vec[offset_idx + 1][out_y][out_x];
                let y = ((out_y * stride - padding) + i) as f32 + offset_h;
                let x = ((out_x * stride - padding) + j) as f32 + offset_w;
                let val = if y <= -1.0 || in_h as f32 <= y || x <= -1.0 || in_w as f32 <= x {
                    0.0
                } else {
                    let h_low = y.floor();
                    let w_low = x.floor();
                    let h_high = h_low + 1.0;
                    let w_high = w_low + 1.0;
                    let lh = y - h_low;
                    let lw = x - w_low;
                    let hh = 1.0 - lh;
                    let hw = 1.0 - lw;
                    let w1 = hh * hw;
                    let w2 = hh * lw;
                    let w3 = lh * hw;
                    let w4 = lh * lw;
                    let v1 = if h_low >= 0.0 && w_low >= 0.0 {
                        input_vec[in_c][h_low as usize][w_low as usize]
                    } else {
                        0.0
                    };
                    let v2 = if h_low >= 0.0 && w_high <= (in_w - 1) as f32 {
                        input_vec[in_c][h_low as usize][w_high as usize]
                    } else {
                        0.0
                    };
                    let v3 = if h_high <= (in_h - 1) as f32 && w_low >= 0.0 {
                        input_vec[in_c][h_high as usize][w_low as usize]
                    } else {
                        0.0
                    };
                    let v4 = if h_high <= (in_h - 1) as f32 && w_high <= (in_w - 1) as f32 {
                        input_vec[in_c][h_high as usize][w_high as usize]
                    } else {
                        0.0
                    };
                    w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4
                };
                columns_vec[out_c + i * ker_w + j][out_y * out_w + out_x] = mask_value * val;
            }
        }
    }

    let columns = Tensor::new(columns_vec, weight.device())?;
    let mut out =
        weight
            .flatten_from(1)?
            .matmul(&columns)?
            .reshape((1, out_channel, out_h, out_w))?;
    if let Some(bias) = bias {
        out = out.broadcast_add(bias)?;
    }
    Ok(out)
}
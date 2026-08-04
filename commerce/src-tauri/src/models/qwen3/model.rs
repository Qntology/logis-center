use anyhow::{Result, anyhow};
use candle_core::Tensor;
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_b, linear_no_bias, rms_norm,
};

use crate::{
    models::{
        common::{GateUpDownMLP, QKNormAttention, eager_attention_forward},
        qwen3::config::Qwen3Config,
    },
    position_embed::rope::{RoPE, apply_rotary_pos_emb},
    utils::tensor_utils::prepare_causal_attention_mask,
};

pub struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    pub kv_cache: Option<(Tensor, Tensor)>,
}

impl Qwen3Attention {
    pub fn new(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_attention_heads = config.num_attention_heads;
        let head_dim = config.head_dim;
        let num_key_value_heads = config.num_key_value_heads;
        let num_kv_groups = num_attention_heads / num_key_value_heads;
        let scaling = 1f64 / f64::sqrt(head_dim as f64);
        let q_proj = linear_b(
            hidden_size,
            num_attention_heads * head_dim,
            config.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            hidden_size,
            num_key_value_heads * head_dim,
            config.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            hidden_size,
            num_key_value_heads * head_dim,
            config.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_attention_heads * head_dim,
            hidden_size,
            config.attention_bias,
            vb.pp("o_proj"),
        )?;
        let q_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(head_dim, config.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_attention_heads,
            num_key_value_heads,
            num_kv_groups,
            head_dim,
            scaling,
            kv_cache: None,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let query_states = self.q_proj.forward(xs)?.reshape((
            b_sz,
            q_len,
            self.num_attention_heads,
            self.head_dim,
        ))?;
        let query_states = self.q_norm.forward(&query_states)?.transpose(1, 2)?;
        let key_states = self.k_proj.forward(xs)?.reshape((
            b_sz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
        ))?;
        let key_states = self.k_norm.forward(&key_states)?.transpose(1, 2)?;
        let value_states = self.v_proj.forward(xs)?;
        let value_states = value_states
            .reshape((b_sz, q_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (query_states, key_states) =
            apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;
        
        // 🌟 [CRITICAL FIX] 동일하게 소유권(take)을 빼앗아 VRAM 2배 널뛰기 스파이크를 완전히 제거합니다.
        let (key_states, value_states) = match self.kv_cache.take() {
            None => (key_states.contiguous()?, value_states.contiguous()?),
            Some((prev_k, prev_v)) => {
                let dev = key_states.device();
                let prev_k = if !prev_k.device().same_device(dev) { prev_k.to_device(dev)? } else { prev_k };
                let prev_v = if !prev_v.device().same_device(dev) { prev_v.to_device(dev)? } else { prev_v };
                let k = Tensor::cat(&[&prev_k, &key_states], 2)?.contiguous()?;
                let v = Tensor::cat(&[&prev_v, &value_states], 2)?.contiguous()?;
                (k, v)
            }
        };
        
        // 🌟 [JIT CPU Offload] 동일하게 보관용 캐시를 연산 직후 즉각 RAM으로 대피시켜 VRAM 스파이크를 없앱니다.
        self.kv_cache = Some((
            key_states.to_device(&candle_core::Device::Cpu)?, 
            value_states.to_device(&candle_core::Device::Cpu)?
        ));
        let attn_output = eager_attention_forward(
            &query_states,
            &key_states,
            &value_states,
            Some(self.num_kv_groups),
            attention_mask,
            self.scaling,
        )?;
        let attn_output =
            attn_output.reshape((b_sz, q_len, self.num_attention_heads * self.head_dim))?;
        let attn_output = attn_output.apply(&self.o_proj)?;
        Ok(attn_output)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Clone)]
pub struct Fp8VramKVCache {
    pub k_fp8: Tensor,
    pub v_fp8: Tensor,
}

pub struct Qwen3DecoderLayer {
    // self_attn: Qwen3Attention,
    self_attn: QKNormAttention,
    mlp: GateUpDownMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pub fp8_cache: Option<Fp8VramKVCache>, // 🌟 [FP8 Compression] RAM으로 내리지 않고 VRAM 내에서 FP8로 압축하여 상주
}

impl Qwen3DecoderLayer {
    pub fn new(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        // let self_attn = Qwen3Attention::new(config, vb.pp("self_attn"))?;
        let self_attn = QKNormAttention::new(
            vb.pp("self_attn"),
            config.hidden_size,
            config.num_attention_heads,
            Some(config.head_dim),
            Some(config.num_key_value_heads),
            config.attention_bias,
            config.rms_norm_eps,
            None,
            None,
            None,
            None,
            None,
            None,
        )?;
        let mlp = GateUpDownMLP::new(
            vb.pp("mlp"),
            config.hidden_size,
            config.intermediate_size,
            config.hidden_act,
            false,
            None,
            None,
            None,
        )?;
        let input_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("input_layernorm"),
        )?;
        let post_attention_layernorm = rms_norm(
            config.hidden_size,
            config.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            fp8_cache: None,
        })
    }


    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // 🌟 [FP8 Compression] VRAM에 보관 중이던 FP8 텐서를 연산을 위해 즉시 BF16/F32로 압축 해제합니다.
        if self.self_attn.kv_cache.is_none() {
            if let Some(cache) = self.fp8_cache.take() {
                let target_dtype = if xs.device().is_cuda() { candle_core::DType::BF16 } else { candle_core::DType::F32 };
                
                // 🌟 [CRITICAL FIX] CUDA 환경에서 FP8(F8E4M3) 형변환 커널이 없을 경우 프로그램이 터지는 현상(CUDA_ERROR_NOT_FOUND)을 막기 위해 안전한 복구(Fallback) 로직을 적용합니다.
                let k_restored = cache.k_fp8.to_dtype(target_dtype).unwrap_or_else(|_| cache.k_fp8.clone());
                let v_restored = cache.v_fp8.to_dtype(target_dtype).unwrap_or_else(|_| cache.v_fp8.clone());
                
                self.self_attn.kv_cache = Some((k_restored, v_restored));
            }
        }

        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, cos, sin, attention_mask)?;
        
        // 🌟 [FP8 KV Cache] 레이어 연산이 끝나는 즉시, VRAM 코어를 사용해 초고속 FP8 압축 상태로 VRAM에 보존합니다.
        // RAM-VRAM 스왑 없이 순수 VRAM 내에서 용량을 50% 절약합니다.
        self.compress_kv_in_vram()?;
        
        let xs = residual.add(&xs)?;
        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = residual.add(&xs)?;
        Ok(xs)
    }

    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
        self.fp8_cache = None;
    }

    pub fn compress_kv_in_vram(&mut self) -> Result<()> {
        if let Some((k, v)) = self.self_attn.kv_cache.take() {
            // 🌟 [CRITICAL FIX] VRAM 내부에서 FP8(F8E4M3) 압축 시도 시, 해당 드라이버 심볼이 없으면(CUDA_ERROR_NOT_FOUND) 원본(BF16/F32)을 그대로 보존하여 에러를 원천 차단합니다.
            let k_fp8 = k.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| k.clone());
            let v_fp8 = v.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| v.clone());
            
            self.fp8_cache = Some(Fp8VramKVCache {
                k_fp8,
                v_fp8,
            });
        }
        Ok(())
    }

    pub fn get_kv_cache(&self) -> Option<(Tensor, Tensor)> {
        if let Some(cache) = &self.self_attn.kv_cache {
            Some(cache.clone())
        } else if let Some(fp8_cache) = &self.fp8_cache {
            // 스냅샷 저장을 위해 호출될 경우, 원래 타입으로 복원하여 반환
            let target_dtype = candle_core::DType::F32; // 저장용 기본 타입
            if let Ok(k) = fp8_cache.k_fp8.to_dtype(target_dtype) {
                if let Ok(v) = fp8_cache.v_fp8.to_dtype(target_dtype) {
                    return Some((k, v));
                }
            }
            None
        } else {
            None
        }
    }

    pub fn set_kv_cache(&mut self, cache: Option<(Tensor, Tensor)>) {
        self.self_attn.kv_cache = cache;
        self.fp8_cache = None;
    }
}

pub struct Qwen3Model {
    embed_tokens: Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    rotary_emb: RoPE,
    lm_head: Linear,
}

impl Qwen3Model {
    pub fn new(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("model");
        let vocab_size = config.vocab_size;
        let embed_tokens = embedding(vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let mut layers = vec![];
        let vb_l = vb.pp("layers");
        for layer_idx in 0..config.num_hidden_layers {
            let layer = Qwen3DecoderLayer::new(config, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, vb.pp("norm"))?;
        let head_dim = config.head_dim;
        let rotary_emb = RoPE::new(head_dim, config.rope_theta, vb.device())?;
        let lm_head = if config.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("lm_head"))?
        };
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            lm_head,
        })
    }
    
    pub fn forward(
        &mut self,
        input_ids: Option<&Tensor>,
        inputs_embeds: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        if input_ids.is_none() && inputs_embeds.is_none() {
            return Err(anyhow::anyhow!(
                "You must specify exactly one of input_ids or inputs_embeds"
            ));
        }
        let inputs_embeds = if let Some(inputs_embeds) = inputs_embeds {
            inputs_embeds.clone()
        } else {
            let input_ids = input_ids.unwrap();
            self.embedding_token_id(input_ids)?
        };

        // 🌟 [VRAM 최적화] CUDA 환경일 때 입력 임베딩을 BF16으로 강제 캐스팅하여 전체 파이프라인의 VRAM 2배 폭식을 방어합니다.
        let target_dtype = if inputs_embeds.device().is_cuda() { candle_core::DType::BF16 } else { candle_core::DType::F32 };
        let inputs_embeds = if inputs_embeds.dtype() != target_dtype { inputs_embeds.to_dtype(target_dtype)? } else { inputs_embeds };

        let (bs, seq_len, _) = inputs_embeds.dims3()?;
        let attention_mask: Option<Tensor> = {
            if seq_len <= 1 {
                None
            } else {
                let mask = prepare_causal_attention_mask(
                    bs,
                    seq_len,
                    seqlen_offset, // 🌟 [CRITICAL FIX] 청크 분할로 인해 누적된 과거 토큰 길이만큼 마스크 크기를 동적으로 연장합니다.
                    inputs_embeds.device(),
                )?;
                Some(if mask.dtype() != target_dtype { mask.to_dtype(target_dtype)? } else { mask })
            }
        };

        let (cos, sin) = self
            .rotary_emb
            .forward(seqlen_offset, seq_len, inputs_embeds.device())?;
            
        // 🌟 [VRAM 최적화] RoPE 테이블(F32)과 Mask가 Attention 연산에 섞여 들어가 전체 텐서를 F32로 강제 승격시키는 현상을 원천 차단합니다.
        let cos = if cos.dtype() != target_dtype { cos.to_dtype(target_dtype)? } else { cos };
        let sin = if sin.dtype() != target_dtype { sin.to_dtype(target_dtype)? } else { sin };

        let mut hidden_states = inputs_embeds;
        for decode_layer in &mut self.layers {
            hidden_states =
                decode_layer.forward(&hidden_states, &cos, &sin, attention_mask.as_ref())?;
        }
        hidden_states = self.norm.forward(&hidden_states)?;
        let hidden_state = hidden_states.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&hidden_state)?;
        Ok(logits)
    }
    
    pub fn embedding_token_id(&self, input_ids: &Tensor) -> Result<Tensor> {
        Ok(self.embed_tokens.forward(input_ids)?)
    }

    // 🌟 [추가] Semantic Bias 연산을 위해 전체 단어장의 벡터(Weight)를 그대로 반환합니다.
    pub fn get_embed_tokens(&self) -> Tensor {
        self.embed_tokens.embeddings().clone()
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }

    pub fn compress_kv_in_vram(&mut self) -> Result<()> {
        for layer in self.layers.iter_mut() {
            layer.compress_kv_in_vram()?;
        }
        Ok(())
    }

    pub fn get_kv_cache(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.layers.iter().map(|l| l.get_kv_cache()).collect()
    }

    pub fn set_kv_cache(&mut self, cache: Vec<Option<(Tensor, Tensor)>>) {
        for (layer, c) in self.layers.iter_mut().zip(cache.into_iter()) {
            layer.set_kv_cache(c);
        }
    }
}

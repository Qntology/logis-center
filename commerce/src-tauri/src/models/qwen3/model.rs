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


pub struct Qwen3DecoderLayer {
    self_attn: QKNormAttention,
    mlp: GateUpDownMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    // 🌟 [BLOCK-KV] 전체 캐시 왕복(Fp8VramKVCache)을 제거하고, qwen과 동일한 1024토큰 블록 시스템으로 교체합니다.
    // VRAM에는 활성 블록(≤1024토큰)만 BF16으로 상주, 과거 블록은 FP8로 RAM에 동결됩니다.
    pub kv_blocks: Vec<crate::models::qwen::quantized_model::KVBlock>,
    pub registry: crate::models::qwen::quantized_model::KVRegistry,
    pub layer_idx: usize,
}

impl Qwen3DecoderLayer {
    pub fn new(config: &Qwen3Config, vb: VarBuilder, layer_idx: usize, registry: crate::models::qwen::quantized_model::KVRegistry) -> Result<Self> {
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
            kv_blocks: Vec::new(),
            registry,
            layer_idx,
        })
    }


    pub fn forward(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs.clone();
        let xs = self.input_layernorm.forward(xs)?;
        // 🌟 [BLOCK-KV] 전체 캐시 압축/해제 없이 블록 단위 어텐션 수행
        let xs = self.forward_block_attention(&xs, cos, sin, attention_mask, seqlen_offset)?;
        let xs = residual.add(&xs)?;
        let residual = xs.clone();
        let xs = self.post_attention_layernorm.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        let xs = residual.add(&xs)?;
        Ok(xs)
    }

    /// 🌟 [BLOCK-KV] qwen의 quantized_model.rs와 동일한 1024토큰 블록 파이프라인입니다.
    /// VRAM에는 활성 블록만 BF16으로 존재하고, 과거 블록은 FP8로 RAM에 동결됩니다.
    /// 토큰당 비용: O(활성블록 1024) + O(과거블록 읽기) — 압축/해제 왕복 없음.
    fn forward_block_attention(
        &mut self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        use crate::models::qwen::quantized_model::{KVBlock, KVLocation};
        let (b_sz, q_len, _) = xs.dims3()?;
        let dev = xs.device();
        let target_dtype = if dev.is_cuda() { candle_core::DType::BF16 } else { candle_core::DType::F32 };

        // 1. Q/K/V 투영 + RoPE
        let query_states = self.self_attn.q_proj.forward(xs)?
            .reshape((b_sz, q_len, self.self_attn.num_attention_heads, self.self_attn.head_dim))?;
        let query_states = self.self_attn.q_norm.forward(&query_states)?.transpose(1, 2)?.contiguous()?;
        let key_states = self.self_attn.k_proj.forward(xs)?
            .reshape((b_sz, q_len, self.self_attn.num_key_value_heads, self.self_attn.head_dim))?;
        let key_states = self.self_attn.k_norm.forward(&key_states)?.transpose(1, 2)?.contiguous()?;
        let value_states = self.self_attn.v_proj.forward(xs)?
            .reshape((b_sz, q_len, self.self_attn.num_key_value_heads, self.self_attn.head_dim))?
            .transpose(1, 2)?.contiguous()?;
        let (query_states, key_states) = apply_rotary_pos_emb(&query_states, &key_states, cos, sin, false)?;
        let query_states = query_states.to_dtype(target_dtype)?.contiguous()?;
        let key_states = key_states.to_dtype(target_dtype)?.contiguous()?;

        // 2. [BLOCK-APPEND] 활성 블록에 추가 or 새 블록 생성
        let mut tokens_to_process = q_len;
        let mut chunk_offset = 0;
        while tokens_to_process > 0 {
            let mut appended = false;
            if let Some(last_block) = self.kv_blocks.last_mut() {
                let mut inner = last_block.inner.write().unwrap();
                let free_space = 1024usize.saturating_sub(inner.len);
                if inner.location == KVLocation::VRAM && free_space > 0 {
                    let take = tokens_to_process.min(free_space);
                    let k_piece = key_states.narrow(2, chunk_offset, take)?;
                    let v_piece = value_states.narrow(2, chunk_offset, take)?;
                    if let (Some(pk), Some(pv)) = (inner.k_cache.take(), inner.v_cache.take()) {
                        let pk_f = pk.to_dtype(target_dtype).unwrap_or_else(|_| pk.clone());
                        let pv_f = pv.to_dtype(target_dtype).unwrap_or_else(|_| pv.clone());
                        let cat_k = Tensor::cat(&[&pk_f, &k_piece], 2)?;
                        let cat_v = Tensor::cat(&[&pv_f, &v_piece], 2)?;
                        inner.k_cache = Some(if dev.is_cuda() { cat_k.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| cat_k.clone()) } else { cat_k });
                        inner.v_cache = Some(if dev.is_cuda() { cat_v.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| cat_v.clone()) } else { cat_v });
                        inner.len += take;
                        tokens_to_process -= take;
                        chunk_offset += take;
                        appended = true;
                        let mut reg = self.registry.entries.write().unwrap();
                        if inner.index < reg.len() {
                            reg[inner.index].token_len = inner.len;
                            if self.layer_idx < reg[inner.index].is_dirty.len() { reg[inner.index].is_dirty[self.layer_idx] = true; }
                        }
                    }
                }
            }
            if !appended {
                let take = tokens_to_process.min(1024);
                let k_piece = key_states.narrow(2, chunk_offset, take)?.contiguous()?;
                let v_piece = value_states.narrow(2, chunk_offset, take)?.contiguous()?;
                let index = self.kv_blocks.len();
                let current_total = seqlen_offset + chunk_offset;
                let new_block = KVBlock::new(KVLocation::VRAM, index, take, current_total);
                {
                    let mut inner = new_block.inner.write().unwrap();
                    inner.k_cache = Some(if dev.is_cuda() { k_piece.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| k_piece.clone()) } else { k_piece });
                    inner.v_cache = Some(if dev.is_cuda() { v_piece.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| v_piece.clone()) } else { v_piece });
                }
                let mut reg = self.registry.entries.write().unwrap();
                if index < reg.len() {
                    reg[index].token_start = current_total;
                    reg[index].token_len = take;
                    if self.layer_idx < reg[index].is_dirty.len() { reg[index].is_dirty[self.layer_idx] = true; }
                    reg[index].location[self.layer_idx] = KVLocation::VRAM;
                }
                self.kv_blocks.push(new_block);
                tokens_to_process -= take;
                chunk_offset += take;
            }
        }

        // 3. [BLOCK-ATTENTION] 온라인 소프트맥스로 블록 순회 (qwen과 동일)
        let total_tokens_now = seqlen_offset + q_len;
        let mut out_res: Option<Tensor> = None;
        let mut m_n: Option<Tensor> = None;
        let mut l_n: Option<Tensor> = None;
        let q_aligned = query_states.to_dtype(target_dtype)?.contiguous()?;
        let (q_b, q_h, q_l, q_d) = q_aligned.dims4()?;
        let kv_h = self.self_attn.num_key_value_heads;
        let q_folded = if self.self_attn.num_kv_groups > 1 {
            q_aligned.reshape((q_b, kv_h, self.self_attn.num_kv_groups * q_l, q_d))?
        } else {
            q_aligned.clone()
        };

        for block in &self.kv_blocks {
            let (index, b_off, _b_len) = {
                let inner = block.inner.read().unwrap();
                (inner.index, inner.offset, inner.len)
            };
            if b_off >= total_tokens_now { continue; }
            let (k_block, v_block) = {
                let mut inner = block.inner.write().unwrap();
                if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                    if inner.location == KVLocation::VRAM {
                        (k.to_dtype(target_dtype).unwrap_or_else(|_| k.clone()), v.to_dtype(target_dtype).unwrap_or_else(|_| v.clone()))
                    } else {
                        (k.to_device(dev)?.to_dtype(target_dtype)?, v.to_device(dev)?.to_dtype(target_dtype)?)
                    }
                } else {
                    let fallback_shape = vec![1, kv_h, _b_len, self.self_attn.head_dim];
                    (Tensor::zeros(fallback_shape.as_slice(), target_dtype, dev)?, Tensor::zeros(fallback_shape.as_slice(), target_dtype, dev)?)
                }
            };
            let k = k_block.contiguous()?;
            let v = v_block.contiguous()?;
            let actual_kv_len = k.dim(2)?;
            let k_t = k.transpose(2, 3)?;
            let s_folded = (q_folded.matmul(&k_t)? * self.self_attn.scaling)?;
            let mut s_chunk = if self.self_attn.num_kv_groups > 1 {
                s_folded.reshape((q_b, q_h, q_l, actual_kv_len))?
            } else {
                s_folded
            };
            if let Some(mask) = attention_mask {
                let mask_len = mask.dim(candle_core::D::Minus1)?;
                if b_off < mask_len {
                    let take = std::cmp::min(actual_kv_len, mask_len - b_off);
                    let chunk_mask = mask.narrow(candle_core::D::Minus1, b_off, take)?;
                    if take < actual_kv_len {
                        let left_masked = s_chunk.narrow(candle_core::D::Minus1, 0, take)?.broadcast_add(&chunk_mask.to_dtype(target_dtype)?)?;
                        let right_unmasked = s_chunk.narrow(candle_core::D::Minus1, take, actual_kv_len - take)?;
                        s_chunk = Tensor::cat(&[&left_masked, &right_unmasked], candle_core::D::Minus1)?;
                    } else {
                        s_chunk = s_chunk.broadcast_add(&chunk_mask.to_dtype(target_dtype)?)?;
                    }
                }
            }
            let s_chunk_f32 = s_chunk.to_dtype(DType::F32)?;
            let m_j = s_chunk_f32.max_keepdim(candle_core::D::Minus1)?;
            let safe_floor = Tensor::new(-10000.0_f32, m_j.device())?.broadcast_as(m_j.shape())?;
            let m_j_safe = m_j.maximum(&safe_floor)?;
            let p_j = s_chunk_f32.broadcast_sub(&m_j_safe)?.exp()?;
            let l_j = p_j.sum_keepdim(candle_core::D::Minus1)?;
            let p_v = p_j.to_dtype(v.dtype())?.contiguous()?;
            let p_folded = if self.self_attn.num_kv_groups > 1 {
                p_v.reshape((q_b, kv_h, self.self_attn.num_kv_groups * q_l, actual_kv_len))?
            } else {
                p_v
            };
            let out_folded = p_folded.matmul(&v)?;
            let out_j = if self.self_attn.num_kv_groups > 1 {
                out_folded.reshape((q_b, q_h, q_l, self.self_attn.head_dim))?
            } else {
                out_folded
            };
            let out_j_f32 = out_j.to_dtype(DType::F32)?;
            match out_res {
                None => { out_res = Some(out_j_f32); m_n = Some(m_j); l_n = Some(l_j); }
                Some(prev_out_f32) => {
                    let prev_m = m_n.as_ref().unwrap();
                    let prev_l = l_n.as_ref().unwrap();
                    let m_new = prev_m.maximum(&m_j_safe)?;
                    let diff_old = prev_m.broadcast_sub(&m_new)?.exp()?;
                    let diff_new = m_j_safe.broadcast_sub(&m_new)?.exp()?;
                    let l_new = prev_l.broadcast_mul(&diff_old)?.add(&l_j.broadcast_mul(&diff_new)?)?;
                    let out_new_f32 = prev_out_f32.broadcast_mul(&diff_old)?.add(&out_j_f32.broadcast_mul(&diff_new)?)?;
                    out_res = Some(out_new_f32); m_n = Some(m_new); l_n = Some(l_new);
                }
            }
            drop(k);
            drop(v);
        }

        let attn_output = if let (Some(out_f32), Some(l_f32)) = (out_res, l_n) {
            out_f32.broadcast_div(&l_f32)?.to_dtype(target_dtype)?
        } else {
            return Err(anyhow::anyhow!("No KV data processed"));
        };
        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, q_len, self.self_attn.num_attention_heads * self.self_attn.head_dim))?;
        let attn_output = attn_output.apply(&self.self_attn.o_proj)?;
        Ok(attn_output)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_blocks.clear();
    }

    /// 🌟 [BLOCK-KV] 전체 캐시 왕복 압축/해제 함수를 삭제합니다.
    /// 블록 시스템에서는 활성 블록만 VRAM에 있고, 과거 블록은 이미 FP8로 RAM에 동결되어 있습니다.
    /// 더 이상 매 토큰마다 전체를 압축/해제할 필요가 없습니다.

    pub fn get_kv_cache(&self) -> Option<(Tensor, Tensor)> {
        let mut ks = Vec::new();
        let mut vs = Vec::new();
        for block in &self.kv_blocks {
            let inner = block.inner.read().unwrap();
            if let (Some(k), Some(v)) = (&inner.k_cache, &inner.v_cache) {
                ks.push(k.to_dtype(DType::F32).unwrap_or_else(|_| k.clone()));
                vs.push(v.to_dtype(DType::F32).unwrap_or_else(|_| v.clone()));
            }
        }
        if ks.is_empty() { return None; }
        let k_cat = Tensor::cat(&ks, 2).ok()?;
        let v_cat = Tensor::cat(&vs, 2).ok()?;
        Some((k_cat, v_cat))
    }

    pub fn set_kv_cache(&mut self, cache: Option<(Tensor, Tensor)>) {
        use crate::models::qwen::quantized_model::{KVBlock, KVLocation};
        self.kv_blocks.clear();
        if let Some((k, v)) = cache {
            let total_len = k.dim(2).unwrap_or(0);
            let mut offset = 0;
            let mut idx = 0;
            while offset < total_len {
                let take = (total_len - offset).min(1024);
                let k_piece = k.narrow(2, offset, take).unwrap();
                let v_piece = v.narrow(2, offset, take).unwrap();
                let new_block = KVBlock::new(KVLocation::VRAM, idx, take, offset);
                {
                    let mut inner = new_block.inner.write().unwrap();
                    inner.k_cache = Some(k_piece);
                    inner.v_cache = Some(v_piece);
                }
                self.kv_blocks.push(new_block);
                offset += take;
                idx += 1;
            }
        }
    }
}

pub struct Qwen3Model {
    embed_tokens: Embedding,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    rotary_emb: RoPE,
    lm_head: Linear,
    pub registry: crate::models::qwen::quantized_model::KVRegistry,
}

impl Qwen3Model {
    pub fn new(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("model");
        let vocab_size = config.vocab_size;
        let embed_tokens = embedding(vocab_size, config.hidden_size, vb.pp("embed_tokens"))?;
        let registry = crate::models::qwen::quantized_model::KVRegistry::new();
        let mut layers = vec![];
        let vb_l = vb.pp("layers");
        for layer_idx in 0..config.num_hidden_layers {
            let layer = Qwen3DecoderLayer::new(config, vb_l.pp(layer_idx), layer_idx, registry.clone())?;
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
            registry,
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
        for (layer_idx, decode_layer) in self.layers.iter_mut().enumerate() {
            hidden_states =
                decode_layer.forward(&hidden_states, &cos, &sin, attention_mask.as_ref(), seqlen_offset)?;
            // 🌟 [VRAM-EVICT] 활성 블록 8개 초과 시 가장 오래된 블록을 FP8로 RAM에 동결
            // qwen의 evacuate_vram_to_ram_only와 동일한 로직 — VRAM 상한 8블록 유지
            let vram_limit = 8;
            let mut vram_indices = Vec::new();
            for (idx, block) in decode_layer.kv_blocks.iter().enumerate() {
                let inner = block.inner.read().unwrap();
                if inner.location == crate::models::qwen::quantized_model::KVLocation::VRAM {
                    vram_indices.push((idx, inner.offset));
                }
            }
            if vram_indices.len() > vram_limit {
                vram_indices.sort_by_key(|k| k.1);
                let num_to_evict = vram_indices.len().saturating_sub(vram_limit);
                for i in 0..num_to_evict {
                    let (idx, _) = vram_indices[i];
                    let mut inner = decode_layer.kv_blocks[idx].inner.write().unwrap();
                    if let (Some(k), Some(v)) = (inner.k_cache.take(), inner.v_cache.take()) {
                        let target_dtype = if k.device().is_cuda() || k.dtype() == candle_core::DType::F8E4M3 { candle_core::DType::F8E4M3 } else { candle_core::DType::F32 };
                        inner.k_cache = Some(k.to_dtype(target_dtype).unwrap_or_else(|_| k.clone()).to_device(&candle_core::Device::Cpu).unwrap_or_else(|_| k.clone()));
                        inner.v_cache = Some(v.to_dtype(target_dtype).unwrap_or_else(|_| v.clone()).to_device(&candle_core::Device::Cpu).unwrap_or_else(|_| v.clone()));
                        inner.location = crate::models::qwen::quantized_model::KVLocation::RAM;
                        let mut reg = self.registry.entries.write().unwrap();
                        if idx < reg.len() { reg[idx].location[layer_idx] = crate::models::qwen::quantized_model::KVLocation::RAM; }
                    }
                }
            }
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

    pub fn get_kv_cache(&self) -> Vec<Option<(Tensor, Tensor)>> {
        self.layers.iter().map(|l| l.get_kv_cache()).collect()
    }

    pub fn set_kv_cache(&mut self, cache: Vec<Option<(Tensor, Tensor)>>) {
        for (layer, c) in self.layers.iter_mut().zip(cache.into_iter()) {
            layer.set_kv_cache(c);
        }
    }
}

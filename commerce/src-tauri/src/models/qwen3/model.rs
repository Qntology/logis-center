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

#[derive(Clone)]
pub struct Fp8VramKVCache {
    pub k_fp8: Tensor,
    pub v_fp8: Tensor,
    // 🌟 [KV RESIDENCY] 이 캐시가 물리적으로 어디에 놓여 있는지 기록합니다.
    //    VRAM 이면 그대로 두고, RAM 이면 연산 직전에만 GPU 로 올렸다 즉시 내립니다.
    pub location: crate::utils::resources::KvResidency,
}

pub struct Qwen3DecoderLayer {
    self_attn: QKNormAttention,
    mlp: GateUpDownMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    pub fp8_cache: Option<Fp8VramKVCache>, // 🌟 [FP8 Compression] RAM으로 내리지 않고 VRAM 내에서 FP8로 압축하여 상주
    // 🌟 [KV RESIDENCY] 디코딩 진입 시 1회 결정된 목표 배치 위치.
    //    기본값은 기존 동작과 100% 동일한 Vram 입니다.
    pub kv_residency: crate::utils::resources::KvResidency,
}

impl Qwen3DecoderLayer {
    pub fn new(config: &Qwen3Config, vb: VarBuilder) -> Result<Self> {
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
            kv_residency: crate::utils::resources::KvResidency::Vram,
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
                let dev = xs.device();
                let target_dtype = if dev.is_cuda() { candle_core::DType::BF16 } else { candle_core::DType::F32 };

                // 🌟 [KV RESIDENCY] RAM 에 대피해 있던 캐시라면 연산 직전에만 GPU 로 올립니다.
                //    PCIe 4.0 x16 기준 28레이어 × 28KB/token ≈ 784KB → 0.03ms 로 무시 가능한 비용입니다.
                let k_src = if cache.location == crate::utils::resources::KvResidency::Vram {
                    cache.k_fp8.clone()
                } else {
                    cache.k_fp8.to_device(dev).unwrap_or_else(|_| cache.k_fp8.clone())
                };
                let v_src = if cache.location == crate::utils::resources::KvResidency::Vram {
                    cache.v_fp8.clone()
                } else {
                    cache.v_fp8.to_device(dev).unwrap_or_else(|_| cache.v_fp8.clone())
                };

                // 🌟 [CRITICAL FIX] CUDA 환경에서 FP8(F8E4M3) 형변환 커널이 없을 경우 프로그램이 터지는 현상(CUDA_ERROR_NOT_FOUND)을 막기 위해 안전한 복구(Fallback) 로직을 적용합니다.
                let k_restored = k_src.to_dtype(target_dtype).unwrap_or_else(|_| k_src.clone());
                let v_restored = v_src.to_dtype(target_dtype).unwrap_or_else(|_| v_src.clone());

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

    /// 🌟 [KV RESIDENCY] 디코딩 진입 시 계산된 목표 배치 위치를 주입합니다.
    pub fn set_kv_residency(&mut self, residency: crate::utils::resources::KvResidency) {
        self.kv_residency = residency;
        // 이미 보관 중인 캐시가 있다면 즉시 목표 위치로 이주시킵니다.
        if let Some(cache) = self.fp8_cache.take() {
            let moved = match residency {
                crate::utils::resources::KvResidency::Vram => cache,
                _ => {
                    let k_cpu = cache
                        .k_fp8
                        .to_device(&candle_core::Device::Cpu)
                        .unwrap_or_else(|_| cache.k_fp8.clone());
                    let v_cpu = cache
                        .v_fp8
                        .to_device(&candle_core::Device::Cpu)
                        .unwrap_or_else(|_| cache.v_fp8.clone());
                    Fp8VramKVCache {
                        k_fp8: k_cpu,
                        v_fp8: v_cpu,
                        location: crate::utils::resources::KvResidency::Ram,
                    }
                }
            };
            self.fp8_cache = Some(moved);
        }
    }

    pub fn compress_kv_in_vram(&mut self) -> Result<()> {
        if let Some((k, v)) = self.self_attn.kv_cache.take() {
            let dev = k.device();
            let k_fp8 = if dev.is_cuda() {
                k.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| k.clone())
            } else {
                k.clone()
            };
            let v_fp8 = if dev.is_cuda() {
                v.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| v.clone())
            } else {
                v.clone()
            };

            let stored = match self.kv_residency {
                crate::utils::resources::KvResidency::Vram => Fp8VramKVCache {
                    k_fp8, v_fp8,
                    location: crate::utils::resources::KvResidency::Vram,
                },
                _ => {
                    let k_cpu = k_fp8.to_device(&candle_core::Device::Cpu)
                                     .unwrap_or_else(|_| k_fp8.clone());
                    let v_cpu = v_fp8.to_device(&candle_core::Device::Cpu)
                                     .unwrap_or_else(|_| v_fp8.clone());
                    Fp8VramKVCache {
                        k_fp8: k_cpu, v_fp8: v_cpu,
                        location: crate::utils::resources::KvResidency::Ram,
                    }
                }
            };
            self.fp8_cache = Some(stored);
        }
        Ok(())
    }

    pub fn get_kv_cache(&self) -> Option<(Tensor, Tensor)> {
        if let Some(cache) = &self.self_attn.kv_cache {
            Some(cache.clone())
        } else if let Some(fp8_cache) = &self.fp8_cache {
            // 스냅샷 저장을 위해 호출될 경우, 원래 타입으로 복원하여 반환
            let target_dtype = candle_core::DType::F32; // 저장용 기본 타입
            // 🌟 [KV RESIDENCY] RAM 대피본이라도 스냅샷 저장 경로는 그대로 동작해야 하므로
            //    device 를 따지지 않고 dtype 만 복원합니다. (safetensors 저장은 CPU 텐서를 요구)
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
    // 🌟 [KV RESIDENCY] KV 크기 계산에 필요한 형상 정보를 보관합니다.
    kv_num_layers: usize,
    kv_num_heads: usize,
    kv_head_dim: usize,
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
            kv_num_layers: config.num_hidden_layers,
            kv_num_heads: config.num_key_value_heads,
            kv_head_dim: config.head_dim,
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

    /// 🌟 [KV RESIDENCY] (레이어 수, KV 헤드 수, head_dim) 을 반환합니다.
    pub fn kv_geometry(&self) -> (usize, usize, usize) {
        (self.kv_num_layers, self.kv_num_heads, self.kv_head_dim)
    }

    /// 🌟 [KV RESIDENCY] 디코딩 진입 시 계산된 배치 위치를 전 레이어에 전파합니다.
    /// Qwen3(0.6B) 경로에는 SSD 오프로딩 인프라가 없으므로 Ssd 판정은 RAM 으로 강등 처리합니다.
    pub fn set_kv_residency(&mut self, residency: crate::utils::resources::KvResidency) {
        let effective = match residency {
            crate::utils::resources::KvResidency::Ssd => {
                println!("[KV-PLAN] Qwen3(0.6B) 경로는 디코딩 SSD 오프로딩을 지원하지 않습니다. RAM 오프로딩으로 강등합니다.");
                crate::utils::resources::KvResidency::Ram
            }
            other => other,
        };
        for layer in self.layers.iter_mut() {
            layer.set_kv_residency(effective);
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

use anyhow::{Result, anyhow};
use candle_core::Tensor;
use crate::utils::GpuDeviceExt;
use candle_nn::{
    Linear, Module, RmsNorm, VarBuilder, embedding, linear_b, linear_no_bias, rms_norm,
};

use crate::{
    models::{
        common::{
            GateUpDownMLP, QKNormAttention, eager_attention_forward,
            gguf::{ExactQLinear, ProjKind, qtensor_from_bytes},
        },
        qwen3::config::Qwen3Config,
        qwen3_5::model::EmbedTable,
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
    // 🌟 [KV-PRECISION] true 면 VRAM 에 있는 동안 KV 를 압축(TurboQuant·FP8)하지 않고 원본 dtype 으로 둡니다.
    //    생성 시작 시 VRAM 여유로 정합니다(kv_turbo::exact_kv_affordable). RAM 으로 내릴 때는 기존처럼 압축합니다.
    pub kv_exact: bool,
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
            kv_exact: false,
        })
    }

    /// 🌟 [Q8-RESIDENT] GGUF 의 Q8_0 가중치를 그대로 GPU 에 두는 층. 프로젝션은 정확 W8A16 (활성 양자화 없음).
    pub fn new_from_gguf_q8(
        config: &Qwen3Config,
        ct: &candle_core::quantized::gguf_file::Content,
        data: &[u8],
        idx: usize,
        device: &candle_core::Device,
        act_dtype: candle_core::DType,
    ) -> Result<Self> {
        let p = format!("blk.{idx}");
        let proj = |n: &str| -> Result<ProjKind> { q8_exact_proj(ct, data, &format!("{p}.{n}.weight"), device) };
        let norm = |n: &str| -> Result<RmsNorm> {
            gguf_norm(ct, data, &format!("{p}.{n}.weight"), device, act_dtype, config.rms_norm_eps)
        };
        let self_attn = QKNormAttention::from_parts(
            proj("attn_q")?,
            proj("attn_k")?,
            proj("attn_v")?,
            proj("attn_output")?,
            norm("attn_q_norm")?,
            norm("attn_k_norm")?,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        );
        let mlp = GateUpDownMLP::from_projs(proj("ffn_gate")?, proj("ffn_up")?, proj("ffn_down")?, config.hidden_act);
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm: norm("attn_norm")?,
            post_attention_layernorm: norm("ffn_norm")?,
            fp8_cache: None,
            kv_residency: crate::utils::resources::KvResidency::Vram,
            kv_exact: false,
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
                let target_dtype = if dev.is_cuda_or_rocm() { candle_core::DType::BF16 } else { candle_core::DType::F32 };

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

        // 🌟 [TURBO-KV] TurboQuant 압축 캐시는 VRAM 상주 전용입니다.
        //    RAM/SSD 대피가 결정되면 압축 캐시를 풀어 기존 FP8 + RAM 경로로 넘기고,
        //    다시 VRAM 상주가 결정되면 다음 forward 에서 FP8 캐시를 압축 캐시로 흡수합니다.
        let turbo_ok = residency == crate::utils::resources::KvResidency::Vram && !self.kv_exact;
        self.self_attn.set_turbo_kv_allowed(turbo_ok);
        if !turbo_ok {
            match self.self_attn.take_turbo_kv() {
                Ok(Some((k, v))) => {
                    self.self_attn.kv_cache = Some((k, v));
                    if let Err(e) = self.compress_kv_in_vram() {
                        eprintln!("[TURBO-KV] FP8 전환 실패: {e}");
                    }
                }
                Ok(None) => {}
                Err(e) => {
                    // 압축 해제가 실패하면 문맥을 잃지 않도록 압축 캐시를 VRAM 에 그대로 둡니다.
                    eprintln!("[TURBO-KV] 압축 캐시 해제 실패, VRAM 에 유지합니다: {e}");
                    self.self_attn.set_turbo_kv_allowed(true);
                }
            }
        }

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
        // 🌟 [KV-PRECISION] 원본 KV 모드에서 VRAM 에 둘 때는 FP8 로 바꾸지 않고 그대로 둡니다.
        if self.kv_exact && self.kv_residency == crate::utils::resources::KvResidency::Vram {
            return Ok(());
        }
        if let Some((k, v)) = self.self_attn.kv_cache.take() {
            let dev = k.device();
            let k_fp8 = if dev.is_cuda_or_rocm() {
                k.to_dtype(candle_core::DType::F8E4M3).unwrap_or_else(|_| k.clone())
            } else {
                k.clone()
            };
            let v_fp8 = if dev.is_cuda_or_rocm() {
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
        } else if self.self_attn.has_turbo_kv() {
            // 🌟 [TURBO-KV] 압축 캐시를 원래 dtype 으로 풀어 스냅샷으로 돌려줍니다. (캐시는 유지)
            match self.self_attn.turbo_kv_snapshot() {
                Ok(kv) => kv,
                Err(e) => {
                    eprintln!("[TURBO-KV] 스냅샷 압축 해제 실패: {e}");
                    None
                }
            }
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
        // 🌟 [TURBO-KV] 스냅샷이 전체 이력을 대체하므로 압축 캐시도 비웁니다.
        //    (TurboQuant 경로라면 다음 forward 에서 이 평문 캐시를 압축 캐시로 흡수합니다)
        self.self_attn.reset_turbo_kv();
        self.self_attn.kv_cache = cache;
        self.fp8_cache = None;
    }
}

/// 🌟 [Q8-RESIDENT] GGUF 행렬 하나를 정확 W8A16 선형층으로 (mmap 바이트에서 바로 GPU 로 복사).
fn q8_exact_proj(
    ct: &candle_core::quantized::gguf_file::Content,
    data: &[u8],
    name: &str,
    device: &candle_core::Device,
) -> Result<ProjKind> {
    let q = qtensor_from_bytes(ct, data, name, device)?;
    Ok(ProjKind::Exact(ExactQLinear::new(std::sync::Arc::new(q))))
}

/// 🌟 [Q8-RESIDENT] GGUF 노름 가중치(F32) → `dtype` RmsNorm. (기존 경로의 F16 경유 이중 반올림 없음)
fn gguf_norm(
    ct: &candle_core::quantized::gguf_file::Content,
    data: &[u8],
    name: &str,
    device: &candle_core::Device,
    dtype: candle_core::DType,
    eps: f64,
) -> Result<RmsNorm> {
    let w = qtensor_from_bytes(ct, data, name, device)?.dequantize(device)?.to_dtype(dtype)?;
    Ok(RmsNorm::new(w, eps))
}

pub struct Qwen3Model {
    // 🌟 [Q8-RESIDENT] Dense(기존 BF16 표) 또는 Quant(Q8_0 표 + 행 단위 정확 복원)
    embed_tokens: EmbedTable,
    layers: Vec<Qwen3DecoderLayer>,
    norm: RmsNorm,
    rotary_emb: RoPE,
    lm_head: ProjKind,
    // 🌟 [Q8-RESIDENT] Some(활성 dtype) 이면 Q8_0 상주 경로 (마지막 노름·lm_head 는 F32 → F32 logits)
    q8_act_dtype: Option<candle_core::DType>,
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
            embed_tokens: EmbedTable::Dense(embed_tokens),
            layers,
            norm,
            rotary_emb,
            lm_head: ProjKind::LinearProj(lm_head),
            q8_act_dtype: None,
            kv_num_layers: config.num_hidden_layers,
            kv_num_heads: config.num_key_value_heads,
            kv_head_dim: config.head_dim,
        })
    }

    /// 🌟 [Q8-RESIDENT] Q8_0 GGUF 를 역양자화하지 않고 GPU 에 그대로 올립니다. (`data` = 파일 전체 메모리 맵)
    ///
    /// - q/k/v/o · gate/up/down: `ExactQLinear` (QTensor::forward_exact — 디코드는 f32 누산 정확 mat-vec,
    ///   프리필은 활성 dtype 타일 역양자화 + f32 누산 GEMM. 활성값을 Q8_1 로 양자화하는 MMVQ/MMQ 는 쓰지 않음)
    /// - 임베딩: Q8_0 표에서 행만 꺼내 정확한 F32 로 복원(get_rows). F16/BF16 표 사본 없음.
    /// - lm_head: 묶인 임베딩 표(같은 Arc)를 정확 mat-vec 으로, F32 입력 → F32 logits.
    /// - `act_dtype`: 활성 dtype (CUDA 기본 BF16, 테스트 기준 F32). 노름 가중치는 이 dtype, 마지막 노름은 F32.
    pub fn new_from_gguf_q8(
        config: &Qwen3Config,
        ct: &candle_core::quantized::gguf_file::Content,
        data: &[u8],
        device: &candle_core::Device,
        act_dtype: candle_core::DType,
    ) -> Result<Self> {
        if config.attention_bias {
            return Err(anyhow!("[QWEN3-Q8] attention_bias 가 있는 체크포인트는 Q8 상주 경로가 지원하지 않습니다"));
        }
        // 정확 mat-vec · get_rows 커널이 검증된 형식(Q8_0)만 받습니다. 다른 형식이면 호출부가 기존 BF16 경로로 돌아갑니다.
        for (name, info) in ct.tensor_infos.iter() {
            let is_matrix = name.ends_with(".weight") && info.shape.dims().len() == 2;
            if is_matrix && info.ggml_dtype != candle_core::quantized::GgmlDType::Q8_0 {
                return Err(anyhow!("[QWEN3-Q8] {name} 형식 {:?} (Q8_0 아님)", info.ggml_dtype));
            }
        }
        let embed_q = std::sync::Arc::new(qtensor_from_bytes(ct, data, "token_embd.weight", device)?);
        let hidden = embed_q.shape().dims2()?.1;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for idx in 0..config.num_hidden_layers {
            layers.push(Qwen3DecoderLayer::new_from_gguf_q8(config, ct, data, idx, device, act_dtype)?);
        }
        // 마지막 노름은 F32 (logits 직전 1행이라 비용 없음, BF16 반올림 제거)
        let norm = gguf_norm(ct, data, "output_norm.weight", device, candle_core::DType::F32, config.rms_norm_eps)?;
        let rotary_emb = RoPE::new(config.head_dim, config.rope_theta, device)?;
        let lm_head = if ct.tensor_infos.contains_key("output.weight") {
            q8_exact_proj(ct, data, "output.weight", device)?
        } else {
            ProjKind::Exact(ExactQLinear::new(embed_q.clone()))
        };
        Ok(Self {
            embed_tokens: EmbedTable::Quant { q: embed_q, hidden },
            layers,
            norm,
            rotary_emb,
            lm_head,
            q8_act_dtype: Some(act_dtype),
            kv_num_layers: config.num_hidden_layers,
            kv_num_heads: config.num_key_value_heads,
            kv_head_dim: config.head_dim,
        })
    }

    /// 🌟 [Q8-RESIDENT] Q8_0 가중치 상주 경로인지.
    pub fn is_q8_resident(&self) -> bool {
        self.q8_act_dtype.is_some()
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
        let target_dtype = match self.q8_act_dtype {
            Some(dt) => dt,
            None if inputs_embeds.device().is_cuda_or_rocm() => candle_core::DType::BF16,
            None => candle_core::DType::F32,
        };
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
        // 🌟 [ACCEL-CHECK] 이번 forward 에서 기록된 TurboQuant 실측 오차가 있으면 한 줄로 남깁니다.
        crate::models::common::accel_check::flush_probe_report(hidden_states.device());
        if self.q8_act_dtype.is_some() {
            // 🌟 [Q8-RESIDENT] 마지막 행만 F32 로 올려 F32 노름 → 정확 mat-vec lm_head → F32 logits.
            let last = hidden_states.narrow(1, seq_len - 1, 1)?.to_dtype(candle_core::DType::F32)?;
            let last = self.norm.forward(&last)?;
            return Ok(self.lm_head.forward(&last)?);
        }
        hidden_states = self.norm.forward(&hidden_states)?;
        let hidden_state = hidden_states.narrow(1, seq_len - 1, 1)?;
        let logits = self.lm_head.forward(&hidden_state)?;
        Ok(logits)
    }

    pub fn embedding_token_id(&self, input_ids: &Tensor) -> Result<Tensor> {
        Ok(self.embed_tokens.forward(input_ids)?)
    }

    /// 🌟 [VOCAB CHUNK] 어휘 수.
    pub fn embed_vocab(&self) -> Result<usize> {
        Ok(self.embed_tokens.vocab()?)
    }

    /// 🌟 [VOCAB CHUNK] 임베딩 행 `off..off+take` 를 F32 로. (Dense: narrow + F32 승격 — 기존과 같음 / Quant: 행만 정확 복원)
    pub fn embed_rows_f32(&self, off: usize, take: usize) -> Result<Tensor> {
        Ok(self.embed_tokens.rows_f32(off, take)?)
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

    /// 🌟 [KV-PRECISION] 이번 생성의 KV 를 원본 정밀도로 둘지(true) 압축할지(false) 전 레이어에 설정합니다.
    /// 생성 시작(캐시가 빈 상태)에 호출합니다.
    pub fn set_kv_exact(&mut self, exact: bool) {
        for layer in self.layers.iter_mut() {
            layer.kv_exact = exact;
            let turbo_ok = layer.kv_residency == crate::utils::resources::KvResidency::Vram && !exact;
            layer.self_attn.set_turbo_kv_allowed(turbo_ok);
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

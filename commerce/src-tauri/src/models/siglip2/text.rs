use candle_core::{Result, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};

use super::Siglip2Config;

// =====================================================================
// 텍스트 인코더 (텍스트 앵커 인코딩용)
// =====================================================================
/// 텍스트 앵커(예: "상품명", "가격", "주문번호")를 1152차원 벡터로 변환합니다.
/// 비전 패치 임베딩과 동일한 차원(1152)이므로 코사인 비교가 가능합니다.
pub struct Siglip2TextModel {
    token_embedding: Embedding,
    position_embedding: Embedding,
    layers: Vec<TextEncoderLayer>,
    final_layernorm: candle_nn::LayerNorm,
    head: Linear,
    hidden_size: usize,
    max_seq_len: usize,
    // 🌟 [CPU EMBEDDING] token_embedding 이 호스트 메모리에 있는지.
    //
    //  ── 왜 분리하는가 ──
    //   token_embedding 은 [256000, 1152] = 294.9M 파라미터로
    //   텍스트 인코더 전체(708M)의 41.7%, BF16 기준 590MB 를 차지합니다.
    //   그런데 이 층의 연산은 index_select, 즉 '행 복사' 하나뿐입니다.
    //   배치 32 × 64토큰이면 실제로 읽는 행은 2,048개(중복 포함)이고
    //   결과 텐서는 32 × 64 × 1152 × 2B = 4.7MB 에 불과합니다.
    //   590MB 를 VRAM 에 올려 두고 4.7MB 만 꺼내 쓰는 셈입니다.
    //
    //  ── 정밀도 ──
    //   gather 에는 부동소수 연산이 없습니다. CPU 에서 행을 복사해 GPU 로 올려도
    //   결과가 비트 단위로 동일합니다. 근사도 양자화도 아닙니다.
    embed_on_cpu: bool,
    device: candle_core::Device,
    dtype: candle_core::DType,
}

/// 텍스트 인코더의 단일 층 (비전과 구조 동일)
pub struct TextEncoderLayer {
    self_attn: TextAttention,
    mlp: TextMLP,
    layer_norm1: candle_nn::LayerNorm,
    layer_norm2: candle_nn::LayerNorm,
}

pub struct TextAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

pub struct TextMLP {
    fc1: Linear,
    fc2: Linear,
}

impl Siglip2TextModel {
    /// `embed_vb` 가 Some 이면 token_embedding 만 그 VarBuilder(보통 CPU)에서 로드합니다.
    /// None 이면 종전대로 `vb` 와 같은 디바이스에 올립니다.
    pub fn new(
        config: &Siglip2Config,
        vb: VarBuilder,
        embed_vb: Option<VarBuilder>,
    ) -> Result<Self> {
        let hidden = config.text_hidden_size;
        let device = vb.device().clone();
        let dtype = vb.dtype();

        // 🌟 토큰 임베딩: (256000, 1152) = 590MB(BF16).
        //    embed_vb 가 주어지면 그 디바이스(호스트)로 보내고, VRAM 에서는 뺍니다.
        let (token_embedding, embed_on_cpu) = match embed_vb {
            Some(evb) => {
                let on_cpu = evb.device().is_cpu();
                let e = candle_nn::embedding(
                    config.text_vocab_size,
                    hidden,
                    evb.pp("embeddings").pp("token_embedding"),
                )?;
                (e, on_cpu)
            }
            None => {
                let e = candle_nn::embedding(
                    config.text_vocab_size,
                    hidden,
                    vb.pp("embeddings").pp("token_embedding"),
                )?;
                (e, false)
            }
        };

        // 27층 인코더
        let mut layers = Vec::with_capacity(config.text_num_layers);
        let encoder_vb = vb.pp("encoder");
        for i in 0..config.text_num_layers {
            let layer_vb = encoder_vb.pp(format!("layers.{}", i));
            let layer = TextEncoderLayer::new(config, layer_vb)?;
            layers.push(layer);
        }

        // 🌟 [TENSOR CONTRACT] text_model.embeddings.position_embedding.weight 는
        //    [64, 1152] 입니다. 512 를 요구하면 shape mismatch 로 즉시 실패합니다.
        let position_embedding = candle_nn::embedding(
            config.text_max_positions,
            hidden,
            vb.pp("embeddings").pp("position_embedding"),
        )?;

        let final_layernorm = candle_nn::layer_norm(
            hidden,
            config.text_layer_norm_eps,
            vb.pp("final_layer_norm"),
        )?;

        // text_model.head.weight [1152,1152] / head.bias [1152] — 실제 Linear 입니다.
        let head = candle_nn::linear(hidden, hidden, vb.pp("head"))?;

        if embed_on_cpu {
            println!(
                "[SigLIP2] token_embedding({}x{}) 을 호스트 메모리에 배치했습니다. VRAM 약 {:.0}MB 절감.",
                config.text_vocab_size,
                hidden,
                (config.text_vocab_size * hidden * 2) as f64 / 1e6
            );
        }

        Ok(Self {
            token_embedding,
            position_embedding,
            layers,
            final_layernorm,
            head,
            hidden_size: hidden,
            max_seq_len: config.text_max_positions,
            embed_on_cpu,
            device,
            dtype,
        })
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// 텍스트 토큰 시퀀스 → 텍스트 임베딩
    ///
    /// 입력:
    ///   token_ids (b, seq_len)  — seq_len 은 반드시 max_seq_len(64) 고정
    ///   attn_mask (b, seq_len)  — 1 = 실토큰, 0 = 패딩 (없으면 전부 유효로 간주)
    /// 출력:
    ///   (b, 1152)
    ///
    /// 🌟 [POOLING] SigLIP 계열은 평균 풀링이 아니라 **마지막 토큰** 풀링입니다.
    ///    HF SiglipTextTransformer:  pooled = last_hidden_state[:, -1, :]  → head(pooled)
    ///    평균 풀링을 쓰면 학습 공간과 좌표계가 달라져 비전 패치와의 코사인이
    ///    구조적으로 의미를 잃습니다.
    ///
    /// 🌟 [MASK] 패딩 위치를 key 로 참여시키지 않기 위해 가산 마스크를 만듭니다.
    ///    (SigLIP 텍스트 인코더에는 causal mask 가 없습니다. 양방향입니다.)
    pub fn forward(&self, token_ids: &Tensor, attn_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq_len) = token_ids.dims2()?;

        // 1. 토큰 임베딩
        //
        // 🌟 [CPU GATHER] 임베딩 테이블이 호스트에 있으면 gather 도 호스트에서 합니다.
        //    옮기는 것은 (b, seq, 1152) 결과 하나뿐입니다.
        //    배치 32 기준 32 × 64 × 1152 × 2B = 4.7MB — PCIe 로 1ms 미만입니다.
        //    반대로 테이블 자체를 VRAM 에 두면 590MB 가 인코딩 내내 묶입니다.
        let mut x = if self.embed_on_cpu {
            let ids_cpu = token_ids.to_device(&candle_core::Device::Cpu)?;
            let gathered = self.token_embedding.forward(&ids_cpu)?;
            gathered.to_device(&self.device)?.to_dtype(self.dtype)?
        } else {
            self.token_embedding.forward(token_ids)?
        };

        // 2. 위치 임베딩
        //    🌟 위치 임베딩은 [64, 1152] = 0.15MB 라 VRAM 에 두는 편이 유리합니다.
        //       인덱스 텐서는 x 의 디바이스에 맞춥니다(token_ids 가 CPU 일 수 있으므로).
        let pos_ids: Vec<u32> = (0..seq_len as u32).collect();
        let pos_ids_tensor = Tensor::new(&pos_ids[..], x.device())?;
        let pos_emb = self.position_embedding.forward(&pos_ids_tensor)?; // (seq, D)
        let pos_emb = pos_emb.unsqueeze(0)?;
        x = x.broadcast_add(&pos_emb)?;

        // 3. 패딩 마스크를 (b, 1, 1, seq) 가산 마스크로 변환
        let additive = match attn_mask {
            Some(m) => {
                let m = m.to_dtype(x.dtype())?; // (b, seq) : 1 / 0
                let neg = ((m.ones_like()? - &m)? * -1e9f64)?;
                Some(neg.reshape((b, 1, 1, seq_len))?)
            }
            None => None,
        };

        // 4. 27층 순전파
        for layer in &self.layers {
            x = layer.forward(&x, additive.as_ref())?;
        }

        // 5. 최종 LayerNorm
        let x = self.final_layernorm.forward(&x)?;

        // 6. 마지막 토큰 풀링
        let pooled = x.narrow(1, seq_len - 1, 1)?.squeeze(1)?.contiguous()?; // (b, D)

        // 7. 프로젝션 헤드
        self.head.forward(&pooled)
    }

    /// 여러 텍스트를 한 번에 인코딩합니다.
    ///
    /// 입력: (token_ids, attn_mask) 쌍의 목록. 모든 시퀀스 길이는 동일해야 합니다.
    /// 출력: (n, 1152) L2 정규화 이전 원본 텐서
    pub fn encode_batch(
        &self,
        batch: &[(Vec<u32>, Vec<u32>)],
        device: &candle_core::Device,
    ) -> Result<Tensor> {
        if batch.is_empty() {
            return Tensor::zeros((0, self.hidden_size), candle_core::DType::F32, device);
        }
        let seq = batch[0].0.len();
        let b = batch.len();

        let mut flat_ids: Vec<u32> = Vec::with_capacity(b * seq);
        let mut flat_mask: Vec<u32> = Vec::with_capacity(b * seq);
        for (ids, mask) in batch {
            flat_ids.extend_from_slice(ids);
            flat_mask.extend_from_slice(mask);
        }

        let ids = Tensor::from_vec(flat_ids, (b, seq), device)?;
        let mask = Tensor::from_vec(flat_mask, (b, seq), device)?;

        self.forward(&ids, Some(&mask))
    }

    /// 레거시 호환: 토큰 시퀀스 목록 → 벡터 목록
    pub fn encode_texts(
        &self,
        token_sequences: &[Vec<u32>],
        device: &candle_core::Device,
    ) -> Result<Vec<Tensor>> {
        let mut embeddings = Vec::with_capacity(token_sequences.len());
        for tokens in token_sequences {
            let ids = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
            let emb = self.forward(&ids, None)?;
            embeddings.push(emb.squeeze(0)?);
        }
        Ok(embeddings)
    }
}

impl TextEncoderLayer {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let hidden = config.text_hidden_size;
        let eps = config.text_layer_norm_eps;

        let self_attn = TextAttention::new(config, vb.pp("self_attn"))?;
        let mlp = TextMLP::new(config, vb.pp("mlp"))?;
        let layer_norm1 = candle_nn::layer_norm(hidden, eps, vb.pp("layer_norm1"))?;
        let layer_norm2 = candle_nn::layer_norm(hidden, eps, vb.pp("layer_norm2"))?;

        Ok(Self {
            self_attn,
            mlp,
            layer_norm1,
            layer_norm2,
        })
    }

    pub fn forward(&self, x: &Tensor, additive_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = x;
        let h = self.layer_norm1.forward(x)?;
        let h = self.self_attn.forward(&h, additive_mask)?;
        let x = (h + residual)?;

        let residual = &x;
        let h = self.layer_norm2.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        let x = (h + residual)?;

        Ok(x)
    }
}

impl TextAttention {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let hidden = config.text_hidden_size;
        let num_heads = config.text_num_heads;
        let head_dim = hidden / num_heads;

        Ok(Self {
            q_proj: candle_nn::linear(hidden, hidden, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear(hidden, hidden, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear(hidden, hidden, vb.pp("v_proj"))?,
            out_proj: candle_nn::linear(hidden, hidden, vb.pp("out_proj"))?,
            num_heads,
            head_dim,
        })
    }

    pub fn forward(&self, x: &Tensor, additive_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let q = self
            .q_proj
            .forward(x)?
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(x)?
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let mut attn = (attn / scale)?;

        // 🌟 [PAD MASK] 패딩 위치를 key 에서 제거합니다.
        //    SigLIP 텍스트 인코더에는 causal mask 가 없으므로 이 가산 마스크가 전부입니다.
        if let Some(m) = additive_mask {
            attn = attn.broadcast_add(m)?;
        }

        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq_len, self.num_heads * self.head_dim))?;
        self.out_proj.forward(&out)
    }
}

impl TextMLP {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: candle_nn::linear(
                config.text_hidden_size,
                config.text_intermediate_size,
                vb.pp("fc1"),
            )?,
            fc2: candle_nn::linear(
                config.text_intermediate_size,
                config.text_hidden_size,
                vb.pp("fc2"),
            )?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 🌟 hidden_act = "gelu_pytorch_tanh" → candle 의 gelu() (tanh 근사)
        let x = self.fc1.forward(x)?;
        let x = x.gelu()?;
        self.fc2.forward(&x)
    }
}
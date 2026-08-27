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
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let hidden = config.text_hidden_size;

        // 토큰 임베딩: (256000, 1152)
        let token_embedding = candle_nn::embedding(
            config.text_vocab_size,
            hidden,
            vb.pp("embeddings").pp("token_embedding"),
        )?;

        // 27층 인코더
        let mut layers = Vec::with_capacity(config.text_num_layers);
        let encoder_vb = vb.pp("encoder");
        for i in 0..config.text_num_layers {
            let layer_vb = encoder_vb.pp(format!("layers.{}", i));
            let layer = TextEncoderLayer::new(config, layer_vb)?;
            layers.push(layer);
        }

        // 🌟 [추가] 텍스트 위치 임베딩: [512, 1152]
        let position_embedding = candle_nn::embedding(
            512,
            hidden,
            vb.pp("embeddings").pp("position_embedding"),
        )?;

        let final_layernorm =
            candle_nn::layer_norm(hidden, 1e-6, vb.pp("final_layer_norm"))?;

        // 🌟 [추가] 텍스트 → 비전 공간 프로젝션 헤드: [1152, 1152]
        let head = candle_nn::linear(hidden, hidden, vb.pp("head"))?;

        Ok(Self {
            token_embedding,
            position_embedding,
            layers,
            final_layernorm,
            head,
            hidden_size: hidden,
            max_seq_len: 512,
        })
    }

    /// 텍스트 토큰 시퀀스 → 텍스트 임베딩
    ///
    /// 입력:  token_ids (1, seq_len) — SentencePiece 토큰 ID
    /// 출력:  (1, 1152) — 평균 풀링된 텍스트 임베딩
    ///
    /// 파이프라인에서 텍스트 앵커("상품명", "배송상태" 등)를
    /// 이 함수로 인코딩하여 비전 패치 임베딩과 코사인 비교합니다.
    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        // 1. 토큰 임베딩: (1, seq_len) → (1, seq_len, 1152)
        let mut x = self.token_embedding.forward(token_ids)?;

        // 2. 위치 임베딩 덧셈: [512, 1152]에서 시퀀스 길이만큼 슬라이스
        let seq_len = token_ids.dims()[1];
        let pos_ids: Vec<u32> = (0..seq_len as u32).collect();
        let pos_ids_tensor = Tensor::new(&pos_ids[..], token_ids.device())?;
        let pos_emb = self.position_embedding.forward(&pos_ids_tensor)?; // (seq_len, 1152)
        let pos_emb = pos_emb.unsqueeze(0)?; // (1, seq_len, 1152)
        x = (x + pos_emb)?;

        // 3. 27층 순전파
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        // 4. 최종 LayerNorm
        let x = self.final_layernorm.forward(&x)?;

        // 5. 평균 풀링: (1, seq_len, 1152) → (1, 1152)
        let x = x.mean(1)?;

        // 6. 프로젝션 헤드: 텍스트 임베딩을 비전 공간으로 변환
        //    (1, 1152) → (1, 1152)
        let x = self.head.forward(&x)?;

        Ok(x)
    }

    /// 여러 텍스트를 배치로 인코딩
    /// 입력: 각 텍스트의 토큰 ID 벡터
    /// 출력: 각 텍스트의 1152차원 임베딩
    pub fn encode_texts(
        &self,
        token_sequences: &[Vec<u32>],
        device: &candle_core::Device,
    ) -> Result<Vec<Tensor>> {
        let mut embeddings = Vec::with_capacity(token_sequences.len());
        for tokens in token_sequences {
            let ids = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?; // (1, seq)
            let emb = self.forward(&ids)?; // (1, 1152)
            embeddings.push(emb.squeeze(0)?); // (1152,)
        }
        Ok(embeddings)
    }
}

impl TextEncoderLayer {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let hidden = config.text_hidden_size;
        let eps = 1e-6;

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

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x;
        let x = self.layer_norm1.forward(x)?;
        let x = self.self_attn.forward(&x)?;
        let x = (x + residual)?;

        let residual = &x;
        let x = self.layer_norm2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (x + residual)?;

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

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self.k_proj.forward(x)?
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self.v_proj.forward(x)?
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        let scale = (self.head_dim as f64).sqrt();
        let attn = q.matmul(&k.transpose(2, 3)?)?;
        let attn = (attn / scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        let out = out.transpose(1, 2)?
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
        let x = self.fc1.forward(x)?;
        let x = x.gelu_erf()?;
        self.fc2.forward(&x)
    }
}
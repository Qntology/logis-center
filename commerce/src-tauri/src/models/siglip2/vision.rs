use candle_core::{IndexOp, Result, Tensor, D};
use candle_nn::{Conv2d, Linear, Module, VarBuilder};

use super::Siglip2Config;

// =====================================================================
// Patch Embedding: 이미지를 16×16 패치로 분할 → 선형 투영 → 1152차원
// =====================================================================
pub struct Siglip2PatchEmbedding {
    projection: Conv2d,
    patch_size: usize,
    device: candle_core::Device,
}

impl Siglip2PatchEmbedding {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        // SigLIP2는 Conv2d(kernel_size=patch_size, stride=patch_size) 사용
        // in_channels=3, out_channels=hidden_size
        let projection = candle_nn::conv2d(
            3,
            config.vision_hidden_size,
            config.patch_size,
            candle_nn::Conv2dConfig {
                stride: config.patch_size,
                ..Default::default()
            },
            vb.pp("patch_embedding"),
        )?;
    }

    /// 입력: (1, 3, H, W) → 출력: (1, num_patches, hidden_size)
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // Conv2d 적용: (1, 3, H, W) → (1, hidden_size, H/16, W/16)
        let x = self.projection.forward(pixel_values)?;
        // (1, hidden_size, grid_h, grid_w) → (1, hidden_size, grid_h * grid_w)
        let (b, c, gh, gw) = x.dims4()?;
        let x = x.reshape((b, c, gh * gw))?;
        // (1, hidden_size, num_patches) → (1, num_patches, hidden_size)
        let x = x.transpose(1, 2)?;
        Ok(x)
    }
}

// =====================================================================
// Multi-Head Self-Attention
// =====================================================================
pub struct Siglip2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl Siglip2Attention {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let hidden = config.vision_hidden_size;
        let num_heads = config.vision_num_heads;
        let head_dim = hidden / num_heads;

        let q_proj = candle_nn::linear(hidden, hidden, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(hidden, hidden, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(hidden, hidden, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(hidden, hidden, vb.pp("out_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
        })
    }

    /// 입력: (1, seq_len, hidden_size) → 출력: (1, seq_len, hidden_size)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, seq_len, _hidden) = x.dims3()?;

        // Q, K, V 프로젝션: (1, seq, hidden)
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // 멀티헤드 분할: (1, seq, hidden) → (1, num_heads, seq, head_dim)
        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // (1, heads, seq, head_dim)
        let k = k
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Scaled Dot-Product Attention
        // attn_weights: (1, heads, seq, seq)
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = (attn_weights / scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // attn_output: (1, heads, seq, head_dim)
        let attn_output = attn_weights.matmul(&v)?;

        // 헤드 재결합: (1, heads, seq, head_dim) → (1, seq, hidden)
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b, seq_len, self.num_heads * self.head_dim))?;

        // 출력 프로젝션
        self.out_proj.forward(&attn_output)
    }
}

// =====================================================================
// MLP (Feed-Forward Network)
// =====================================================================
pub struct Siglip2MLP {
    fc1: Linear,
    fc2: Linear,
}

impl Siglip2MLP {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let hidden = config.vision_hidden_size;
        let intermediate = config.vision_intermediate_size;

        let fc1 = candle_nn::linear(hidden, intermediate, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(intermediate, hidden, vb.pp("fc2"))?;

        Ok(Self { fc1, fc2 })
    }

    /// 입력: (1, seq, hidden) → 출력: (1, seq, hidden)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SigLIP2는 GELU 활성화 사용
        let x = self.fc1.forward(x)?;
        let x = x.gelu_erf()?; // GELU (erf 기반)
        self.fc2.forward(&x)
    }
}

// =====================================================================
// 단일 Transformer 인코더 층
// =====================================================================
pub struct Siglip2EncoderLayer {
    self_attn: Siglip2Attention,
    mlp: Siglip2MLP,
    layer_norm1: candle_nn::LayerNorm,
    layer_norm2: candle_nn::LayerNorm,
}

impl Siglip2EncoderLayer {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let hidden = config.vision_hidden_size;
        let eps = 1e-6;

        let self_attn = Siglip2Attention::new(config, vb.pp("self_attn"))?;
        let mlp = Siglip2MLP::new(config, vb.pp("mlp"))?;
        let layer_norm1 = candle_nn::layer_norm(hidden, eps, vb.pp("layer_norm1"))?;
        let layer_norm2 = candle_nn::layer_norm(hidden, eps, vb.pp("layer_norm2"))?;

        Ok(Self {
            self_attn,
            mlp,
            layer_norm1,
            layer_norm2,
        })
    }

    /// Pre-Norm 구조: LayerNorm → Attention → 잔차 → LayerNorm → MLP → 잔차
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Self-Attention + 잔차 연결
        let residual = x;
        let x = self.layer_norm1.forward(x)?;
        let x = self.self_attn.forward(&x)?;
        let x = (x + residual)?;

        // MLP + 잔차 연결
        let residual = &x;
        let x = self.layer_norm2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (x + residual)?;

        Ok(x)
    }
}

// =====================================================================
// 전체 비전 인코더 (27층 스택)
// =====================================================================
pub struct Siglip2VisionModel {
    patch_embedding: Siglip2PatchEmbedding,
    position_embedding: candle_nn::Embedding,
    max_num_patches: usize,
    layers: Vec<Siglip2EncoderLayer>,
    post_layernorm: candle_nn::LayerNorm,
    head: candle_nn::Linear,
    hidden_size: usize,
}

impl Siglip2VisionModel {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let patch_embedding =
            Siglip2PatchEmbedding::new(config, vb.pp("embeddings").pp("patch_embedding"))?;

        // 🌟 [추가] 위치 임베딩: [256, 1152] — 패치 수만큼 슬라이스하여 덧셈
        let position_embedding = candle_nn::embedding(
            config.max_num_patches,
            config.vision_hidden_size,
            vb.pp("embeddings").pp("position_embedding"),
        )?;

        let mut layers = Vec::with_capacity(config.vision_num_layers);
        let encoder_vb = vb.pp("encoder");
        for i in 0..config.vision_num_layers {
            let layer_vb = encoder_vb.pp(format!("layers.{}", i));
            let layer = Siglip2EncoderLayer::new(config, layer_vb)?;
            layers.push(layer);
        }

        let post_layernorm = candle_nn::layer_norm(
            config.vision_hidden_size,
            1e-6,
            vb.pp("post_layernorm"),
        )?;

        // 🌟 [추가] 비전 → 텍스트 공간 프로젝션 헤드: [1152, 1152]
        let head = candle_nn::linear(
            config.vision_hidden_size,
            config.vision_hidden_size,
            vb.pp("head"),
        )?;

        Ok(Self {
            patch_embedding,
            position_embedding,
            max_num_patches: config.max_num_patches,
            layers,
            post_layernorm,
            head,
            hidden_size: config.vision_hidden_size,
        })
    }

    /// 핵심 순전파: 이미지 텐서 → 패치 임베딩 행렬
    ///
    /// 입력:  pixel_values (1, 3, H, W) — 정규화된 이미지
    /// 출력:  (1, num_patches, 1152) — 각 패치의 임베딩 벡터
    ///
    /// 이 출력이 파이프라인의 STEP 1 결과입니다.
    /// 각 행이 하나의 16×16 패치에 대응하며,
    /// 텍스트 앵커 임베딩과 코사인 유사도를 계산합니다.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // 1. 패치 임베딩: (1, 3, H, W) → (1, num_patches, 1152)
        let mut x = self.patch_embedding.forward(pixel_values)?;

        // 2. 위치 임베딩 덧셈
        //    position_embedding.weight: [256, 1152]
        //    실제 패치 수만큼 슬라이스하여 더합니다.
        let num_patches = x.dims()[1];
        let pos_ids: Vec<u32> = (0..num_patches as u32).collect();
        let pos_ids_tensor = Tensor::new(&pos_ids[..], &self.patch_embedding.device)?;
        let pos_emb = self.position_embedding.forward(&pos_ids_tensor)?; // (num_patches, 1152)
        let pos_emb = pos_emb.unsqueeze(0)?; // (1, num_patches, 1152)
        x = (x + pos_emb)?;

        // 3. 27층 Transformer 순전파
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        // 4. 최종 LayerNorm
        let x = self.post_layernorm.forward(&x)?;

        // 5. 프로젝션 헤드: (1, num_patches, 1152) → (1, num_patches, 1152)
        //    비전 임베딩을 텍스트 임베딩과 동일한 공간으로 변환합니다.
        let x = self.head.forward(&x)?;

        Ok(x)
    }

    /// 편의 메서드: 배치 차원을 제거하고 패치 임베딩만 반환
    /// 출력: (num_patches, 1152)
    pub fn get_patch_embeddings(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let out = self.forward(pixel_values)?; // (1, num_patches, 1152)
        out.squeeze(0) // (num_patches, 1152)
    }
}
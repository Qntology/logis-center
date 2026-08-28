use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder};

use super::Siglip2Config;

// =====================================================================
// Patch Embedding (NaFlex)
// ---------------------------------------------------------------------
// 🌟 [TENSOR CONTRACT] vision_model.embeddings.patch_embedding.weight 는
//    [1152, 768] 즉 Linear 입니다. 768 = 16 * 16 * 3.
//    Conv2d 로 로드하면 candle 이 [1152, 3, 16, 16] 을 요구해 즉시 실패합니다.
//
//    flatten 순서는 HF image_processing_siglip2.convert_image_to_patches 와
//    반드시 동일해야 합니다.
//      (nh, p, nw, p, C) -> transpose(0,2,1,3,4) -> (nh, nw, p, p, C)
//    즉 패치 내부 인덱스는 (patch_row, patch_col, channel) 순서이며
//    채널이 가장 빠르게 변합니다.
// =====================================================================
pub struct Siglip2PatchEmbedding {
    projection: Linear,
    patch_size: usize,
}

impl Siglip2PatchEmbedding {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let in_dim = config.patch_size * config.patch_size * 3;
        let projection = candle_nn::linear(in_dim, config.vision_hidden_size, vb)?;
        Ok(Self {
            projection,
            patch_size: config.patch_size,
        })
    }

    /// 입력: (1, 3, H, W)  (H = rows*p, W = cols*p)
    /// 출력: (1, rows*cols, hidden_size)
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = pixel_values.dims4()?;
        let p = self.patch_size;
        let nh = h / p;
        let nw = w / p;

        // (b, C, nh, p, nw, p)
        let x = pixel_values.reshape((b, c, nh, p, nw, p))?;
        // (b, nh, nw, p, p, C)
        let x = x.permute(&[0usize, 2, 4, 3, 5, 1][..])?.contiguous()?;
        // (b, nh*nw, p*p*C)
        let x = x.reshape((b, nh * nw, p * p * c))?;

        self.projection.forward(&x)
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

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // 멀티헤드 분할: (1, seq, hidden) → (1, num_heads, seq, head_dim)
        let q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;                    // ← 추가
        let k = k
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;                    // ← 추가
        let v = v
            .reshape((b, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;                    // ← 추가

        // Scaled Dot-Product Attention
        let scale = (self.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;   // ← 추가
        let attn_weights = q.matmul(&k_t)?;
        let attn_weights = (attn_weights / scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // attn_output: (1, heads, seq, head_dim)
        let attn_output = attn_weights.matmul(&v)?;

        // 헤드 재결합: (1, heads, seq, head_dim) → (1, seq, hidden)
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?                     // ← 추가
            .reshape((b, seq_len, self.num_heads * self.head_dim))?;

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
        // 🌟 [ACT] config 의 hidden_act 는 "gelu_pytorch_tanh" 입니다.
        //    candle 의 gelu() 가 tanh 근사, gelu_erf() 가 정확한 erf 구현이므로
        //    학습 시점과 동일한 tanh 근사를 사용해야 층마다 오차가 누적되지 않습니다.
        let x = self.fc1.forward(x)?;
        let x = x.gelu()?;
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
// Multihead Attention Pooling Head
// ---------------------------------------------------------------------
// 🌟 [TENSOR CONTRACT] vision_model.head 는 Linear 가 아닙니다.
//    head.probe                     [1, 1, 1152]
//    head.attention.in_proj_weight  [3456, 1152]   = q|k|v 세로 concat
//    head.attention.in_proj_bias    [3456]
//    head.attention.out_proj.weight [1152, 1152]
//    head.attention.out_proj.bias   [1152]
//    head.layernorm.weight/bias     [1152]
//    head.mlp.fc1 / fc2
//
//  HF 순전파:
//    h = MHA(query=probe, key=hidden, value=hidden)
//    h = h + mlp(layernorm(h))
//    return h[:, 0]
// =====================================================================
pub struct Siglip2AttentionPoolingHead {
    probe: Tensor,          // (1, 1, D)
    in_proj_weight: Tensor, // (3D, D)
    in_proj_bias: Tensor,   // (3D)
    out_proj: Linear,
    layernorm: candle_nn::LayerNorm,
    mlp: Siglip2MLP,
    num_heads: usize,
    head_dim: usize,
    hidden: usize,
}

impl Siglip2AttentionPoolingHead {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let hidden = config.vision_hidden_size;
        let num_heads = config.vision_num_heads;
        let head_dim = hidden / num_heads;

        let probe = vb.get((1, 1, hidden), "probe")?;

        let attn_vb = vb.pp("attention");
        let in_proj_weight = attn_vb.get((3 * hidden, hidden), "in_proj_weight")?;
        let in_proj_bias = attn_vb.get(3 * hidden, "in_proj_bias")?;
        let out_proj = candle_nn::linear(hidden, hidden, attn_vb.pp("out_proj"))?;

        let layernorm =
            candle_nn::layer_norm(hidden, config.vision_layer_norm_eps, vb.pp("layernorm"))?;
        let mlp = Siglip2MLP::new(config, vb.pp("mlp"))?;

        Ok(Self {
            probe,
            in_proj_weight,
            in_proj_bias,
            out_proj,
            layernorm,
            mlp,
            num_heads,
            head_dim,
            hidden,
        })
    }

    /// in_proj_weight 를 q(0) / k(1) / v(2) 로 잘라 선형 투영합니다.
    fn proj(&self, x: &Tensor, slot: usize) -> Result<Tensor> {
        let w = self
            .in_proj_weight
            .narrow(0, slot * self.hidden, self.hidden)?
            .contiguous()?;
        let b = self
            .in_proj_bias
            .narrow(0, slot * self.hidden, self.hidden)?
            .contiguous()?;
        Linear::new(w, Some(b)).forward(x)
    }

    /// 헤드의 잔차 블록. (attention 이후 / 패치 투영 이후 공통)
    fn residual_block(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();
        let h = self.layernorm.forward(x)?;
        residual + self.mlp.forward(&h)?
    }

    /// 이미지 전체를 하나의 벡터로 풀링합니다. 출력: (b, D)
    pub fn pool(&self, hidden: &Tensor) -> Result<Tensor> {
        let (b, n, _) = hidden.dims3()?;
        let probe = self.probe.expand((b, 1, self.hidden))?.contiguous()?;
        let q = self.proj(&probe, 0)?;
        let k = self.proj(hidden, 1)?;
        let v = self.proj(hidden, 2)?;

        let q = q
            .reshape((b, 1, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;                    // ← 추가
        let k = k
            .reshape((b, n, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;                    // ← 추가
        let v = v
            .reshape((b, n, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;                    // ← 추가

        let scale = (self.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;   // ← 추가
        let attn = q.matmul(&k_t)?;
        let attn = (attn / scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        let out = out
            .transpose(1, 2)?
            .contiguous()?                     // ← 추가
            .reshape((b, 1, self.hidden))?;
        let out = self.out_proj.forward(&out)?;
        let out = self.residual_block(&out)?;
        out.narrow(1, 0, 1)?.squeeze(1)
    }

    /// 🌟 [DENSE FEATURE] 패치별 공유공간 투영.
    ///
    ///  어텐션 출력은 out_proj( Σ αᵢ · v(xᵢ) ) 이므로,
    ///  패치 i 가 풀링 벡터에 기여하는 성분은 정확히 out_proj( v(xᵢ) ) 입니다.
    ///  여기에 헤드의 잔차 블록을 동일하게 적용하면
    ///  텍스트 헤드 출력과 같은 좌표계의 dense patch feature 가 됩니다.
    ///  이 값이 STEP 2 / STEP 3 코사인 히트맵의 원본입니다.
    ///
    ///  입력: (b, N, D) → 출력: (b, N, D)
    pub fn project_patches(&self, hidden: &Tensor) -> Result<Tensor> {
        let v = self.proj(hidden, 2)?;
        let o = self.out_proj.forward(&v)?;
        self.residual_block(&o)
    }
}

// =====================================================================
// 비전 인코더 순전파 산출물
// =====================================================================
pub struct VisionForward {
    /// post_layernorm 직후 패치 표현 (1, N, D)
    pub patch_hidden: Tensor,
    /// 텍스트 공유공간으로 투영된 패치 표현 (1, N, D)
    pub patch_shared: Tensor,
    /// 이미지 전체 풀링 벡터 (1, D)
    pub pooled: Tensor,
}

// =====================================================================
// 전체 비전 인코더 (27층 스택 + 어텐션 풀링 헤드)
// =====================================================================
pub struct Siglip2VisionModel {
    patch_embedding: Siglip2PatchEmbedding,
    layers: Vec<Siglip2EncoderLayer>,
    post_layernorm: candle_nn::LayerNorm,
    head: Siglip2AttentionPoolingHead,
    hidden_size: usize,
    /// 🌟 위치 임베딩 원본 격자를 f32 호스트 메모리에 보관합니다.
    ///    NaFlex 는 (rows, cols) 가 이미지마다 달라지므로
    ///    매 순전파마다 16×16 격자를 bilinear 로 리샘플링해야 합니다.
    pos_grid: Vec<f32>,
    pos_side: usize,
    device: Device,
    dtype: DType,
}

impl Siglip2VisionModel {
    pub fn new(config: &Siglip2Config, vb: VarBuilder) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();
        let hidden = config.vision_hidden_size;
        let side = config.pos_grid_side();

        let patch_embedding =
            Siglip2PatchEmbedding::new(config, vb.pp("embeddings").pp("patch_embedding"))?;

        // 위치 임베딩 원본 [256, 1152] 을 f32 로 내려 받아 격자로 보관합니다.
        let pos_w = vb
            .pp("embeddings")
            .pp("position_embedding")
            .get((config.max_num_patches, hidden), "weight")?;
        let pos_grid = pos_w
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let mut layers = Vec::with_capacity(config.vision_num_layers);
        let encoder_vb = vb.pp("encoder");
        for i in 0..config.vision_num_layers {
            let layer_vb = encoder_vb.pp(format!("layers.{}", i));
            layers.push(Siglip2EncoderLayer::new(config, layer_vb)?);
        }

        let post_layernorm = candle_nn::layer_norm(
            hidden,
            config.vision_layer_norm_eps,
            vb.pp("post_layernorm"),
        )?;

        let head = Siglip2AttentionPoolingHead::new(config, vb.pp("head"))?;

        Ok(Self {
            patch_embedding,
            layers,
            post_layernorm,
            head,
            hidden_size: hidden,
            pos_grid,
            pos_side: side,
            device,
            dtype,
        })
    }

    /// 🌟 [NaFlex POSITION INTERPOLATION]
    ///  16×16 격자를 (rows, cols) 로 bilinear 리샘플링합니다.
    ///  torch F.interpolate(mode="bilinear", align_corners=False) 규약과 동일한
    ///  좌표 매핑 ( (i + 0.5) * src / dst - 0.5 ) 을 사용합니다.
    ///
    ///  기존 코드처럼 0..N 을 단순 슬라이스하면
    ///  18×14 격자의 (0,0)~(17,13) 이 원본의 1차원 순번 0..251 에 매핑되어
    ///  행/열 좌표가 완전히 어긋납니다. 좌표 크롭 파이프라인에서는 치명적입니다.
    fn interpolate_pos(&self, rows: usize, cols: usize) -> Result<Tensor> {
        let s = self.pos_side;
        let d = self.hidden_size;
        let mut out = vec![0f32; rows * cols * d];

        for r in 0..rows {
            let fy = (((r as f32) + 0.5) * (s as f32) / (rows as f32) - 0.5)
                .clamp(0.0, (s - 1) as f32);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(s - 1);
            let wy = fy - y0 as f32;

            for c in 0..cols {
                let fx = (((c as f32) + 0.5) * (s as f32) / (cols as f32) - 0.5)
                    .clamp(0.0, (s - 1) as f32);
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(s - 1);
                let wx = fx - x0 as f32;

                let base = (r * cols + c) * d;
                let p00 = (y0 * s + x0) * d;
                let p01 = (y0 * s + x1) * d;
                let p10 = (y1 * s + x0) * d;
                let p11 = (y1 * s + x1) * d;

                for k in 0..d {
                    let top = self.pos_grid[p00 + k] * (1.0 - wx) + self.pos_grid[p01 + k] * wx;
                    let bot = self.pos_grid[p10 + k] * (1.0 - wx) + self.pos_grid[p11 + k] * wx;
                    out[base + k] = top * (1.0 - wy) + bot * wy;
                }
            }
        }

        Tensor::from_vec(out, (1, rows * cols, d), &self.device)?.to_dtype(self.dtype)
    }

    /// 핵심 순전파.
    ///
    /// 입력:  pixel_values (1, 3, H, W) — 정규화된 이미지
    ///        grid_rows / grid_cols     — 전처리기가 확정한 패치 격자
    /// 출력:  VisionForward { patch_hidden, patch_shared, pooled }
    pub fn forward(
        &self,
        pixel_values: &Tensor,
        grid_rows: usize,
        grid_cols: usize,
    ) -> Result<VisionForward> {
        // 1. 패치 임베딩 (Linear)
        let mut x = self.patch_embedding.forward(pixel_values)?;

        // 2. 위치 임베딩 bilinear 보간 후 덧셈
        let pos = self.interpolate_pos(grid_rows, grid_cols)?;
        x = (x + pos)?;

        // 3. 27층 Transformer
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        // 4. post LayerNorm
        let patch_hidden = self.post_layernorm.forward(&x)?;

        // 5. 헤드: 풀링 벡터 + 패치별 공유공간 투영
        let pooled = self.head.pool(&patch_hidden)?;
        let patch_shared = self.head.project_patches(&patch_hidden)?;

        Ok(VisionForward {
            patch_hidden,
            patch_shared,
            pooled,
        })
    }

    /// 편의 메서드: 공유공간 패치 임베딩만 (num_patches, D) 로 반환합니다.
    pub fn get_patch_embeddings(
        &self,
        pixel_values: &Tensor,
        grid_rows: usize,
        grid_cols: usize,
    ) -> Result<Tensor> {
        let out = self.forward(pixel_values, grid_rows, grid_cols)?;
        out.patch_shared.squeeze(0)
    }
}
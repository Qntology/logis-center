pub mod vision;
pub mod text;
pub mod preprocessor;
pub mod vision_crop;
pub mod vision_encoder;
// 🌟 [STEP 2.5] 패치 격자 단위 판독 가능성 지도 (블러 / 마스킹 / 여백 판정)
pub mod legibility;
// 🌟 [STEP 6] 추출값이 크롭 안에 실제로 인쇄되어 있는지 검증
pub mod value_grounding;
pub mod tokenizer;


use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::path::Path;

/// SigLIP2 통합 모델 (비전 인코더 + 텍스트 인코더 + 토크나이저)
///
/// ── 텐서 계약 (print_tensors.py 실측 대조 완료) ──
///   vision_model.embeddings.patch_embedding.weight   [1152, 768]   ← Linear (16*16*3)
///   vision_model.embeddings.position_embedding.weight[256, 1152]   ← 16×16 격자
///   vision_model.encoder.layers.{0..26}.*
///   vision_model.post_layernorm.*
///   vision_model.head.probe                          [1, 1, 1152]
///   vision_model.head.attention.in_proj_weight       [3456, 1152]  ← q|k|v concat
///   vision_model.head.attention.out_proj.*
///   vision_model.head.layernorm.*  /  head.mlp.fc1,fc2
///   text_model.embeddings.token_embedding.weight     [256000, 1152]
///   text_model.embeddings.position_embedding.weight  [64, 1152]    ← seq len 64 고정
///   text_model.final_layer_norm.*
///   text_model.head.weight/bias                      [1152,1152] / [1152]
///   logit_scale / logit_bias                         [1] / [1]
pub struct Siglip2Model {
    /// 🌟 [OPTIONAL VISION] 텍스트 전용 로드를 허용하기 위해 Option 으로 둡니다.
    ///
    ///  ── 왜 필요한가 ──
    ///   STEP 6 값 접지 검증과 검색 질의 벡터 생성은 텍스트 인코더만 씁니다.
    ///   패치 임베딩은 STEP 1 산출물(grid.patches ≈ 1.2MB)이 CPU 메모리에 이미 있고,
    ///   질의 벡터는 애초에 이미지와 무관합니다.
    ///   구조체가 비전을 필수로 요구하면 그 두 경로가 항상 820MB 를 함께 올려야 했습니다.
    ///   특히 검색은 질의마다 반복되므로 누적 비용이 큽니다.
    pub vision: Option<vision::Siglip2VisionModel>,
    pub text: Option<text::Siglip2TextModel>,
    pub tokenizer: Option<tokenizer::Siglip2Tokenizer>,
    pub logit_scale: f32,
    pub logit_bias: f32,
    pub device: Device,
    pub dtype: DType,
    pub config: Siglip2Config,
}

/// 모델 설정 (config.json에서 파싱)
#[derive(Debug, Clone)]
pub struct Siglip2Config {
    // vision_config
    pub vision_hidden_size: usize,       // 1152
    pub vision_intermediate_size: usize, // 4304
    pub vision_num_layers: usize,        // 27
    pub vision_num_heads: usize,         // 16
    pub patch_size: usize,               // 16
    pub max_num_patches: usize,          // 256 (NaFlex)
    pub vision_layer_norm_eps: f64,      // 1e-6
    // text_config
    pub text_hidden_size: usize,         // 1152
    pub text_intermediate_size: usize,   // 4304
    pub text_num_layers: usize,          // 27
    pub text_num_heads: usize,           // 16
    pub text_vocab_size: usize,          // 256000
    pub text_max_positions: usize,       // 64  ← 512 아님
    pub text_pad_token_id: u32,          // 1
    pub text_layer_norm_eps: f64,        // 1e-6
}

impl Siglip2Config {
    pub fn from_json(config_path: &Path) -> anyhow::Result<Self> {
        let raw = std::fs::read_to_string(config_path)?;
        let v: serde_json::Value = serde_json::from_str(&raw)?;
        let vc = &v["vision_config"];
        let tc = &v["text_config"];
        Ok(Self {
            vision_hidden_size: vc["hidden_size"].as_u64().unwrap_or(1152) as usize,
            vision_intermediate_size: vc["intermediate_size"].as_u64().unwrap_or(4304) as usize,
            vision_num_layers: vc["num_hidden_layers"].as_u64().unwrap_or(27) as usize,
            vision_num_heads: vc["num_attention_heads"].as_u64().unwrap_or(16) as usize,
            // 🌟 하드코딩 폐기: config 의 patch_size / num_patches 를 실제로 읽습니다.
            patch_size: vc["patch_size"].as_u64().unwrap_or(16) as usize,
            max_num_patches: vc["num_patches"].as_u64().unwrap_or(256) as usize,
            vision_layer_norm_eps: vc["layer_norm_eps"].as_f64().unwrap_or(1e-6),
            text_hidden_size: tc["hidden_size"].as_u64().unwrap_or(1152) as usize,
            text_intermediate_size: tc["intermediate_size"].as_u64().unwrap_or(4304) as usize,
            text_num_layers: tc["num_hidden_layers"].as_u64().unwrap_or(27) as usize,
            text_num_heads: tc["num_attention_heads"].as_u64().unwrap_or(16) as usize,
            text_vocab_size: tc["vocab_size"].as_u64().unwrap_or(256000) as usize,
            // 🌟 실측 텐서가 [64, 1152] 이므로 기본값 64.
            text_max_positions: tc["max_position_embeddings"].as_u64().unwrap_or(64) as usize,
            text_pad_token_id: tc["pad_token_id"].as_u64().unwrap_or(1) as u32,
            text_layer_norm_eps: tc["layer_norm_eps"].as_f64().unwrap_or(1e-6),
        })
    }

    /// 위치 임베딩 격자 한 변의 길이. 256 → 16.
    pub fn pos_grid_side(&self) -> usize {
        let s = (self.max_num_patches as f64).sqrt().round() as usize;
        s.max(1)
    }
}

impl Siglip2Model {
    /// 비전 인코더만 mmap 으로 로드합니다.
    ///
    /// 🌟 [MEMORY] 기존 `candle_core::safetensors::load()` 는 4.3GB 전체를
    ///    HashMap 으로 올린 뒤 vision_model.* 만 골라냈습니다.
    ///    mmap 백엔드는 요청한 텐서만 페이지 인 하므로 상주량이 실제 사용분으로 제한됩니다.
    pub fn load_vision_only(
        safetensors_path: &Path,
        config: &Siglip2Config,
        device: &Device,
        dtype: DType,
    ) -> anyhow::Result<Self> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[safetensors_path], dtype, device)?
        };

        let vision_model = vision::Siglip2VisionModel::new(config, vb.pp("vision_model"))?;

        // logit_scale / logit_bias 는 루트 스칼라입니다. 없으면 기본값.
        let logit_scale = vb
            .get(1, "logit_scale")
            .ok()
            .and_then(|t| t.to_dtype(DType::F32).ok())
            .and_then(|t| t.to_vec1::<f32>().ok())
            .and_then(|v| v.first().copied())
            .unwrap_or(0.0);
        let logit_bias = vb
            .get(1, "logit_bias")
            .ok()
            .and_then(|t| t.to_dtype(DType::F32).ok())
            .and_then(|t| t.to_vec1::<f32>().ok())
            .and_then(|v| v.first().copied())
            .unwrap_or(0.0);

        println!(
            "[SigLIP2] Vision encoder loaded (layers={}, hidden={}, patch={}, max_patches={}, logit_scale={:.4})",
            config.vision_num_layers,
            config.vision_hidden_size,
            config.patch_size,
            config.max_num_patches,
            logit_scale
        );

        Ok(Self {
            vision: Some(vision_model),
            text: None,
            tokenizer: None,
            logit_scale,
            logit_bias,
            device: device.clone(),
            dtype,
            config: config.clone(),
        })
    }

    /// 🌟 텍스트 인코더 + 토크나이저만 로드합니다. 비전 가중치는 올리지 않습니다.
    ///
    ///  ── 언제 쓰는가 ──
    ///   · STEP 6 값 접지 검증 : 값 텍스트만 인코딩. 패치는 STEP 1 산출물을 재사용.
    ///   · 검색 질의 벡터 생성 : 질의는 텍스트이므로 비전이 애초에 불필요.
    ///   두 경로 모두 load_vision_only 를 거치면 820MB 를 헛되이 점유합니다.
    ///
    ///  ── mmap ──
    ///   VarBuilder::from_mmaped_safetensors 는 요청한 텐서만 페이지 인 하므로,
    ///   같은 model.safetensors 를 열어도 text_model.* 만 실제로 상주합니다.
    pub fn load_text_only(
        model_dir: &Path,
        config: &Siglip2Config,
        device: &Device,
        dtype: DType,
    ) -> anyhow::Result<Self> {
        let safetensors_path = model_dir.join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&safetensors_path], dtype, device)?
        };

        let text_model = text::Siglip2TextModel::new(config, vb.pp("text_model"))?;

        let logit_scale = vb
            .get(1, "logit_scale")
            .ok()
            .and_then(|t| t.to_dtype(DType::F32).ok())
            .and_then(|t| t.to_vec1::<f32>().ok())
            .and_then(|v| v.first().copied())
            .unwrap_or(0.0);
        let logit_bias = vb
            .get(1, "logit_bias")
            .ok()
            .and_then(|t| t.to_dtype(DType::F32).ok())
            .and_then(|t| t.to_vec1::<f32>().ok())
            .and_then(|v| v.first().copied())
            .unwrap_or(0.0);

        let tok = tokenizer::Siglip2Tokenizer::from_dir(
            model_dir,
            config.text_pad_token_id,
            config.text_max_positions,
        )?;

        println!(
            "[SigLIP2] Text-only mode loaded (layers={}, vocab={}, seq_len={}). Vision weights NOT loaded (~820MB saved).",
            config.text_num_layers,
            config.text_vocab_size,
            config.text_max_positions
        );

        Ok(Self {
            vision: None,
            text: Some(text_model),
            tokenizer: Some(tok),
            logit_scale,
            logit_bias,
            device: device.clone(),
            dtype,
            config: config.clone(),
        })
    }

    /// 🌟 비전 인코더 가중치를 나중에 부착합니다.
    ///    텍스트 전용으로 올린 인스턴스에 이미지 처리가 필요해졌을 때 사용합니다.
    pub fn load_vision_encoder(&mut self, model_dir: &Path) -> anyhow::Result<()> {
        if self.vision.is_some() {
            return Ok(());
        }
        let safetensors_path = model_dir.join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&safetensors_path], self.dtype, &self.device)?
        };
        let vision_model = vision::Siglip2VisionModel::new(&self.config, vb.pp("vision_model"))?;
        self.vision = Some(vision_model);
        println!(
            "[SigLIP2] Vision encoder ATTACHED to existing instance (layers={}, patch={}).",
            self.config.vision_num_layers, self.config.patch_size
        );
        Ok(())
    }

    /// 텍스트 인코더 + 토크나이저를 추가로 로드합니다.
    ///
    /// `model_dir` 에는 model.safetensors 와 tokenizer.json 이 함께 있어야 합니다.
    pub fn load_text_encoder(&mut self, model_dir: &Path) -> anyhow::Result<()> {
        let safetensors_path = model_dir.join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&safetensors_path], self.dtype, &self.device)?
        };

        let text_model = text::Siglip2TextModel::new(&self.config, vb.pp("text_model"))?;
        self.text = Some(text_model);

        let tok = tokenizer::Siglip2Tokenizer::from_dir(
            model_dir,
            self.config.text_pad_token_id,
            self.config.text_max_positions,
        )?;
        self.tokenizer = Some(tok);

        println!(
            "[SigLIP2] Text encoder + tokenizer loaded (layers={}, vocab={}, seq_len={}, pad_id={})",
            self.config.text_num_layers,
            self.config.text_vocab_size,
            self.config.text_max_positions,
            self.config.text_pad_token_id
        );
        Ok(())
    }

    pub fn has_text(&self) -> bool {
        self.text.is_some() && self.tokenizer.is_some()
    }

    /// 🌟 비전 인코더가 실제로 상주 중인지 확인합니다.
    ///    ensure_siglip2 가 '요구 사양과 현재 상태' 를 비교할 때 씁니다.
    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }
}
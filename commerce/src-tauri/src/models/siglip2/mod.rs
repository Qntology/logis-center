pub mod vision;
pub mod text;
pub mod preprocessor;
pub mod vision_crop;
pub mod vision_encoder;
pub mod legibility;
pub mod nms_arena;
pub mod value_grounding;
pub mod tokenizer;
pub mod phrase_cache;


use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use std::path::Path;

pub struct Siglip2Model {
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
            patch_size: vc["patch_size"].as_u64().unwrap_or(16) as usize,
            max_num_patches: vc["num_patches"].as_u64().unwrap_or(256) as usize,
            vision_layer_norm_eps: vc["layer_norm_eps"].as_f64().unwrap_or(1e-6),
            text_hidden_size: tc["hidden_size"].as_u64().unwrap_or(1152) as usize,
            text_intermediate_size: tc["intermediate_size"].as_u64().unwrap_or(4304) as usize,
            text_num_layers: tc["num_hidden_layers"].as_u64().unwrap_or(27) as usize,
            text_num_heads: tc["num_attention_heads"].as_u64().unwrap_or(16) as usize,
            text_vocab_size: tc["vocab_size"].as_u64().unwrap_or(256000) as usize,
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
        let embed_vb = if device.is_cpu() {
            None
        } else {
            match unsafe {
                VarBuilder::from_mmaped_safetensors(&[&safetensors_path], dtype, &Device::Cpu)
            } {
                Ok(v) => Some(v.pp("text_model")),
                Err(e) => {
                    println!(
                        "[SigLIP2] CPU 임베딩 VarBuilder 생성 실패({}). token_embedding 을 VRAM 에 유지합니다.",
                        e
                    );
                    None
                }
            }
        };
        let text_model = text::Siglip2TextModel::new(config, vb.pp("text_model"), embed_vb)?;

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

    pub fn load_text_encoder(&mut self, model_dir: &Path) -> anyhow::Result<()> {
        let safetensors_path = model_dir.join("model.safetensors");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&safetensors_path], self.dtype, &self.device)?
        };

        // 🌟 [CPU EMBEDDING] load_text_only 와 동일 정책. 590MB 를 VRAM 에서 뺍니다.
        let embed_vb = if self.device.is_cpu() {
            None
        } else {
            match unsafe {
                VarBuilder::from_mmaped_safetensors(&[&safetensors_path], self.dtype, &Device::Cpu)
            } {
                Ok(v) => Some(v.pp("text_model")),
                Err(e) => {
                    println!(
                        "[SigLIP2] CPU 임베딩 VarBuilder 생성 실패({}). token_embedding 을 VRAM 에 유지합니다.",
                        e
                    );
                    None
                }
            }
        };
        let text_model = text::Siglip2TextModel::new(&self.config, vb.pp("text_model"), embed_vb)?;
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

    pub fn has_vision(&self) -> bool {
        self.vision.is_some()
    }

    pub fn detach_vision(&mut self) -> bool {
        if self.vision.is_none() {
            return false;
        }
        self.vision = None;
        println!("[SigLIP2] Vision encoder DETACHED (약 856MB VRAM 반납).");
        true
    }

    pub fn detach_text(&mut self) -> bool {
        if self.text.is_none() && self.tokenizer.is_none() {
            return false;
        }
        self.text = None;
        self.tokenizer = None;
        println!("[SigLIP2] Text encoder DETACHED.");
        true
    }
}
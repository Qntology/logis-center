pub mod vision;
pub mod text;
pub mod preprocessor;

use candle_core::{Device, DType, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

/// SigLIP2 통합 모델 (비전 인코더 + 텍스트 인코더)
/// 파이프라인에서 비전 인코더만 사용하므로 텍스트는 옵션입니다.
pub struct Siglip2Model {
    pub vision: vision::Siglip2VisionModel,
    pub text: Option<text::Siglip2TextModel>,
    pub device: Device,
    pub dtype: DType,
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
    // text_config
    pub text_hidden_size: usize,         // 1152
    pub text_intermediate_size: usize,   // 4304
    pub text_num_layers: usize,          // 27
    pub text_num_heads: usize,           // 16
    pub text_vocab_size: usize,          // 256000
    pub text_projection_size: usize,     // 1152
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
            patch_size: 16,
            max_num_patches: 256,
            text_hidden_size: tc["hidden_size"].as_u64().unwrap_or(1152) as usize,
            text_intermediate_size: tc["intermediate_size"].as_u64().unwrap_or(4304) as usize,
            text_num_layers: tc["num_hidden_layers"].as_u64().unwrap_or(27) as usize,
            text_num_heads: tc["num_attention_heads"].as_u64().unwrap_or(16) as usize,
            text_vocab_size: tc["vocab_size"].as_u64().unwrap_or(256000) as usize,
            text_projection_size: tc["projection_size"].as_u64().unwrap_or(1152) as usize,
        })
    }
}

impl Siglip2Model {
    /// 원본 safetensors에서 비전 인코더만 선택 로드합니다.
    /// 텍스트 인코더는 필요 시 별도 로드합니다.
    pub fn load_vision_only(
        safetensors_path: &Path,
        config: &Siglip2Config,
        device: &Device,
        dtype: DType,
    ) -> anyhow::Result<Self> {
        let all_tensors = candle_core::safetensors::load(safetensors_path, device)?;

        // "vision_model." 프리픽스 텐서만 추출
        let mut vision_tensors = std::collections::HashMap::new();
        for (name, tensor) in all_tensors.iter() {
            if name.starts_with("vision_model.") {
                let short_name = name.strip_prefix("vision_model.").unwrap_or(name);
                vision_tensors.insert(short_name.to_string(), tensor.clone());
            }
        }

        if vision_tensors.is_empty() {
            return Err(anyhow::anyhow!(
                "No vision_model.* tensors found in safetensors"
            ));
        }

        println!(
            "[SigLIP2] Loaded {} vision tensors from safetensors",
            vision_tensors.len()
        );

        let vb = VarBuilder::from_tensors(vision_tensors, dtype, device);
        let vision_model = vision::Siglip2VisionModel::new(config, vb)?;

        Ok(Self {
            vision: vision_model,
            text: None,
            device: device.clone(),
            dtype,
        })
    }

    /// 텍스트 인코더를 추가로 로드합니다.
    pub fn load_text_encoder(
        &mut self,
        safetensors_path: &Path,
        config: &Siglip2Config,
    ) -> anyhow::Result<()> {
        let all_tensors = candle_core::safetensors::load(safetensors_path, &self.device)?;

        let mut text_tensors = std::collections::HashMap::new();
        for (name, tensor) in all_tensors.iter() {
            if name.starts_with("text_model.") {
                let short_name = name.strip_prefix("text_model.").unwrap_or(name);
                text_tensors.insert(short_name.to_string(), tensor.clone());
            }
        }

        if text_tensors.is_empty() {
            return Err(anyhow::anyhow!(
                "No text_model.* tensors found in safetensors"
            ));
        }

        println!(
            "[SigLIP2] Loaded {} text tensors from safetensors",
            text_tensors.len()
        );

        let vb = VarBuilder::from_tensors(text_tensors, self.dtype, &self.device);
        let text_model = text::Siglip2TextModel::new(config, vb)?;
        self.text = Some(text_model);
        Ok(())
    }
}
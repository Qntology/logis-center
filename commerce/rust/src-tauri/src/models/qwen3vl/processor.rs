use std::collections::HashMap;
use crate::{
    models::qwen3vl::config::PreprocessorConfig,
    openai_types::{
        ChatCompletionParameters, ChatCompletionRequestMessage,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestMessageContentPart,
    },
};
use anyhow::{Result, anyhow};
use image::DynamicImage;
use half::f16;

pub struct Qwen3VLProcessor {
    img_process_cfg: PreprocessorConfig,
    image_token: String,
}

impl Qwen3VLProcessor {
    pub fn new_native(path: &str) -> Result<Self> {
        let img_process_cfg_file = std::path::Path::new(path).join("preprocessor_config.json");
        let img_process_cfg: PreprocessorConfig = if img_process_cfg_file.exists() {
            serde_json::from_slice(&std::fs::read(img_process_cfg_file)?)?
        } else {
            PreprocessorConfig::default()
        };

        Ok(Self {
            img_process_cfg,
            image_token: "<|image_pad|>".to_string(),
        })
    }

    pub fn process_info_native(&self, _mes: &ChatCompletionParameters, render: &str) -> Result<NativeInputInfo> {
        // [SIMPLIFIED] Native 이미지 전처리 로직은 필요 시 native_backend의 커널로 구현
        // 현재는 텍스트만 전달하는 구조로 우선 정리
        Ok(NativeInputInfo {
            replace_text: render.to_string(),
            pixel_values: None,
            image_grid_thw: None,
        })
    }
}

pub struct NativeInputInfo {
    pub replace_text: String,
    pub pixel_values: Option<Vec<f16>>,
    pub image_grid_thw: Option<[u32; 3]>,
}
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

    pub fn process_info_native(&self, mes: &ChatCompletionParameters, render: &str) -> Result<NativeInputInfo> {
        let mut pixel_values = None;
        let mut image_grid_thw = None;

        // 1. Search for images in messages
        for m in &mes.messages {
            if let ChatCompletionRequestMessage::User(um) = m {
                match &um.content {
                    ChatCompletionRequestUserMessageContent::Array(parts) => {
                        for p in parts {
                            if let ChatCompletionRequestMessageContentPart::ImageURL(img_part) = p {
                                let url = &img_part.image_url.url;
                                if url.starts_with("data:image/") {
                                    let b64 = url.split(',').nth(1).ok_or(anyhow!("Invalid data URL"))?;
                                    use base64::Engine;
                                    let bytes = base64::prelude::BASE64_STANDARD.decode(b64)?;
                                    let img = image::load_from_memory(&bytes)?;
                                    let (pv, thw) = self.preprocess_image(img)?;
                                    pixel_values = Some(pv);
                                    image_grid_thw = Some(thw);
                                } else if std::path::Path::new(url).exists() {
                                    let img = image::open(url)?;
                                    let (pv, thw) = self.preprocess_image(img)?;
                                    pixel_values = Some(pv);
                                    image_grid_thw = Some(thw);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(NativeInputInfo {
            replace_text: render.to_string(),
            pixel_values,
            image_grid_thw,
        })
    }

    fn preprocess_image(&self, img: DynamicImage) -> Result<(Vec<f16>, [u32; 3])> {
        let rgb = img.to_rgb8();
        let (w, h) = rgb.dimensions();
        
        // Qwen3-VL standard: factor of 28
        let new_w = ((w as f32 / 28.0).round() * 28.0) as u32;
        let new_h = ((h as f32 / 28.0).round() * 28.0) as u32;
        
        let resized = image::imageops::resize(&rgb, new_w, new_h, image::imageops::FilterType::Triangle);
        
        let mut pixels = Vec::with_capacity((new_w * new_h * 3) as usize);
        for p in resized.pixels() {
            for &c in &[p[0], p[1], p[2]] {
                let val = (c as f32 / 255.0 - 0.48145466) / 0.26862954; // Mean/Std normalization
                pixels.push(f16::from_f32(val));
            }
        }

        let grid_t = 1u32; // Temporal (static image)
        let grid_h = new_h / 14; 
        let grid_w = new_w / 14;

        Ok((pixels, [grid_t, grid_h, grid_w]))
    }
}

pub struct NativeInputInfo {
    pub replace_text: String,
    pub pixel_values: Option<Vec<f16>>,
    pub image_grid_thw: Option<[u32; 3]>,
}
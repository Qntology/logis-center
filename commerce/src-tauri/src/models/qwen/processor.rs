use std::collections::HashMap;

use crate::{
    models::qwen::config::PreprocessorConfig,
    openai_types::{
        ChatCompletionParameters, ChatCompletionRequestMessage,
        ChatCompletionRequestUserMessageContent, ChatCompletionRequestMessageContentPart,
    },
    utils::{
        img_utils::{get_image, img_smart_resize, img_transform},
    },
};
use anyhow::Result;
use candle_core::{DType, Device, Shape, Tensor};
use image::DynamicImage;
use sysinfo::System;

#[derive(Clone)]
pub struct VisionInput {
    pub data: Tensor,
    pub grid_thw: Tensor,
}

#[derive(Clone)]
pub struct GeneralInput {
    pub replace_text: String,
    pub pixel_values: Option<Tensor>,
    pub image_grid_thw: Option<Tensor>,
    pub pixel_values_video: Option<Tensor>,
    pub video_grid_thw: Option<Tensor>,
}

#[allow(unused)]
pub struct QwenVLProcessor {
    img_process_cfg: PreprocessorConfig,
    device: Device,
    dtype: DType,
    image_token: String,
    video_token: String,
    vision_start_token: String,
    vision_end_token: String,
}

impl QwenVLProcessor {
    pub fn new(path: &str, device: &Device, dtype: DType) -> Result<Self> {
        let img_process_cfg_file = std::path::Path::new(path).join("preprocessor_config.json");
        let img_process_cfg: PreprocessorConfig = if img_process_cfg_file.exists() {
            serde_json::from_slice(&std::fs::read(img_process_cfg_file)?)?
        } else {
            PreprocessorConfig::default()
        };

        let image_token = "<|image_pad|>".to_string();
        let video_token = "<|video_pad|>".to_string();
        let vision_start_token = "<|vision_start|>".to_string();
        let vision_end_token = "<|vision_end|>".to_string();
        Ok(Self {
            img_process_cfg,
            device: device.clone(),
            dtype,
            image_token,
            video_token,
            vision_start_token,
            vision_end_token,
        })
    }

    pub fn extract_vision_info(
        &self,
        mes: &ChatCompletionParameters,
    ) -> Result<HashMap<String, Vec<String>>> {
        let mut vision_map = HashMap::new();
        vision_map.insert("image".to_string(), Vec::new());
        vision_map.insert("video".to_string(), Vec::new());
        for chat_mes in mes.messages.clone() {
            match chat_mes {
                ChatCompletionRequestMessage::User(user_msg) => {
                    if let ChatCompletionRequestUserMessageContent::Array(parts) = user_msg.content {
                        for part in parts {
                            if let ChatCompletionRequestMessageContentPart::ImageURL(img_part) = part {
                                vision_map.get_mut("image").unwrap().push(img_part.image_url.url);
                            } 
                            // Video support removed for simplicity/compilation
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(vision_map)
    }

    /// 🌟 [VRAM 역산] Qwen3VL 프로세서와 동일한 이차식 모델을 사용합니다.
    ///   mem(N) = A·N + B·N²  →  B·N² + A·N - usable = 0 의 양근이 안전 패치 수입니다.
    fn compute_adaptive_max_pixels(&self, config_max: u32) -> u32 {
        const A: f64 = 24_000.0;
        // 🌟 [VISION-TILE 반영] 쿼리축 타일링으로 어텐션 전이 버퍼에 상한이 걸렸으므로
        //   N² 계수를 32 → 4 로 낮춥니다. (qwen3vl/processor.rs 와 동일 근거)
        const B: f64 = 4.0;
        const RESERVE: u64 = 800_000_000;

        let patch_area = (self.img_process_cfg.patch_size * self.img_process_cfg.patch_size) as u64;
        if patch_area == 0 { return config_max; }

        let nvml = match nvml_wrapper::Nvml::init() {
            Ok(n) => n,
            Err(_) => return config_max,
        };
        let dev = match nvml.device_by_index(0) {
            Ok(d) => d,
            Err(_) => return config_max,
        };
        let mem = match dev.memory_info() {
            Ok(m) => m,
            Err(_) => return config_max,
        };

        let free_vram = mem.free;
        let usable = free_vram.saturating_sub(RESERVE);
        if usable == 0 {
            let floor_px = 1_048_576u32;
            println!(
                "[PROCESSOR] Free VRAM {:.0}MB is below reserve. Forcing minimum {}px².",
                free_vram as f64 / 1e6, floor_px
            );
            return config_max.min(floor_px);
        }

        let disc = (A * A + 4.0 * B * usable as f64).sqrt();
        let n = ((disc - A) / (2.0 * B)) as u64;
        if n == 0 { return config_max.min(1_048_576); }

        let adaptive = n.saturating_mul(patch_area);
        let adaptive = if adaptive > u32::MAX as u64 { u32::MAX } else { adaptive as u32 };

        if adaptive < config_max {
            let side = (adaptive as f64).sqrt() as u32;
            println!(
                "[PROCESSOR] Free VRAM {:.0}MB (usable {:.0}MB) → max {} patches → max_pixels {} (~{}x{}).",
                free_vram as f64 / 1e6, usable as f64 / 1e6, n, adaptive, side, side
            );
            adaptive
        } else {
            config_max
        }
    }

    pub fn process_img(
        &self,
        img: &DynamicImage,
        img_mean: &Tensor,
        img_std: &Tensor,
    ) -> Result<Tensor> {
        self.process_img_with_budget(img, img_mean, img_std, 1)
    }

    /// 🌟 [VRAM 역산] budget_divisor 는 이 배치에서 동시 상주할 이미지 수입니다.
    pub fn process_img_with_budget(
        &self,
        img: &DynamicImage,
        img_mean: &Tensor,
        img_std: &Tensor,
        budget_divisor: usize,
    ) -> Result<Tensor> {
        let img_h = img.height();
        let img_w = img.width();

        // 🌟 [BUGFIX] 기존 코드는 img_smart_resize 의 max_pixels(면적) 인자에
        //    1024 / 896 / 768 이라는 '변 길이' 를 넣고 있었습니다.
        //    768px² = 0.75 패치라 사실상 모든 이미지가 최소 크기로 뭉개졌습니다.
        //    면적 단위로 통일하고 이차식 역산으로 교체합니다.
        let mut max_pixels = self.img_process_cfg.size.longest_edge as u32;
        let shortest_edge = self.img_process_cfg.size.shortest_edge as u32;

        if max_pixels > 16_777_216 { max_pixels = 16_777_216; }

        // 1. System RAM Check (면적 기준으로 환산)
        let mut sys = System::new_all();
        sys.refresh_memory();
        let free_ram = sys.available_memory();
        if free_ram < 2_000_000_000 {
            let cap = 3_145_728u32; // 약 1772x1772 의 '면적' (OCR 품질 보존을 위해 상향)
            if max_pixels > cap {
                println!("[PROCESSOR] Low RAM ({:.2} GB). Capping to {}px².", free_ram as f64 / 1e9, cap);
                max_pixels = cap;
            }
        }

        // 2. VRAM 이차식 역산
        let adaptive = self.compute_adaptive_max_pixels(max_pixels);
        if adaptive < max_pixels { max_pixels = adaptive; }

        // 3. 다중 이미지 배치 보정
        let divisor = budget_divisor.max(1);
        if divisor > 1 {
            let shared = max_pixels / divisor as u32;
            let floor_px = 1_048_576u32;
            let shared = shared.max(floor_px.min(max_pixels));
            if shared < max_pixels {
                println!("[PROCESSOR] Batch of {} images → per-image max_pixels {} (was {}).", divisor, shared, max_pixels);
                max_pixels = shared;
            }
        }

        let shortest_edge = if shortest_edge > max_pixels { max_pixels } else { shortest_edge };

        let (resize_h, resize_w) = img_smart_resize(
            img_h,
            img_w,
            (self.img_process_cfg.patch_size * self.img_process_cfg.merge_size) as u32,
            shortest_edge,
            max_pixels,
        )?;
        // ----------------------------------

        let img = img.resize_exact(resize_w, resize_h, image::imageops::FilterType::Lanczos3);
        let img_tensor = img_transform(&img, img_mean, img_std, &self.device, self.dtype)?;
        let img_tensor = img_tensor.unsqueeze(0)?;
        Ok(img_tensor)
    }

    pub fn process_vision_tensor(&self, img_tensor: &Tensor) -> Result<(Tensor, Tensor)> {
        let t = img_tensor.dim(0)?;
        let img_tensor = if t % self.img_process_cfg.temporal_patch_size != 0 {
            let repeat_num = self.img_process_cfg.temporal_patch_size
                - t % self.img_process_cfg.temporal_patch_size;
                
            // [CRITICAL FIX] .i() 대신 narrow를 사용하여 4D 차원(1, C, H, W)을 안전하게 보존!
            let repeats = img_tensor.narrow(0, t - 1, 1)?.repeat((repeat_num, 1, 1, 1))?;
            Tensor::cat(&[img_tensor, &repeats], 0)?
        } else {
            img_tensor.clone()
        };
        let channel = img_tensor.dim(1)?;
        let grid_t = img_tensor.dim(0)? / self.img_process_cfg.temporal_patch_size;
        let grid_h = img_tensor.dim(2)? / self.img_process_cfg.patch_size;
        let grid_w = img_tensor.dim(3)? / self.img_process_cfg.patch_size;
        let shape = Shape::from(vec![
            grid_t,
            self.img_process_cfg.temporal_patch_size,
            channel,
            grid_h / self.img_process_cfg.merge_size,
            self.img_process_cfg.merge_size,
            self.img_process_cfg.patch_size,
            grid_w / self.img_process_cfg.merge_size,
            self.img_process_cfg.merge_size,
            self.img_process_cfg.patch_size,
        ]);
        let img_tensor = img_tensor.reshape(shape)?;
        
        // [CRITICAL FIX] permute 이후에는 메모리가 비연속적으로 변하므로, 
        // 다음 reshape을 수행하기 전에 반드시 contiguous()를 호출해야 프레임워크가 뻗지(Crash) 않습니다!
        let img_tensor = img_tensor.permute(vec![0, 3, 6, 4, 7, 2, 1, 5, 8])?.contiguous()?; 
        
        let img_tensor = img_tensor
            .reshape((
                grid_t * grid_h * grid_w,
                channel * self.img_process_cfg.temporal_patch_size * self.img_process_cfg.patch_size * self.img_process_cfg.patch_size,
            ))?; // 끝에 있던 contiguous()는 위로 당겨졌으므로 여기선 삭제
        let grid_thw = Tensor::from_vec(
            vec![grid_t as u32, grid_h as u32, grid_w as u32],
            (1, 3),
            &self.device,
        )?;
        Ok((img_tensor, grid_thw))
    }

    pub fn process_images(
        &self,
        imgs: Vec<DynamicImage>,
        img_mean: &Tensor,
        img_std: &Tensor,
    ) -> Result<VisionInput> {
        let mut pixel_values_vec = Vec::new();
        let mut vision_grid_thws_vec = Vec::new();

        // 🌟 [VRAM 역산] 결과가 최종적으로 Tensor::cat 으로 합쳐지므로 장수를 예산 분모로 넘깁니다.
        let total = imgs.len().max(1);
        if total > 1 {
            println!("[PROCESSOR] process_images: {} images will co-reside. Splitting VRAM budget.", total);
        }

        for img in imgs {
            let raw = self.process_img_with_budget(&img, img_mean, img_std, total)?;
            let (patched, grid_thw) = self.process_vision_tensor(&raw)?;

            // 🌟 [VRAM] reshape/permute 소스는 patched 확보 직후 폐기합니다.
            drop(raw);

            pixel_values_vec.push(patched);
            vision_grid_thws_vec.push(grid_thw);

            if self.device.is_cuda() {
                let _ = self.device.synchronize();
            }
        }

        let pixel_values = Tensor::cat(&pixel_values_vec, 0)?;
        let vision_grid_thws = Tensor::cat(&vision_grid_thws_vec, 0)?;

        drop(pixel_values_vec);
        drop(vision_grid_thws_vec);
        if self.device.is_cuda() {
            let _ = self.device.synchronize();
        }

        Ok(VisionInput {
            data: pixel_values,
            grid_thw: vision_grid_thws,
        })
    }

    pub fn process_info(
        &self,
        messages: &ChatCompletionParameters,
        text: &str,
    ) -> Result<GeneralInput> {
        let mut pixel_values = None;
        let mut image_grid_thw = None;
        let pixel_values_video = None;
        let video_grid_thw: Option<Tensor> = None;

        // [BRANCH] If the model doesn't define an image token, it's pure text. Skip vision processing.
        if self.image_token.is_empty() || self.image_token == "" {
            return Ok(GeneralInput {
                replace_text: text.to_string(),
                pixel_values: None,
                image_grid_thw: None,
                pixel_values_video: None,
                video_grid_thw: None,
            });
        }

        let vision_map = self.extract_vision_info(messages)?;
        let img_mean =
            Tensor::from_slice(&self.img_process_cfg.image_mean, (3, 1, 1), &self.device)?
                .to_dtype(self.dtype)?;
        let img_std = Tensor::from_slice(&self.img_process_cfg.image_std, (3, 1, 1), &self.device)?
            .to_dtype(self.dtype)?;
        
        for (key, vec) in vision_map {
            if key.eq("image") {
                let mut file_vec = Vec::new();
                for file in &vec {
                    let image = get_image(file);
                    match image {
                        Ok(img) => file_vec.push(img),
                        Err(e) => println!("get_image err: {e:?}"),
                    };
                }
                if !file_vec.is_empty() {
                    let vision_input = self.process_images(file_vec, &img_mean, &img_std);
                    match vision_input {
                        Ok(img_input) => {
                            pixel_values = Some(img_input.data);
                            image_grid_thw = Some(img_input.grid_thw);
                        }
                        Err(e) => println!("img process_images err: {e:?}"),
                    };
                }
            }
        }
        let merge_length = self.img_process_cfg.merge_size.pow(2);
        let mut text = text.to_string();
        if let Some(ref image_grid_thw) = image_grid_thw {
            // [CRITICAL FIX] while 루프 안에서 매번 GPU를 멈추는 to_vec1 호출을 막기 위해 
            // 단 한 번만 통째로 CPU 캐시 배열로 복사해 둡니다!
            let grid_thw_cpu = image_grid_thw.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
            let mut index = 0;
            
            while text.contains(&self.image_token) {
                // [FIX] O(1) CPU 배열 직접 접근으로 GPU 스톨(Stall) 완벽 제거
                let grid_i = &grid_thw_cpu[index];
                let repeat_num = grid_i.iter().product::<u32>() as usize / merge_length;
                let replace = "<|placeholder|>".repeat(repeat_num);
                text = text.replacen(&self.image_token, &replace, 1);
                index += 1;
            }
            text = text.replace("<|placeholder|>", &self.image_token);
        }
        
        let input = GeneralInput {
            replace_text: text,
            pixel_values,
            image_grid_thw,
            pixel_values_video,
            video_grid_thw,
        };
        Ok(input)
    }
}
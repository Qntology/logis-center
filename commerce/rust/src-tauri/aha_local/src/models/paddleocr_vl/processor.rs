use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Shape, Tensor};
use image::DynamicImage;

use crate::{
    models::paddleocr_vl::config::PaddleOCRVLPreprocessorConfig,
    utils::img_utils::{extract_images, img_smart_resize, img_transform},
};

pub struct PaddleOCRVLProcessor {
    process_cfg: PaddleOCRVLPreprocessorConfig,
    device: Device,
    dtype: DType,
    image_token: String,
}

impl PaddleOCRVLProcessor {
    pub fn new(
        config: PaddleOCRVLPreprocessorConfig,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let image_token = "<|IMAGE_PLACEHOLDER|>".to_string();
        Ok(Self {
            process_cfg: config,
            device: device.clone(),
            dtype,
            image_token,
        })
    }

    pub fn process_img(
        &self,
        img: &DynamicImage,
        img_mean: &Tensor,
        img_std: &Tensor,
    ) -> Result<Tensor> {
        let img_h = img.height();
        let img_w = img.width();
        //  h,w resize成 32的倍数
        let (resize_h, resize_w) = img_smart_resize(
            img_h,
            img_w,
            (self.process_cfg.patch_size * self.process_cfg.merge_size) as u32,
            self.process_cfg.min_pixels,
            self.process_cfg.max_pixels,
        )?;
        let img = img.resize_exact(resize_w, resize_h, image::imageops::FilterType::CatmullRom);
        let img_tensor = img_transform(&img, img_mean, img_std, &self.device, self.dtype)?;
        // (c, h, w) => (1, c, h, w)
        let img_tensor = img_tensor.unsqueeze(0)?;
        Ok(img_tensor)
    }

    pub fn process_vision_tensor(&self, img_tensor: &Tensor) -> Result<(Tensor, Tensor)> {
        let channel = img_tensor.dim(1)?;
        // img_temsor.dim[0] = 1, temporal_patch_size = 1, grid_t = 1
        let grid_t = img_tensor.dim(0)? / self.process_cfg.temporal_patch_size;
        let grid_h = img_tensor.dim(2)? / self.process_cfg.patch_size;
        let grid_w = img_tensor.dim(3)? / self.process_cfg.patch_size;
        let shape = Shape::from(vec![
            grid_t,
            self.process_cfg.temporal_patch_size,
            channel,
            grid_h,
            self.process_cfg.patch_size,
            grid_w,
            self.process_cfg.patch_size,
        ]);
        let img_tensor = img_tensor.reshape(shape)?;
        // shape to // grid_t,
        // grid_h,
        // grid_w,
        // channel,
        // temporal_patch_size
        // patch_size,
        // patch_size,
        let img_tensor = img_tensor.permute(vec![0, 3, 5, 2, 1, 4, 6])?;
        let img_tensor = img_tensor
            .reshape((
                grid_t * grid_h * grid_w,
                channel,
                self.process_cfg.patch_size,
                self.process_cfg.patch_size,
            ))?
            .contiguous()?;
        let grid_thw = Tensor::from_vec(
            vec![grid_t as u32, grid_h as u32, grid_w as u32],
            (1, 3),
            &self.device,
        )?;
        Ok((img_tensor, grid_thw))
    }

    pub fn process_images(
        &self,
        imgs: &Vec<DynamicImage>,
        img_mean: &Tensor,
        img_std: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let mut pixel_values_vec = Vec::new();
        let mut vision_grid_thws_vec = Vec::new();
        for img in imgs {
            let img_tensor = self.process_img(img, img_mean, img_std)?;
            let (img_tensor, grid_thw) = self.process_vision_tensor(&img_tensor)?;
            pixel_values_vec.push(img_tensor);
            vision_grid_thws_vec.push(grid_thw);
        }
        let pixel_values = Tensor::cat(&pixel_values_vec, 0)?;
        let vision_grid_thws = Tensor::cat(&vision_grid_thws_vec, 0)?;
        Ok((pixel_values, vision_grid_thws))
    }

    pub fn process_info(
        &self,
        messages: &ChatCompletionParameters,
        text: &str,
    ) -> Result<(String, Option<Tensor>, Option<Tensor>)> {
        let imgs = extract_images(messages)?;
        let img_mean = Tensor::from_slice(&self.process_cfg.image_mean, (3, 1, 1), &self.device)?
            .to_dtype(self.dtype)?;
        let img_std = Tensor::from_slice(&self.process_cfg.image_std, (3, 1, 1), &self.device)?
            .to_dtype(self.dtype)?;
        let (pixel_values, image_grid_thw) = if !imgs.is_empty() {
            let (pixel_values, image_grid_thw) = self.process_images(&imgs, &img_mean, &img_std)?;
            (Some(pixel_values), Some(image_grid_thw))
        } else {
            (None, None)
        };

        let merge_length = self.process_cfg.merge_size.pow(2);
        let mut text = text.to_string();
        if let Some(ref image_grid_thw) = image_grid_thw {
            let mut index = 0;
            while text.contains(&self.image_token) {
                let grid_i = image_grid_thw.i(index)?;
                let repeat_num =
                    grid_i.to_vec1::<u32>()?.iter().product::<u32>() as usize / merge_length;
                let replace = "<|placeholder|>".repeat(repeat_num);
                text = text.replacen(&self.image_token, &replace, 1);
                index += 1;
            }
            text = text.replace("<|placeholder|>", &self.image_token);
        }
        Ok((text, pixel_values, image_grid_thw))
    }
}

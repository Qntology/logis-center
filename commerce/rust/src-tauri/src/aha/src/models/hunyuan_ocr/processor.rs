use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Shape, Tensor};
use image::DynamicImage;

use crate::{
    models::hunyuan_ocr::config::HunyuanOCRPreprocessorConfig,
    tokenizer::TokenizerModel,
    utils::{
        img_utils::{extract_images, img_smart_resize, img_transform},
        tensor_utils::{get_eq_indices, get_equal_mask},
    },
};

pub struct HunyuanVLProcessor {
    image_token_id: u32,
    image_token: String,
    placeholder_token: String,
    process_cfg: HunyuanOCRPreprocessorConfig,
    device: Device,
    dtype: DType,
}

impl HunyuanVLProcessor {
    pub fn new(path: &str, device: &Device, dtype: DType) -> Result<Self> {
        let path = path.to_string();
        assert!(
            std::path::Path::new(&path).exists(),
            "model path file not exists"
        );
        let process_cfg_file = path.clone() + "/preprocessor_config.json";
        assert!(
            std::path::Path::new(&process_cfg_file).exists(),
            "preprocessor_config.json not exists in model path"
        );
        let process_cfg: HunyuanOCRPreprocessorConfig =
            serde_json::from_slice(&std::fs::read(process_cfg_file)?)?;
        let image_token_id = 120120u32;
        let image_token = "<｜hy_place▁holder▁no▁102｜>".to_string();
        let placeholder_token = "<｜hy_place▁holder▁no▁799｜>".to_string();
        // let pad_id = 120002u32;
        Ok(Self {
            image_token_id,
            image_token,
            placeholder_token,
            process_cfg,
            device: device.clone(),
            dtype,
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
            self.process_cfg.min_pixels as u32,
            self.process_cfg.max_pixels as u32,
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
            channel,
            grid_h / self.process_cfg.merge_size,
            self.process_cfg.merge_size,
            self.process_cfg.patch_size,
            grid_w / self.process_cfg.merge_size,
            self.process_cfg.merge_size,
            self.process_cfg.patch_size,
        ]);
        let img_tensor = img_tensor.reshape(shape)?;
        // shape to // grid_t,
        // grid_h / merge_size,
        // merge_size,
        // grid_w / merge_size,
        // merge_size,
        // channel,
        // patch_size,
        // patch_size,
        let img_tensor = img_tensor.permute(vec![0, 2, 3, 5, 6, 1, 4, 7])?;
        let img_tensor = img_tensor
            .reshape((
                grid_t * grid_h * grid_w,
                channel * self.process_cfg.patch_size * self.process_cfg.patch_size,
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
        tokenizer: &TokenizerModel,
        text: &str,
    ) -> Result<HunyuanData> {
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

        let mut image_tokens_cumsum = vec![0];
        let mut text = text.to_string();
        if !imgs.is_empty()
            && let Some(grid_thw) = image_grid_thw.as_ref()
        {
            let mut index = 0;
            while text.contains(&self.image_token) {
                let grid_i = grid_thw.i(index)?;
                let grid_h = grid_i.i(1)?.to_scalar::<u32>()?;
                let grid_w = grid_i.i(2)?.to_scalar::<u32>()?;
                let patch_h = grid_h / self.process_cfg.merge_size as u32;
                let patch_w = grid_w / self.process_cfg.merge_size as u32;
                let num_image_tokens = patch_h * (patch_w + 1) + 2;
                let num_id = image_tokens_cumsum[image_tokens_cumsum.len() - 1] + num_image_tokens;
                image_tokens_cumsum.push(num_id);
                let replace = self.placeholder_token.repeat(num_image_tokens as usize);
                text = text.replacen(&self.image_token, &replace, 1);
                index += 1;
            }
        }

        text = text.replace(&self.placeholder_token, &self.image_token);
        let input_ids = tokenizer.text_encode(text, &self.device)?;
        let seq_len = input_ids.dim(1)?;
        let position_ids = Tensor::arange(0, seq_len as u32, &self.device)?;
        let mut position_ids_w = Tensor::arange(0, seq_len as u32, &self.device)?;
        let mut position_ids_h = Tensor::arange(0, seq_len as u32, &self.device)?;
        let mut position_ids_t = Tensor::arange(0, seq_len as u32, &self.device)?;
        if !imgs.is_empty()
            && let Some(grid_thw) = image_grid_thw.as_ref()
        {
            let image_token_pos_indices = get_eq_indices(&input_ids.i(0)?, self.image_token_id)?;
            for i in 0..grid_thw.dim(0)? {
                let grid_i = grid_thw.i(i)?;
                let grid_h = grid_i.i(1)?.to_scalar::<u32>()?;
                let grid_w = grid_i.i(2)?.to_scalar::<u32>()?;
                let patch_h = grid_h / self.process_cfg.merge_size as u32;
                let patch_w = grid_w / self.process_cfg.merge_size as u32;
                let start_pos = image_token_pos_indices
                    .i(image_tokens_cumsum[i] as usize)?
                    .to_scalar::<u32>()? as usize
                    + 1;
                let replace_num = ((patch_w + 1) * patch_h) as usize;
                let pos_w: Vec<u32> = (0..patch_h).flat_map(|_| 0u32..patch_w + 1).collect();
                position_ids_w = position_ids_w.slice_assign(
                    &[start_pos..start_pos + replace_num],
                    &Tensor::new(pos_w, &self.device)?,
                )?;
                let pos_h: Vec<u32> = (0..patch_h)
                    .flat_map(|h| vec![h; (patch_w + 1) as usize])
                    .collect();
                position_ids_h = position_ids_h.slice_assign(
                    &[start_pos..start_pos + replace_num],
                    &Tensor::new(pos_h, &self.device)?,
                )?;
                position_ids_t = position_ids_t.slice_assign(
                    &[start_pos..start_pos + replace_num],
                    &Tensor::new(vec![0u32; replace_num], &self.device)?,
                )?;
            }
        }
        let position_ids = Tensor::stack(
            &[position_ids, position_ids_h, position_ids_w, position_ids_t],
            0,
        )?
        .unsqueeze(0)?;
        let image_mask = get_equal_mask(&input_ids, self.image_token_id)?;
        let data = HunyuanData {
            input_ids,
            position_ids,
            image_mask,
            pixel_values,
            image_grid_thw,
        };
        Ok(data)
    }
}

pub struct HunyuanData {
    pub input_ids: Tensor,
    pub position_ids: Tensor,
    pub image_mask: Tensor,
    pub pixel_values: Option<Tensor>,
    pub image_grid_thw: Option<Tensor>,
}

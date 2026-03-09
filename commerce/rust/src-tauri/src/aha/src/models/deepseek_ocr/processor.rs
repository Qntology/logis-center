use aha_openai_dive::v1::resources::chat::ChatCompletionParameters;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::utils::img_utils::dynamic_preprocess;
use crate::{
    tokenizer::TokenizerModel,
    utils::{
        extract_mes,
        img_utils::{extract_images, img_transform, resize_with_edge_padding},
    },
};

pub struct DeepseekOCRProcessor {
    device: Device,
    dtype: DType,
    image_token: String,
    image_token_id: u32,
    patch_size: u32,
    downsample_ratio: u32,
}

impl DeepseekOCRProcessor {
    pub fn new(device: &Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            device: device.clone(),
            dtype,
            image_token: "<image>".to_string(),
            image_token_id: 128815,
            patch_size: 16,
            downsample_ratio: 4,
        })
    }

    fn get_prompt(&self, mes_vec: Vec<(String, String)>) -> Result<String> {
        let sep = "\n";
        let sep2 = "";
        let mut ret = "".to_string();
        for (i, (_, message)) in mes_vec.iter().enumerate() {
            if message.chars().count() > 0 {
                if i % 2 == 0 {
                    ret = ret + message + sep;
                } else {
                    ret = ret + message + sep2;
                }
            }
        }
        ret = ret.trim().to_string();
        Ok(ret)
    }

    pub fn process_info(
        &self,
        mes: &ChatCompletionParameters,
        tokenizer: &TokenizerModel,
        base_size: u32,
        image_size: u32,
        crop_mode: bool,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let imgs = extract_images(mes)?;
        let mes_vec = extract_mes(mes)?;
        let prompt = self.get_prompt(mes_vec.clone())?;
        let text_splits: Vec<&str> = prompt.split(&self.image_token).collect();
        let img_mean =
            Tensor::from_slice(&[0.5, 0.5, 0.5], (3, 1, 1), &self.device)?.to_dtype(self.dtype)?;
        let img_std =
            Tensor::from_slice(&[0.5, 0.5, 0.5], (3, 1, 1), &self.device)?.to_dtype(self.dtype)?;
        let mut images_list = Vec::new();
        let mut images_crop_list = Vec::new();
        let mut images_seq_mask = vec![0u32];
        let mut tokenized_id = vec![0u32];
        let mut images_spatial_crop = Vec::new();
        for (text_seq, image) in text_splits.iter().zip(imgs) {
            if !text_seq.is_empty() {
                let token_ids = tokenizer.text_encode_vec(text_seq.to_string(), false)?;
                tokenized_id.extend_from_slice(&token_ids);
                let seq_mask = vec![0u32; token_ids.len()];
                images_seq_mask.extend_from_slice(&seq_mask);
            }
            if crop_mode {
                let mut images_crop_raw = Vec::new();
                let crop_ratio = if image.height() <= 640 && image.width() <= 640 {
                    (1u32, 1u32)
                } else {
                    let (img_crop, ratio) = dynamic_preprocess(&image, image_size, false)?;
                    images_crop_raw = img_crop.clone();
                    ratio
                };

                let gloabal_view =
                    resize_with_edge_padding(&image, base_size, base_size, [127u8; 3]);

                let global_img_trans =
                    img_transform(&gloabal_view, &img_mean, &img_std, &self.device, self.dtype)?;
                images_list.push(global_img_trans);

                images_spatial_crop.push(vec![crop_ratio.0, crop_ratio.1]);

                if crop_ratio.0 > 1 || crop_ratio.1 > 1 {
                    for img in images_crop_raw {
                        let img_t =
                            img_transform(&img, &img_mean, &img_std, &self.device, self.dtype)?;
                        images_crop_list.push(img_t);
                    }
                }

                let num_queries = image_size / self.patch_size / self.downsample_ratio;
                let num_queries_base = base_size / self.patch_size / self.downsample_ratio;
                let mut token_repeat = num_queries_base.pow(2) + num_queries_base + 1;
                if crop_ratio.0 > 1 || crop_ratio.1 > 1 {
                    token_repeat += (num_queries * crop_ratio.0 + 1) * (num_queries * crop_ratio.1);
                }
                let tokenized_image = vec![self.image_token_id; token_repeat as usize];
                tokenized_id.extend_from_slice(&tokenized_image);
                let seq_mask = vec![1u32; tokenized_image.len()];
                images_seq_mask.extend_from_slice(&seq_mask);
            } else {
                let global_view = if image_size <= 640 {
                    image.resize_exact(
                        image_size,
                        image_size,
                        image::imageops::FilterType::CatmullRom,
                    )
                } else {
                    resize_with_edge_padding(&image, image_size, image_size, [127u8; 3])
                };
                let global_img_trans =
                    img_transform(&global_view, &img_mean, &img_std, &self.device, self.dtype)?;
                images_list.push(global_img_trans);

                images_spatial_crop.push(vec![1, 1]);
                let num_queries = image_size / self.patch_size / self.downsample_ratio;
                let token_repeat = num_queries.pow(2) + num_queries + 1;
                let tokenized_image = vec![self.image_token_id; token_repeat as usize];
                tokenized_id.extend_from_slice(&tokenized_image);
                let seq_mask = vec![1u32; tokenized_image.len()];
                images_seq_mask.extend_from_slice(&seq_mask);
            }
        }
        let token_ids =
            tokenizer.text_encode_vec(text_splits[text_splits.len() - 1].to_string(), false)?;
        tokenized_id.extend_from_slice(&token_ids);
        let seq_mask = vec![0u32; token_ids.len()];
        images_seq_mask.extend_from_slice(&seq_mask);
        let input_ids = Tensor::new(tokenized_id, &self.device)?.unsqueeze(0)?;
        let image_seq_mask = Tensor::new(images_seq_mask, &self.device)?.unsqueeze(0)?;
        let (images_ori, images_spatial_crop_t, image_crop) = if images_list.is_empty() {
            let images_ori = Tensor::zeros(
                (1usize, 3usize, image_size as usize, image_size as usize),
                self.dtype,
                &self.device,
            )?;
            let images_spatial_crop_t = Tensor::zeros((1, 2), DType::F64, &self.device)?;
            let image_crop = Tensor::zeros(
                (1usize, 3usize, base_size as usize, base_size as usize),
                self.dtype,
                &self.device,
            )?;
            (images_ori, images_spatial_crop_t, image_crop)
        } else {
            let images_ori = Tensor::stack(&images_list, 0)?;
            let images_spatial_crop_t = Tensor::new(images_spatial_crop, &self.device)?;
            let image_crop = if !images_crop_list.is_empty() {
                Tensor::stack(&images_crop_list, 0)?
            } else {
                Tensor::zeros(
                    (1usize, 3usize, base_size as usize, base_size as usize),
                    self.dtype,
                    &self.device,
                )?
            };
            (images_ori, images_spatial_crop_t, image_crop)
        };

        Ok((
            input_ids,
            images_ori,
            image_crop,
            image_seq_mask,
            images_spatial_crop_t,
        ))
    }
}

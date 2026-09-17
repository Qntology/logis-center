use candle_core::{Device, Result, Tensor};
use image::DynamicImage;

use super::Siglip2Config;

pub struct PreprocessedImage {
    pub pixel_values: Tensor,
    pub grid_cols: usize,
    pub grid_rows: usize,
    pub scale_x: f64,
    pub scale_y: f64,
    pub orig_width: u32,
    pub orig_height: u32,
}

pub fn preprocess_image(
    image: &DynamicImage,
    config: &Siglip2Config,
    device: &Device,
) -> Result<PreprocessedImage> {
    let orig_width = image.width();
    let orig_height = image.height();

    let patch_size = config.patch_size; // 16
    let max_patches = config.max_num_patches; // 256

    let scaled_dim = |scale: f64, size: u32| -> usize {
        let scaled = scale * size as f64;
        let divisible = (scaled / patch_size as f64).ceil() as usize * patch_size;
        divisible.max(patch_size)
    };

    let eps = 1e-5f64;
    let mut scale_min = eps;
    let mut scale_max = 100.0f64;
    while (scale_max - scale_min) >= eps {
        let scale = (scale_min + scale_max) / 2.0;
        let th = scaled_dim(scale, orig_height);
        let tw = scaled_dim(scale, orig_width);
        let n = (th / patch_size) * (tw / patch_size);
        if n <= max_patches {
            scale_min = scale;
        } else {
            scale_max = scale;
        }
    }

    let target_h = scaled_dim(scale_min, orig_height);
    let target_w = scaled_dim(scale_min, orig_width);

    let grid_rows = (target_h / patch_size).max(1);
    let grid_cols = (target_w / patch_size).max(1);

    let resized_w = grid_cols * patch_size;
    let resized_h = grid_rows * patch_size;

    println!(
        "[SigLIP2/NaFlex] {}x{} → {}x{} (grid {}x{} = {} patches / max {})",
        orig_width, orig_height, resized_w, resized_h,
        grid_cols, grid_rows, grid_cols * grid_rows, max_patches
    );

    let scale_x = orig_width as f64 / resized_w as f64;
    let scale_y = orig_height as f64 / resized_h as f64;
    let resized = image.resize_exact(
        resized_w as u32,
        resized_h as u32,
        image::imageops::FilterType::Triangle, // bilinear
    );

    let rgb = resized.to_rgb8();
    let plane = resized_h * resized_w;
    let mut pixel_data = vec![0.0f32; 3 * plane];

    for y in 0..resized_h {
        for x in 0..resized_w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            let idx = y * resized_w + x;
            pixel_data[idx] = (pixel[0] as f32 / 255.0) * 2.0 - 1.0;
            pixel_data[plane + idx] = (pixel[1] as f32 / 255.0) * 2.0 - 1.0;
            pixel_data[2 * plane + idx] = (pixel[2] as f32 / 255.0) * 2.0 - 1.0;
        }
    }

    let pixel_values = Tensor::from_vec(
        pixel_data,
        (1, 3, resized_h, resized_w),
        device,
    )?;

    Ok(PreprocessedImage {
        pixel_values,
        grid_cols,
        grid_rows,
        scale_x,
        scale_y,
        orig_width,
        orig_height,
    })
}

pub fn patch_index_to_bbox(
    patch_idx: usize,
    grid_cols: usize,
    patch_size: usize,
    scale_x: f64,
    scale_y: f64,
) -> (u32, u32, u32, u32) {
    let row = patch_idx / grid_cols.max(1);
    let col = patch_idx % grid_cols.max(1);

    let x = (col as f64 * patch_size as f64 * scale_x).floor().max(0.0) as u32;
    let y = (row as f64 * patch_size as f64 * scale_y).floor().max(0.0) as u32;
    let x2 = ((col + 1) as f64 * patch_size as f64 * scale_x).ceil().max(0.0) as u32;
    let y2 = ((row + 1) as f64 * patch_size as f64 * scale_y).ceil().max(0.0) as u32;

    (x, y, x2.max(x + 1), y2.max(y + 1))
}

pub fn patches_to_bounding_box(
    patch_indices: &[usize],
    grid_cols: usize,
    patch_size: usize,
    scale_x: f64,
    scale_y: f64,
) -> Option<(u32, u32, u32, u32)> {
    if patch_indices.is_empty() || grid_cols == 0 {
        return None;
    }

    let mut x_min = u32::MAX;
    let mut y_min = u32::MAX;
    let mut x_max = 0u32;
    let mut y_max = 0u32;

    for &idx in patch_indices {
        let (px, py, px2, py2) =
            patch_index_to_bbox(idx, grid_cols, patch_size, scale_x, scale_y);
        x_min = x_min.min(px);
        y_min = y_min.min(py);
        x_max = x_max.max(px2);
        y_max = y_max.max(py2);
    }

    if x_min == u32::MAX || y_min == u32::MAX {
        return None;
    }

    Some((x_min, y_min, x_max.max(x_min + 1), y_max.max(y_min + 1)))
}
use anyhow::{Result, anyhow};
use num::integer::lcm;

use crate::utils::{ceil_by_factor, floor_by_factor, round_by_factor};

// use std::{fs::File, io::Write};
// use ffmpeg_next as ffmpeg;

pub fn video_smart_resize(
    num_frames: u32,
    height: u32,
    width: u32,
    temporal_factor: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
    video_ratio: Option<u32>,
) -> Result<(u32, u32)> {
    if num_frames < temporal_factor {
        return Err(anyhow!(format!(
            "{num_frames} must be larger than temporal_factor {temporal_factor}"
        )));
    }
    if height < factor || width < factor {
        return Err(anyhow!(format!(
            "height:{height} or width:{width} must be larger than factor:{factor}"
        )));
    }
    if std::cmp::max(height, width) / std::cmp::min(height, width) > 200 {
        return Err(anyhow!(format!(
            "absolute aspect ratio mush be smaller than {}, got {}",
            200,
            std::cmp::max(height, width) / std::cmp::min(height, width)
        )));
    }
    let mut image_factor = factor;
    if let Some(ratio) = video_ratio {
        image_factor = lcm(image_factor, ratio);
    }
    let mut h_bar = round_by_factor(height, image_factor);
    let mut w_bar = round_by_factor(width, image_factor);
    let t_bar = round_by_factor(num_frames, temporal_factor);
    if t_bar * h_bar * w_bar > max_pixels {
        let beta = ((num_frames * height * width) as f32 / max_pixels as f32).sqrt();
        h_bar = std::cmp::max(
            image_factor,
            floor_by_factor(height as f32 / beta, image_factor),
        );
        w_bar = std::cmp::max(
            image_factor,
            floor_by_factor(width as f32 / beta, image_factor),
        );
    } else if t_bar * h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f32 / (num_frames * height * width) as f32).sqrt();
        h_bar = ceil_by_factor(height as f32 * beta, image_factor);
        w_bar = ceil_by_factor(width as f32 * beta, image_factor);
    }
    Ok((h_bar, w_bar))
}

// #[allow(unused)]
// fn save_file(
//     frame: &ffmpeg::frame::Video,
//     index: usize,
// ) -> std::result::Result<(), std::io::Error> {
//     let mut file = File::create(format!("frame{}.ppm", index))?;
//     file.write_all(format!("P6\n{} {}\n255\n", frame.width(), frame.height()).as_bytes())?;
//     file.write_all(frame.data(0))?;
//     Ok(())
// }

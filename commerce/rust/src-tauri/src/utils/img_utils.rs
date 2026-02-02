use anyhow::Result;
use image::DynamicImage;

pub fn get_image(path: &str) -> Result<DynamicImage> {
    Ok(image::open(path)?)
}

pub fn img_smart_resize(
    h: u32,
    w: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<(u32, u32)> {
    let mut fh = h as f32;
    let mut fw = w as f32;
    
    let pixels = fh * fw;
    if pixels > max_pixels as f32 {
        let scale = (max_pixels as f32 / pixels).sqrt();
        fh *= scale;
        fw *= scale;
    }
    
    if fh * fw < min_pixels as f32 {
        let scale = (min_pixels as f32 / (fh * fw)).sqrt();
        fh *= scale;
        fw *= scale;
    }

    let rh = ((fh / factor as f32).round() as u32) * factor;
    let rw = ((fw / factor as f32).round() as u32) * factor;
    
    Ok((rh, rw))
}

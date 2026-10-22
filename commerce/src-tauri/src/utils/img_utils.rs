use std::io::Cursor;
use std::{collections::HashSet, path::PathBuf};

use anyhow::{Result, anyhow};
use base64::{Engine, engine::general_purpose};
use candle_core::{DType, Device, Tensor};
use image::{DynamicImage, ImageBuffer, ImageReader, Rgb, RgbImage, imageops};

use crate::utils::{ceil_by_factor, floor_by_factor, round_by_factor};

pub fn load_image_from_base64(base64_data: &str) -> Result<DynamicImage> {
    let image_data = general_purpose::STANDARD
        .decode(base64_data)
        .map_err(|e| anyhow!(format!("Failed to decode image: {}", e)))?;
    let cursor = Cursor::new(image_data);
    let img = ImageReader::new(cursor)
        .with_guessed_format()
        .map_err(|e| anyhow!(format!("Failed to read image format: {}", e)))?
        .decode()
        .map_err(|e| anyhow!(format!("Failed to decode image: {}", e)))?;
    Ok(img)
}

// 🌟 [IN-PROCESS IMAGE] 같은 프로세스 안에서 크롭을 생성기로 넘길 때 PNG 왕복을 생략합니다.
//
//  ── 실측 ──
//   비전 크롭 호출은 크롭을 PNG 로 인코딩 → base64 → data URL 로 감싸 넘기고, 프로세서가 다시
//   base64 → PNG 를 풀었습니다. dev 프로필은 png · miniz_oxide · base64 를 최적화하지 않으므로
//   2048x1108 크롭 한 장에 인코딩 545ms + 디코딩 93ms (png_roundtrip_tests 실측)가 LLM 호출마다
//   (문서 1장 약 20회) 들었습니다. PNG 는 무손실이라 화소를 바꾸지도 않는 왕복입니다.
//
//  ── 해결 ──
//   크롭을 프로세스 안 보관소에 맡기고 'mem://<번호>' 주소만 넘깁니다. get_image 는 그 주소면 보관소의
//   이미지를 그대로 꺼냅니다. 같은 DynamicImage(같은 화소 · 같은 색 형식)라 pixel_values 와
//   VISION-CACHE 키가 예전과 같습니다. 보관 기간은 호출 하나이며, 가드가 사라지면 함께 지웁니다.
static MEM_IMAGES: once_cell::sync::Lazy<std::sync::Mutex<std::collections::HashMap<u64, DynamicImage>>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));
static MEM_IMAGE_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
const MEM_IMAGE_SCHEME: &str = "mem://";

/// 보관소에서 이미지를 지우는 가드. 생성 호출이 끝날 때까지 들고 있어야 합니다.
pub struct MemImageGuard {
    id: u64,
}

impl Drop for MemImageGuard {
    fn drop(&mut self) {
        MEM_IMAGES.lock().unwrap_or_else(|e| e.into_inner()).remove(&self.id);
    }
}

/// 이미지를 보관소에 맡기고 (get_image 가 읽는 주소, 가드) 를 돌려줍니다.
pub fn register_mem_image(img: DynamicImage) -> (String, MemImageGuard) {
    let id = MEM_IMAGE_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    MEM_IMAGES.lock().unwrap_or_else(|e| e.into_inner()).insert(id, img);
    (format!("{}{}", MEM_IMAGE_SCHEME, id), MemImageGuard { id })
}

pub fn get_image(file: &str) -> Result<DynamicImage> {
    if let Some(id) = file.strip_prefix(MEM_IMAGE_SCHEME) {
        let id: u64 = id.parse().map_err(|_| anyhow!(format!("bad in-process image id: {}", id)))?;
        return MEM_IMAGES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&id)
            .cloned()
            .ok_or_else(|| anyhow!(format!("in-process image {} is no longer registered", id)));
    }
    let mut img = None;
    if file.starts_with("http://") || file.starts_with("https://") {
        return Err(anyhow!("HTTP URLs not supported in local mode".to_string()));
    }
    if file.starts_with("file://") {
        let path = url::Url::parse(file)?;
        let path = path.to_file_path();
        let path = match path {
            Ok(path) => path,
            Err(_) => {
                let mut path = file.to_owned();
                path = path.split_off(7);
                PathBuf::from(path)
            }
        };
        img = Some(
            ImageReader::open(path)
                .map_err(|e| anyhow!(format!("Failed to open file: {}", e)))?
                .decode()
                .map_err(|e| anyhow!(format!("Failed to decode image: {}", e)))?,
        );
    }
    if file.starts_with("data:image") && file.contains("base64,") {
        let data: Vec<&str> = file.split("base64,").collect();
        let data = data[1];
        img = Some(load_image_from_base64(data)?);
    }
    if let Some(img) = img {
        return Ok(img);
    }
    Err(anyhow!("get image from message failed".to_string()))
}

pub fn generate_target_ratios_sorted(min_num: u32, max_num: u32) -> Vec<(u32, u32)> {
    let mut target_ratios = HashSet::new();

    for n in min_num..=max_num {
        for i in 1..=n {
            for j in 1..=n {
                let product = i * j;
                if product <= max_num && product >= min_num {
                    target_ratios.insert((i, j));
                }
            }
        }
    }
    // Convert to vector and sort by the product of elements (i*j)
    let mut sorted_ratios: Vec<(u32, u32)> = target_ratios.into_iter().collect();
    sorted_ratios.sort_by_key(|&(i, j)| i * j);

    sorted_ratios
}

pub fn find_closest_aspect_ratio(
    aspect_ratio: f64,
    target_ratios: &[(u32, u32)],
    width: u32,
    height: u32,
    image_size: u32,
) -> (u32, u32) {
    let mut best_ratio_diff = f64::INFINITY;
    let mut best_ratio = (1, 1);
    let area = width * height;

    for &ratio in target_ratios {
        let target_aspect_ratio = ratio.0 as f64 / ratio.1 as f64;
        let ratio_diff = (aspect_ratio - target_aspect_ratio).abs();

        if ratio_diff < best_ratio_diff {
            best_ratio_diff = ratio_diff;
            best_ratio = ratio;
        } else if (ratio_diff - best_ratio_diff).abs() < 1e-10 {
            // 当多个候选比例具有相同的宽高比差异时，根据图像的实际面积来选择最优比例。
            let target_area = 0.5 * (image_size as f64).powi(2) * (ratio.0 * ratio.1) as f64;
            if area as f64 > target_area {
                best_ratio = ratio;
            }
        }
    }

    best_ratio
}

pub fn dynamic_preprocess(
    image: &DynamicImage,
    image_size: u32,
    use_thumbnail: bool,
) -> Result<(Vec<DynamicImage>, (u32, u32))> {
    let orig_width = image.width();
    let orig_height = image.height();
    let aspect_ratio = orig_width as f64 / orig_height as f64;
    // 控制分块数量在2-9之间
    let target_ratios = generate_target_ratios_sorted(2, 9);
    let target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio,
        &target_ratios,
        orig_width,
        orig_height,
        image_size,
    );
    let target_width = image_size * target_aspect_ratio.0;
    let target_height = image_size * target_aspect_ratio.1;
    let blocks = target_aspect_ratio.0 * target_aspect_ratio.1;
    let mut resized_img = image.resize_exact(
        target_width,
        target_height,
        image::imageops::FilterType::CatmullRom,
    );
    let mut processed_images = Vec::new();
    let grid_width = target_width / image_size;
    for i in 0..blocks {
        // Calculate box coordinates
        let x1 = (i % grid_width) * image_size;
        let y1 = (i / grid_width) * image_size;

        // Crop the image
        let split_img = resized_img.crop(x1, y1, image_size, image_size);
        processed_images.push(split_img);
    }
    assert_eq!(processed_images.len() as u32, blocks);

    if use_thumbnail && processed_images.len() != 1 {
        let thumbnail_img = image.resize_exact(
            image_size,
            image_size,
            image::imageops::FilterType::CatmullRom,
        );
        processed_images.push(thumbnail_img);
    }
    Ok((processed_images, target_aspect_ratio))
}

pub fn resize_with_edge_padding(
    img: &DynamicImage,
    width: u32,
    height: u32,
    color: [u8; 3],
) -> DynamicImage {
    // 按图像原比例resize,可能不是输入的宽高
    let mut img = img.resize(width, height, image::imageops::FilterType::CatmullRom);
    // 使用输入像素颜色填充为输入宽高
    if img.height() != height || img.width() != width {
        let (img_h, img_w) = (img.height(), img.width());
        let img_buffer = img.to_rgb8();
        let mut canvas: ImageBuffer<Rgb<u8>, Vec<u8>> =
            RgbImage::from_pixel(width, height, Rgb(color));
        let x_offset = (width - img_w) / 2;
        let y_offset = (height - img_h) / 2;
        imageops::overlay(&mut canvas, &img_buffer, x_offset as i64, y_offset as i64);
        img = DynamicImage::ImageRgb8(canvas);
    }
    img
}

pub fn img_transform(
    img: &DynamicImage,
    mean: &Tensor,
    std: &Tensor,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let img_h = img.height();
    let img_w = img.width();
    let img_vec = img.to_rgb8().into_raw();
    // (h, w, c) => (c, h, w)
    let img_tensor = Tensor::from_slice(&img_vec, (img_h as usize, img_w as usize, 3), device)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?;
    // 0-255 rescale to 0-1
    let img_tensor = img_tensor.affine(1.0 / 255.0, 0.)?;
    // normalize
    let img_tensor = img_tensor
        .broadcast_sub(&mean.to_dtype(DType::F32)?)?
        .broadcast_div(&std.to_dtype(DType::F32)?)?
        .to_dtype(dtype)?;
    Ok(img_tensor)
}

pub fn img_smart_resize(
    img_h: u32,
    img_w: u32,
    factor: u32,
    min_pixels: u32,
    max_pixels: u32,
) -> Result<(u32, u32)> {
    if std::cmp::max(img_h, img_w) / std::cmp::min(img_h, img_w) > 200 {
        return Err(anyhow!(format!(
            "absolute aspect ratio mush be smaller than {}, got {}",
            200,
            std::cmp::max(img_h, img_w) / std::cmp::min(img_h, img_w)
        )));
    }
    let image_factor = factor;
    let mut h_bar = std::cmp::max(image_factor, round_by_factor(img_h, image_factor));
    let mut w_bar = std::cmp::max(image_factor, round_by_factor(img_w, image_factor));

    if h_bar * w_bar > max_pixels {
        let beta = ((img_h * img_w) as f32 / max_pixels as f32).sqrt();
        h_bar = std::cmp::max(
            image_factor,
            floor_by_factor(img_h as f32 / beta, image_factor),
        );
        w_bar = std::cmp::max(
            image_factor,
            floor_by_factor(img_w as f32 / beta, image_factor),
        );
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f32 / (img_h * img_w) as f32).sqrt();
        h_bar = ceil_by_factor(img_h as f32 * beta, image_factor);
        w_bar = ceil_by_factor(img_w as f32 * beta, image_factor);
    }
    Ok((h_bar, w_bar))
}

pub fn img_transform_with_resize(
    img: &DynamicImage,
    h: u32,
    w: u32,
    mean: &Tensor,
    std: &Tensor,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let img_resize = img.resize_exact(w, h, imageops::FilterType::CatmullRom);
    let img_tensor = img_transform(&img_resize, mean, std, device, dtype)?;
    Ok(img_tensor)
}

pub fn float_tensor_to_dynamic_image(tensor: &Tensor) -> Result<DynamicImage> {
    let tensor = tensor.affine(255.0, 0.0)?.clamp(0.0, 255.0)?;
    let tensor_u8 = tensor.to_dtype(DType::U8)?.to_device(&Device::Cpu)?;
    let (c, h, w) = tensor_u8.dims3()?;
    match c {
        1 => {
            let tensor_u8 = tensor_u8.reshape((h, w))?;
            let data: Vec<u8> = tensor_u8.flatten_all()?.to_vec1()?;
            let img = ImageBuffer::from_raw(w as u32, h as u32, data)
                .ok_or_else(|| anyhow!("Failed to create image buffer"))?;
            Ok(DynamicImage::ImageLuma8(img))
        }
        3 => {
            let tensor_u8 = tensor_u8.permute((1, 2, 0))?;
            let data: Vec<u8> = tensor_u8.flatten_all()?.to_vec1()?;
            let img = ImageBuffer::from_raw(w as u32, h as u32, data)
                .ok_or_else(|| anyhow!("Failed to create image buffer"))?;
            Ok(DynamicImage::ImageRgb8(img))
        }
        _ => Err(anyhow!(format!("Unsupported number of channels: {}", c))),
    }
}
#[cfg(test)]
mod png_roundtrip_tests {
    //! 🌟 [CROP PNG ROUNDTRIP] 크롭 전송(PNG 인코딩 → base64 → 디코딩)의 비용과 무손실성을 잽니다.
    //! dev 프로필은 png · miniz_oxide · image 를 최적화하지 않으므로 이 경로가 크롭 호출마다 반복됩니다.
    //! 실행: cargo test --lib -- png_roundtrip --ignored --nocapture
    use super::*;
    use base64::engine::general_purpose::STANDARD;

    /// 흰 바탕에 글자 획 같은 검은 띠가 있는 문서 크롭 모양 이미지 (2048x1108, 실측 header 크롭 크기).
    fn doc_like(w: u32, h: u32) -> DynamicImage {
        let mut img = RgbImage::from_pixel(w, h, Rgb([255, 255, 255]));
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        for row in (40..h.saturating_sub(40)).step_by(56) {
            let mut col = 40u32;
            while col + 30 < w {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let glyph_w = 10 + (x % 18) as u32;
                for yy in row..(row + 22).min(h) {
                    for xx in col..(col + glyph_w).min(w) {
                        if (xx + yy + (x as u32 % 5)) % 3 != 0 {
                            img.put_pixel(xx, yy, Rgb([20, 20, 30]));
                        }
                    }
                }
                col += glyph_w + 6 + (x % 11) as u32;
            }
        }
        DynamicImage::ImageRgb8(img)
    }

    fn roundtrip(img: &DynamicImage, fast: bool) -> (f64, f64, usize, DynamicImage) {
        let t0 = std::time::Instant::now();
        let mut buf = Cursor::new(Vec::new());
        if fast {
            let enc = image::codecs::png::PngEncoder::new_with_quality(
                &mut buf,
                image::codecs::png::CompressionType::Fast,
                image::codecs::png::FilterType::Adaptive,
            );
            img.write_with_encoder(enc).unwrap();
        } else {
            img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        }
        let b64 = STANDARD.encode(buf.into_inner());
        let enc_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = std::time::Instant::now();
        let back = load_image_from_base64(&b64).unwrap();
        let dec_ms = t1.elapsed().as_secs_f64() * 1000.0;
        (enc_ms, dec_ms, b64.len(), back)
    }

    #[test]
    fn in_process_image_handoff_returns_the_same_pixels_until_the_guard_drops() {
        let img = doc_like(320, 180);
        let (url, guard) = register_mem_image(img.clone());
        assert!(url.starts_with("mem://"));
        let t0 = std::time::Instant::now();
        let back = get_image(&url).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(back.color(), img.color());
        assert_eq!(back.to_rgb8().as_raw(), img.to_rgb8().as_raw());
        println!("[IN-PROCESS IMAGE] 보관소 조회 {:.2}ms (PNG 왕복 대신)", ms);
        drop(guard);
        assert!(get_image(&url).is_err(), "가드가 사라지면 보관소에서도 지워집니다");
        assert!(get_image("mem://not-a-number").is_err());
    }

    #[test]
    #[ignore = "timing: dev 프로필의 크롭 PNG 왕복 비용 측정"]
    fn png_roundtrip_cost_and_losslessness() {
        let img = doc_like(2048, 1108);
        for fast in [false, true] {
            let (enc, dec, bytes, back) = roundtrip(&img, fast);
            assert_eq!(back.to_rgb8().as_raw(), img.to_rgb8().as_raw(), "PNG 는 무손실이어야 합니다");
            println!(
                "[CROP PNG ROUNDTRIP] {} | 인코딩+base64 {:.0}ms | base64+디코딩 {:.0}ms | {:.1} MB",
                if fast { "Fast 압축" } else { "기본 압축(현행)" },
                enc,
                dec,
                bytes as f64 / 1e6
            );
        }
    }
}

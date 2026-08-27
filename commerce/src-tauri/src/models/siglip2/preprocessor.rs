use candle_core::{Device, Result, Tensor};
use image::DynamicImage;

use super::Siglip2Config;

/// NaFlex 전처리 결과
pub struct PreprocessedImage {
    /// 정규화된 이미지 텐서: (1, 3, H, W)
    pub pixel_values: Tensor,
    /// 패치 그리드 열 수 (예: 18)
    pub grid_cols: usize,
    /// 패치 그리드 행 수 (예: 14)
    pub grid_rows: usize,
    /// 원본 → 리사이즈 스케일 (좌표 역변환용)
    pub scale_x: f64,
    /// 원본 → 리사이즈 스케일
    pub scale_y: f64,
    /// 원본 이미지 너비
    pub orig_width: u32,
    /// 원본 이미지 높이
    pub orig_height: u32,
}

/// NaFlex 전처리: 원본 이미지 → 정규화된 패치 텐서
///
/// 처리 순서 (preprocessor_config.json 기반):
///   ① do_rescale: 픽셀값 / 255.0 → [0, 1]
///   ② do_normalize: (x - 0.5) / 0.5 → [-1, 1]
///   ③ do_resize: 종횡비 보존 + 총 패치 ≤ 256 (NaFlex)
///   ④ 16×16 패치 그리드 분할
///
/// 예: 1024×768 (4:3) 이미지
///   → 종횡비 4:3 유지하면서 총 패치 ≤ 256
///   → 18×14 = 252 패치
///   → 리사이즈: 288×224 픽셀
///   → 스케일: scale_x = 1024/288 = 3.556, scale_y = 768/224 = 3.429
pub fn preprocess_image(
    image: &DynamicImage,
    config: &Siglip2Config,
    device: &Device,
) -> Result<PreprocessedImage> {
    let orig_width = image.width();
    let orig_height = image.height();

    // ── ③ NaFlex 리사이즈: 종횡비 보존 + 총 패치 ≤ max_num_patches ──
    let patch_size = config.patch_size; // 16
    let max_patches = config.max_num_patches; // 256

    // 종횡비를 보존하면서 패치 그리드 크기 결정
    // 원본 종횡비 = orig_width / orig_height
    let aspect = orig_width as f64 / orig_height as f64;

    // grid_cols * grid_rows ≤ max_patches 이면서
    // grid_cols / grid_rows ≈ aspect 인 정수 그리드 찾기
    let mut best_cols = 1usize;
    let mut best_rows = 1usize;
    let mut best_diff = f64::MAX;

    for rows in 1..=max_patches {
        let cols_float = (max_patches as f64 / rows as f64).sqrt() * aspect.sqrt();
        let cols = cols_float.round() as usize;
        if cols == 0 || rows == 0 {
            continue;
        }
        if cols * rows > max_patches {
            continue;
        }
        let grid_aspect = cols as f64 / rows as f64;
        let diff = (grid_aspect - aspect).abs();
        if diff < best_diff && cols * rows > 0 {
            best_diff = diff;
            best_cols = cols;
            best_rows = rows;
        }
    }

    // 최소 1×1 보장
    let grid_cols = best_cols.max(1);
    let grid_rows = best_rows.max(1);

    // 리사이즈된 이미지 크기
    let resized_w = grid_cols * patch_size; // 예: 18 * 16 = 288
    let resized_h = grid_rows * patch_size; // 예: 14 * 16 = 224

    // 스케일 (원본 좌표 역변환용)
    let scale_x = orig_width as f64 / resized_w as f64;
    let scale_y = orig_height as f64 / resized_h as f64;

    // 이미지 리사이즈 (bilinear, resample=2)
    let resized = image.resize_exact(
        resized_w as u32,
        resized_h as u32,
        image::imageops::FilterType::Triangle, // bilinear
    );

    // ── ① rescale + ② normalize ──
    // 픽셀값 → [0,1] → (x - 0.5) / 0.5 → [-1, 1]
    let rgb = resized.to_rgb8();
    let mut pixel_data = vec![0.0f32; 3 * resized_h * resized_w];

    for y in 0..resized_h {
        for x in 0..resized_w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            let base = (y * resized_w + x) * 3;
            // rescale: /255, normalize: (x - 0.5) / 0.5 = x * 2.0 - 1.0
            pixel_data[base] = (pixel[0] as f32 / 255.0) * 2.0 - 1.0;
            pixel_data[base + 1] = (pixel[1] as f32 / 255.0) * 2.0 - 1.0;
            pixel_data[base + 2] = (pixel[2] as f32 / 255.0) * 2.0 - 1.0;
        }
    }

    // Tensor 생성: (1, 3, H, W)
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

/// 패치 인덱스 → 원본 이미지 바운딩 박스 좌표
///
/// 패치 인덱스 i:
///   row = i / grid_cols
///   col = i % grid_cols
///   원본 x = col * 16 * scale_x
///   원본 y = row * 16 * scale_y
///   원본 박스 = [x, y, x + 16*scale_x, y + 16*scale_y]
///
/// 이 함수가 STEP 3 (Column Cosine Matching)에서
/// 코사인 히트맵의 패치 위치를 원본 이미지 좌표로 변환합니다.
pub fn patch_index_to_bbox(
    patch_idx: usize,
    grid_cols: usize,
    patch_size: usize,
    scale_x: f64,
    scale_y: f64,
) -> (u32, u32, u32, u32) {
    let row = patch_idx / grid_cols;
    let col = patch_idx % grid_cols;

    let x = (col as f64 * patch_size as f64 * scale_x) as u32;
    let y = (row as f64 * patch_size as f64 * scale_y) as u32;
    let x2 = ((col + 1) as f64 * patch_size as f64 * scale_x) as u32;
    let y2 = ((row + 1) as f64 * patch_size as f64 * scale_y) as u32;

    (x, y, x2, y2)
}

/// 코사인 임계값 초과 패치들의 최소 포함 직사각형 (MBR) 계산
///
/// 이 함수가 STEP 4 (Vision NMS & Cropping)에서
/// 바운딩 박스를 생성합니다.
pub fn patches_to_bounding_box(
    patch_indices: &[usize],
    grid_cols: usize,
    patch_size: usize,
    scale_x: f64,
    scale_y: f64,
) -> Option<(u32, u32, u32, u32)> {
    if patch_indices.is_empty() {
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

    Some((x_min, y_min, x_max, y_max))
}
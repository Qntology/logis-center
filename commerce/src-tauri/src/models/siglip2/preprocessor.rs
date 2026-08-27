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
    //
    // 🌟 [HF PARITY] image_processing_siglip2.get_image_size_for_max_num_patches 이식.
    //    스케일 s 를 이진 탐색하여
    //      ceil(H*s / p) * ceil(W*s / p) <= max_num_patches
    //    를 만족하는 최대 s 를 찾습니다.
    //    기존 구현은 rows 를 순회하며 cols 를 근사식으로 계산해
    //    패치 예산을 크게 낭비했고(해상도 손실 = 크롭 좌표 정밀도 손실),
    //    종횡비도 반올림 오차만큼 흔들렸습니다.
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

    // 리사이즈된 이미지 크기 (항상 patch_size 의 배수)
    let resized_w = grid_cols * patch_size;
    let resized_h = grid_rows * patch_size;

    println!(
        "[SigLIP2/NaFlex] {}x{} → {}x{} (grid {}x{} = {} patches / max {})",
        orig_width, orig_height, resized_w, resized_h,
        grid_cols, grid_rows, grid_cols * grid_rows, max_patches
    );

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
    //  preprocessor_config: image_mean = image_std = [0.5,0.5,0.5]
    //  → (x/255 - 0.5) / 0.5  ==  x/255 * 2 - 1
    //
    // 🌟 [LAYOUT] 반드시 CHW 로 채워야 합니다.
    //    vision.rs 의 patchify 가 (b, C, H, W) 를 (b,C,nh,p,nw,p) 로 reshape 한 뒤
    //    permute(0,2,4,3,5,1) 로 (b,nh,nw,p,p,C) 를 만들기 때문입니다.
    //    기존 코드는 HWC 순서로 채워 놓고 (1,3,H,W) 로 reshape 하여
    //    채널이 이미지 세로 방향으로 흩어지는 심각한 레이아웃 손상이 있었습니다.
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
///   원본 x = col * patch_size * scale_x
///   원본 y = row * patch_size * scale_y
///
/// 🌟 [SCALE 정의] scale_x = orig_width / resized_width 입니다.
///    resized_width 는 grid_cols * patch_size 이므로,
///    col == grid_cols 인 경우 정확히 orig_width 가 됩니다.
///    부동소수 반올림으로 1~2px 초과할 수 있어 호출부에서 클램프해야 하며,
///    여기서는 좌표 순서만 보장합니다(x2 >= x, y2 >= y).
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

/// 코사인 임계값 초과 패치들의 최소 포함 직사각형 (MBR) 계산
///
/// 이 함수가 STEP 4 (Vision NMS & Cropping)에서
/// 바운딩 박스를 생성합니다.
/// 코사인 임계값 초과 패치들의 최소 포함 직사각형 (MBR) 계산
///
/// 🌟 [LEGACY] 단순 MBR 이 필요한 호출부를 위해 유지합니다.
///    실제 크롭 파이프라인은 vision_crop.rs 의 연결 성분 + 배타 배정을 씁니다.
///    (MBR 하나만 쓰면 문서 상단과 하단에 흩어진 두 반응이
///     페이지 전체를 덮는 거대한 박스로 합쳐집니다)
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
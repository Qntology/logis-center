// =====================================================================
// 🌟 [SigLIP2 VISION NMS & CROPPING]
// ---------------------------------------------------------------------
//  이 파일은 기획서의 STEP 3 을 담당합니다.
//
//   입력  : CategoryHeatmap[]  (Part 5 의 build_column_heatmaps 산출물)
//   출력  : CropPlan[]         (카테고리 → 원본 이미지 좌표 + 크롭된 이미지)
//
//  ── parsing.rs::get_trade_doc_slice_config 를 대체하는 지점 ──
//   기존:  ("header", 0.00, 0.25) 처럼 '문서 세로 비율' 을 서식마다 손으로 기재
//          → 가로 2단 배치(B/L 의 shipper|consignee)를 세로로 자르면 두 카테고리가 뭉개지고
//          → 표가 하단에 몰린 문서는 items 슬라이스가 빈 여백만 잡습니다.
//   변경:  히트맵이 실제로 반응한 패치를 연결 성분으로 묶고,
//          배타 배정으로 카테고리마다 서로 다른 영역을 확정한 뒤,
//          라벨-값 관계를 고려한 컨텍스트 마진을 붙여 크롭합니다.
//
//  ── 배타 배정을 쓰는 이유 ──
//   header 의 doc_number 와 parties 의 sender_name 은 둘 다 문서 상단에 인쇄됩니다.
//   배타 배정이 없으면 두 카테고리가 같은 영역을 크롭해 LLM 을 중복 호출하고,
//   정작 financials 는 아무 영역도 받지 못합니다.
//   scheduler.rs 의 PLINKO 배타 배정(exclusive_assign_by_score)과 같은 원리입니다.
// =====================================================================

use image::DynamicImage;

use super::preprocessor::patch_index_to_bbox;
use super::vision_encoder::{CategoryHeatmap, PatchGrid};
use crate::utils::ai_utils::exclusive_assign_by_score;

// =====================================================================
// 산출물
// =====================================================================

/// 카테고리 하나에 대한 크롭 계획.
#[derive(Debug, Clone)]
pub struct CropPlan {
    pub category: String,
    /// 원본 이미지 좌표 (x_min, y_min, x_max, y_max)
    pub bbox: (u32, u32, u32, u32),
    /// 이 영역을 확정한 근거 점수 (연결 성분의 최고 surprisal)
    pub score: f32,
    /// 배타 배정 마진. 0 에 가까우면 다른 카테고리와 사실상 동률이었다는 뜻입니다.
    pub margin: f32,
    /// 이 성분에 포함된 패치 수
    pub patch_count: usize,
    /// 이 카테고리에서 가장 강하게 반응한 필드명 (진단용)
    pub top_field: String,
}

/// 연결 성분 하나 (내부 표현)
#[derive(Debug, Clone)]
struct Component {
    /// 패치 인덱스 목록
    indices: Vec<usize>,
    /// 격자 좌표 경계
    r_min: usize,
    r_max: usize,
    c_min: usize,
    c_max: usize,
    /// 이 성분 안의 최고 점수
    peak: f32,
    /// 이 성분 안의 점수 합
    total: f32,
}

// =====================================================================
// ① + ② 활성 패치 선별 → 연결 성분 추출
// =====================================================================

/// 4-이웃 flood fill 로 활성 패치를 성분으로 묶습니다.
///
/// 🌟 [ACTIVATION GATE] surprisal > 0 만 활성으로 봅니다.
///    0 은 극값이론에서 유도된 값(√(2 ln N))이므로 매직 상수가 아니며,
///    "N개를 무작위로 뽑은 기대 최댓값보다 실제로 더 가깝다" 는 뜻입니다.
///    무관한 카테고리는 전 패치가 음수라 성분이 하나도 생기지 않습니다.
fn extract_components(scores: &[f32], rows: usize, cols: usize) -> Vec<Component> {
    let n = rows * cols;
    if scores.len() < n {
        return Vec::new();
    }

    let mut visited = vec![false; n];
    let mut out: Vec<Component> = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }
        if scores[start] <= 0.0 {
            visited[start] = true;
            continue;
        }

        let mut stack = vec![start];
        visited[start] = true;

        let mut indices: Vec<usize> = Vec::new();
        let mut r_min = usize::MAX;
        let mut r_max = 0usize;
        let mut c_min = usize::MAX;
        let mut c_max = 0usize;
        let mut peak = f32::MIN;
        let mut total = 0.0f32;

        while let Some(cur) = stack.pop() {
            let r = cur / cols;
            let c = cur % cols;

            indices.push(cur);
            if r < r_min { r_min = r; }
            if r > r_max { r_max = r; }
            if c < c_min { c_min = c; }
            if c > c_max { c_max = c; }
            if scores[cur] > peak { peak = scores[cur]; }
            total += scores[cur];

            // 4-이웃
            let mut push = |nr: isize, nc: isize, stack: &mut Vec<usize>, visited: &mut Vec<bool>| {
                if nr < 0 || nc < 0 {
                    return;
                }
                let (nr, nc) = (nr as usize, nc as usize);
                if nr >= rows || nc >= cols {
                    return;
                }
                let idx = nr * cols + nc;
                if visited[idx] || scores[idx] <= 0.0 {
                    return;
                }
                visited[idx] = true;
                stack.push(idx);
            };

            push(r as isize - 1, c as isize, &mut stack, &mut visited);
            push(r as isize + 1, c as isize, &mut stack, &mut visited);
            push(r as isize, c as isize - 1, &mut stack, &mut visited);
            push(r as isize, c as isize + 1, &mut stack, &mut visited);
        }

        out.push(Component {
            indices,
            r_min,
            r_max,
            c_min,
            c_max,
            peak,
            total,
        });
    }

    // 강한 성분부터
    out.sort_by(|a, b| b.peak.partial_cmp(&a.peak).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// 🌟 [FRAGMENT MERGE] 같은 행 대역에 있는 인접 성분을 병합합니다.
///
///  문서에서 하나의 논리 블록(예: Shipper 박스)은
///  라벨 텍스트와 값 텍스트 사이에 여백이 있어 성분이 2~3조각으로 쪼개집니다.
///  행 범위가 겹치고 열 간격이 좁으면 같은 블록으로 봅니다.
///
///  판정 기준은 '격자 간격' 이라는 구조적 사실이며 어휘 사전이 아닙니다.
fn merge_adjacent(mut comps: Vec<Component>, cols: usize) -> Vec<Component> {
    if comps.len() <= 1 {
        return comps;
    }

    // 열 간격 허용치: 격자 폭의 1/8 (최소 1). 라벨-값 사이 여백 정도.
    let gap_tol = (cols / 8).max(1);

    let mut changed = true;
    let mut guard = 0usize;
    while changed && guard < 32 {
        guard += 1;
        changed = false;

        'outer: for i in 0..comps.len() {
            for j in (i + 1)..comps.len() {
                let a = &comps[i];
                let b = &comps[j];

                // 행 범위가 겹치는가
                let row_overlap = a.r_min <= b.r_max && b.r_min <= a.r_max;
                if !row_overlap {
                    continue;
                }

                // 열 간격이 허용치 이내인가
                let col_gap = if a.c_max < b.c_min {
                    b.c_min - a.c_max
                } else if b.c_max < a.c_min {
                    a.c_min - b.c_max
                } else {
                    0
                };
                if col_gap > gap_tol {
                    continue;
                }

                // 병합
                let mut merged = Component {
                    indices: a.indices.clone(),
                    r_min: a.r_min.min(b.r_min),
                    r_max: a.r_max.max(b.r_max),
                    c_min: a.c_min.min(b.c_min),
                    c_max: a.c_max.max(b.c_max),
                    peak: a.peak.max(b.peak),
                    total: a.total + b.total,
                };
                merged.indices.extend_from_slice(&b.indices);

                comps[i] = merged;
                comps.remove(j);
                changed = true;
                break 'outer;
            }
        }
    }

    comps.sort_by(|a, b| b.peak.partial_cmp(&a.peak).unwrap_or(std::cmp::Ordering::Equal));
    comps
}

// =====================================================================
// ③ 카테고리 배타 배정
// =====================================================================

/// 전체 카테고리의 성분을 하나의 풀로 모아 배타 배정합니다.
///
/// 🌟 [EXCLUSIVE] scheduler.rs 의 PLINKO 배타 배정과 동일 원리입니다.
///    행렬[category][component] = 그 카테고리가 그 성분에서 얻은 최고 점수.
///    exclusive_assign_by_score 는 절대 점수가 큰 주장부터 1:1 로 잠급니다.
///
///    성분 풀은 '전 카테고리 성분의 합집합' 입니다.
///    서로 다른 카테고리가 같은 픽셀 영역을 잡는 사고를 여기서 막습니다.
struct ComponentPool {
    /// 격자 좌표 경계 목록
    boxes: Vec<(usize, usize, usize, usize)>, // (r_min, r_max, c_min, c_max)
    /// 각 성분의 패치 인덱스 수
    counts: Vec<usize>,
}

/// 두 격자 박스의 IoU. 중복 성분 dedup 에 씁니다.
fn grid_iou(a: (usize, usize, usize, usize), b: (usize, usize, usize, usize)) -> f32 {
    let (ar0, ar1, ac0, ac1) = a;
    let (br0, br1, bc0, bc1) = b;

    let r0 = ar0.max(br0);
    let r1 = ar1.min(br1);
    let c0 = ac0.max(bc0);
    let c1 = ac1.min(bc1);

    if r0 > r1 || c0 > c1 {
        return 0.0;
    }

    let inter = ((r1 - r0 + 1) * (c1 - c0 + 1)) as f32;
    let area_a = ((ar1 - ar0 + 1) * (ac1 - ac0 + 1)) as f32;
    let area_b = ((br1 - br0 + 1) * (bc1 - bc0 + 1)) as f32;
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

// =====================================================================
// ④ 바운딩 박스 확장 + ⑤ 크롭
// =====================================================================

/// 격자 박스 → 원본 픽셀 박스 (컨텍스트 마진 포함).
///
/// 🌟 [CONTEXT MARGIN] 히트맵은 '라벨' 에 반응하지만 값은 라벨의 오른쪽/아래에 있습니다.
///    라벨만 크롭하면 Qwen 3.5 가 읽을 값이 없습니다.
///    문서 레이아웃의 보편 사실(라벨-값은 같은 행 또는 바로 아래)에 근거해
///    가로는 넉넉히, 세로는 적당히 확장합니다.
///    서식별 하드코딩이 아니라 '격자 비율' 기반이므로 어떤 문서에도 동일 적용됩니다.
fn to_pixel_bbox(
    gbox: (usize, usize, usize, usize),
    grid: &PatchGrid,
) -> (u32, u32, u32, u32) {
    let (r_min, r_max, c_min, c_max) = gbox;

    // 가로 마진: 격자 폭의 15% 또는 최소 2패치
    let mx = ((grid.grid_cols as f32 * 0.15) as usize).max(2);
    // 세로 마진: 격자 높이의 6% 또는 최소 1패치
    let my = ((grid.grid_rows as f32 * 0.06) as usize).max(1);

    let r0 = r_min.saturating_sub(my);
    let r1 = (r_max + my).min(grid.grid_rows.saturating_sub(1));
    let c0 = c_min.saturating_sub(mx);
    let c1 = (c_max + mx).min(grid.grid_cols.saturating_sub(1));

    let top_left = patch_index_to_bbox(
        r0 * grid.grid_cols + c0,
        grid.grid_cols,
        grid.patch_size,
        grid.scale_x,
        grid.scale_y,
    );
    let bottom_right = patch_index_to_bbox(
        r1 * grid.grid_cols + c1,
        grid.grid_cols,
        grid.patch_size,
        grid.scale_x,
        grid.scale_y,
    );

    let x0 = top_left.0.min(grid.orig_width.saturating_sub(1));
    let y0 = top_left.1.min(grid.orig_height.saturating_sub(1));
    let x1 = bottom_right.2.min(grid.orig_width);
    let y1 = bottom_right.3.min(grid.orig_height);

    (x0, y0, x1.max(x0 + 1), y1.max(y0 + 1))
}

/// 🌟 [MIN READABLE] 크롭 결과가 너무 작으면 OCR 이 불가능합니다.
///    Qwen 3.5 의 비전 인코더가 실질적으로 읽을 수 있는 하한선까지 박스를 넓힙니다.
///    (원본을 벗어나지 않는 선에서 중심을 유지하며 확장)
fn ensure_min_size(
    bbox: (u32, u32, u32, u32),
    orig_w: u32,
    orig_h: u32,
    min_w: u32,
    min_h: u32,
) -> (u32, u32, u32, u32) {
    let (mut x0, mut y0, mut x1, mut y1) = bbox;

    if x1 - x0 < min_w {
        let need = min_w - (x1 - x0);
        let half = need / 2;
        x0 = x0.saturating_sub(half);
        x1 = (x1 + (need - half)).min(orig_w);
        if x1 - x0 < min_w {
            x0 = x1.saturating_sub(min_w);
        }
    }
    if y1 - y0 < min_h {
        let need = min_h - (y1 - y0);
        let half = need / 2;
        y0 = y0.saturating_sub(half);
        y1 = (y1 + (need - half)).min(orig_h);
        if y1 - y0 < min_h {
            y0 = y1.saturating_sub(min_h);
        }
    }

    (x0, y0, x1.min(orig_w), y1.min(orig_h))
}

// =====================================================================
// 공개 API
// =====================================================================

/// 🌟 [STEP 3] 히트맵 목록 → 카테고리별 크롭 계획.
///
///  ── 처리 순서 ──
///   ① 카테고리마다 활성 패치를 연결 성분으로 추출
///   ② 인접 조각 병합 (라벨-값 여백으로 쪼개진 성분 복구)
///   ③ 전 카테고리 성분을 하나의 풀로 모아 IoU dedup
///   ④ (카테고리 × 성분) 행렬 → 배타 배정 (1:1)
///   ⑤ 격자 박스 → 원본 픽셀 박스 + 컨텍스트 마진 + 최소 해상도 보장
pub fn plan_crops(
    heatmaps: &[CategoryHeatmap],
    grid: &PatchGrid,
    emit: &dyn Fn(&str),
) -> Vec<CropPlan> {
    if heatmaps.is_empty() || grid.len() == 0 {
        return Vec::new();
    }

    let rows = grid.grid_rows;
    let cols = grid.grid_cols;

    // ── ①② 카테고리별 성분 추출 + 병합 ──
    let mut per_cat: Vec<(String, String, Vec<Component>)> = Vec::new();
    for hm in heatmaps.iter() {
        let comps = merge_adjacent(extract_components(&hm.scores, rows, cols), cols);
        if comps.is_empty() {
            emit(&format!(
                "    ⚪ [NO REGION] '{}' 는 활성 패치가 없어 크롭 대상에서 제외합니다.",
                hm.category
            ));
            continue;
        }
        emit(&format!(
            "    🧩 [COMPONENTS] '{}' | 성분 {}개 | Top: {}({:+.4})",
            hm.category,
            comps.len(),
            if hm.top_field.is_empty() { "-" } else { &hm.top_field },
            hm.top_score
        ));
        per_cat.push((hm.category.clone(), hm.top_field.clone(), comps));
    }

    if per_cat.is_empty() {
        return Vec::new();
    }

    // ── ③ 성분 풀 구축 (IoU dedup) ──
    let mut pool = ComponentPool {
        boxes: Vec::new(),
        counts: Vec::new(),
    };
    // (cat_idx, comp) -> pool_idx
    let mut cat_pool_scores: Vec<Vec<(usize, f32)>> = Vec::with_capacity(per_cat.len());

    for (_, _, comps) in per_cat.iter() {
        let mut mine: Vec<(usize, f32)> = Vec::new();
        for comp in comps.iter() {
            let gbox = (comp.r_min, comp.r_max, comp.c_min, comp.c_max);

            // 이미 풀에 유사 영역이 있으면 재사용합니다.
            let mut pool_idx: Option<usize> = None;
            for (pi, existing) in pool.boxes.iter().enumerate() {
                if grid_iou(gbox, *existing) >= 0.5 {
                    pool_idx = Some(pi);
                    break;
                }
            }

            let pi = match pool_idx {
                Some(v) => v,
                None => {
                    pool.boxes.push(gbox);
                    pool.counts.push(comp.indices.len());
                    pool.boxes.len() - 1
                }
            };

            // 같은 풀 인덱스에 여러 성분이 매핑되면 최고 점수만 남깁니다.
            if let Some(slot) = mine.iter_mut().find(|(i, _)| *i == pi) {
                if comp.peak > slot.1 {
                    slot.1 = comp.peak;
                }
            } else {
                mine.push((pi, comp.peak));
            }
        }
        cat_pool_scores.push(mine);
    }

    let pool_n = pool.boxes.len();
    if pool_n == 0 {
        return Vec::new();
    }

    emit(&format!(
        "    🎯 [REGION POOL] 후보 영역 {}개 | 경쟁 카테고리 {}개",
        pool_n,
        per_cat.len()
    ));

    // ── ④ 배타 배정 ──
    //    행렬[cat][pool] = 점수. 무효 칸은 -1.0 (exclusive_assign_by_score 계약).
    let mut matrix: Vec<Vec<f32>> = vec![vec![-1.0f32; pool_n]; per_cat.len()];
    for (ci, mine) in cat_pool_scores.iter().enumerate() {
        for (pi, score) in mine.iter() {
            matrix[ci][*pi] = *score;
        }
    }

    let assign = exclusive_assign_by_score(&matrix, 0.0, 0.0);

    // ── ⑤ 픽셀 박스 확정 ──
    //    최소 해상도: 원본의 12% 폭 / 6% 높이 (그보다 작으면 OCR 불가)
    let min_w = ((grid.orig_width as f32 * 0.12) as u32).max(64);
    let min_h = ((grid.orig_height as f32 * 0.06) as u32).max(48);

    let mut plans: Vec<CropPlan> = Vec::new();

    for (ci, a) in assign.iter().enumerate() {
        let (pi, score, margin) = match a {
            Some(v) => *v,
            None => {
                emit(&format!(
                    "    ⚪ [UNASSIGNED] '{}' 는 다른 카테고리에 영역을 선점당해 크롭하지 않습니다.",
                    per_cat[ci].0
                ));
                continue;
            }
        };

        let gbox = pool.boxes[pi];
        let raw = to_pixel_bbox(gbox, grid);
        let bbox = ensure_min_size(raw, grid.orig_width, grid.orig_height, min_w, min_h);

        emit(&format!(
            "    ✂️ [CROP PLAN] '{}' ← grid(r{}~{}, c{}~{}) → px({},{})-({},{}) | Score: {:+.4} | Margin: {:+.4} | Field: {}",
            per_cat[ci].0,
            gbox.0, gbox.1, gbox.2, gbox.3,
            bbox.0, bbox.1, bbox.2, bbox.3,
            score, margin,
            if per_cat[ci].1.is_empty() { "-" } else { &per_cat[ci].1 }
        ));

        plans.push(CropPlan {
            category: per_cat[ci].0.clone(),
            bbox,
            score,
            margin,
            patch_count: pool.counts[pi],
            top_field: per_cat[ci].1.clone(),
        });
    }

    plans
}

/// 크롭 계획 하나를 실제 이미지로 잘라냅니다.
///
/// 🌟 [UPSCALE] 잘린 조각이 작으면 Qwen 3.5 의 비전 인코더가 글자를 못 읽습니다.
///    짧은 변이 target_short 미만이면 Lanczos3 로 확대합니다.
///    (원본 픽셀이 이미 존재하므로 정보 손실 없이 가독성만 올립니다)
pub fn crop_region(
    image: &DynamicImage,
    plan: &CropPlan,
    target_short: u32,
) -> DynamicImage {
    let (x0, y0, x1, y1) = plan.bbox;
    let w = x1.saturating_sub(x0).max(1);
    let h = y1.saturating_sub(y0).max(1);

    let cropped = image.crop_imm(x0, y0, w, h);

    let short = w.min(h);
    if short >= target_short {
        return cropped;
    }

    let factor = target_short as f32 / short as f32;
    let nw = ((w as f32 * factor).round() as u32).max(1);
    let nh = ((h as f32 * factor).round() as u32).max(1);

    // 지나친 확대는 메모리만 먹고 이득이 없으므로 상한을 둡니다.
    let nw = nw.min(2048);
    let nh = nh.min(2048);

    cropped.resize_exact(nw, nh, image::imageops::FilterType::Lanczos3)
}

/// 🌟 [FALLBACK] 히트맵이 아무 영역도 못 찾았을 때의 안전망.
///
///  ── 왜 필요한가 ──
///   get_trade_doc_slice_config 를 폐기하면 '항상 무언가는 크롭한다' 는 보장이 사라집니다.
///   스캔 품질이 나쁘거나 앵커가 전부 음수면 크롭이 0건이 되어 추출 자체가 실패합니다.
///
///  ── 무엇을 하는가 ──
///   고정 비율 표를 되살리지 않습니다. 대신 '문서 전체' 를 단일 영역으로 넘겨
///   Qwen 3.5 가 원본을 그대로 읽게 합니다.
///   좌표를 창작하는 것보다 원본을 통째로 주는 편이 안전합니다.
pub fn whole_page_fallback(categories: &[&str], grid: &PatchGrid) -> Vec<CropPlan> {
    categories
        .iter()
        .map(|c| CropPlan {
            category: c.to_string(),
            bbox: (0, 0, grid.orig_width, grid.orig_height),
            score: 0.0,
            margin: 0.0,
            patch_count: grid.len(),
            top_field: String::new(),
        })
        .collect()
}
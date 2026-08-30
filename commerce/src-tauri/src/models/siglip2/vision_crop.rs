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

/// 양수 점수의 (평균, 표준편차, 개수) 를 돌려줍니다.
///
/// 🌟 [WHY] surprisal > 0 만으로는 게이트가 너무 헐겁습니다.
///    실측(CI 인보이스)에서 cargo 카테고리는 252 패치 중 158개(63%)가 양수였고,
///    4-이웃 flood fill 이 문서 전면을 단일 성분으로 묶어 버렸습니다.
///    그 결과 배타 배정에서 header 가 후보를 하나도 못 받고 굶었습니다.
///    분포의 (평균 + 표준편차) 는 그 분포에서 유도된 값이므로 매직 상수가 아닙니다.
fn positive_stats(scores: &[f32]) -> (f32, f32, usize) {
    let pos: Vec<f32> = scores.iter().copied().filter(|s| *s > 0.0).collect();
    if pos.len() < 2 {
        return (0.0, 0.0, pos.len());
    }
    let n = pos.len() as f32;
    let mean = pos.iter().sum::<f32>() / n;
    let var = pos.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n;
    (mean, var.sqrt(), pos.len())
}

/// 성분 seed 로 삼을 '핵심(core)' 임계값. (평균 + 표준편차)
fn core_threshold(scores: &[f32]) -> f32 {
    let (mean, sd, cnt) = positive_stats(scores);
    if cnt < 4 {
        return 0.0;
    }
    mean + sd
}

/// 4-이웃 flood fill 로 활성 패치를 성분으로 묶습니다.
///
/// 🌟 [ACTIVATION GATE v2] 게이트를 인자로 받습니다.
///    · gate = 0.0 → 기존 동작 (surprisal > 0)
///    · gate = mean+sd → 봉우리만 seed 로 삼아 거대 성분을 방지
///    확장(이웃 편입)도 같은 게이트를 적용해야 flood 가 다시 전면으로 번지지 않습니다.
fn extract_components(scores: &[f32], rows: usize, cols: usize, gate: f32) -> Vec<Component> {
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
        if scores[start] <= gate {
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
                if visited[idx] || scores[idx] <= gate {
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

/// 🌟 [CONTENT MASK] 전 카테고리 히트맵의 패치별 최댓값.
///
///  ── 무엇을 근사하는가 ──
///   "이 패치가 어떤 스키마 개념과도 관련이 있는 정도" 입니다.
///   문서에서 잉크가 찍힌 영역(라벨 텍스트 + 값 텍스트 + 표 셀)의 근사가 되고,
///   여백·괘선·로고는 전 카테고리에서 음수라 자연히 제외됩니다.
///
///  ── 어디에 쓰는가 ──
///   라벨은 히트맵에 반응하지만 값은 반응하지 않는 경우가 많습니다.
///   (앵커가 'shipper / exporter' 라서 라벨 셀만 잡히고 회사명 셀은 안 잡힘)
///   같은 행 밴드 안에서 content 가 이어지는 만큼 좌우로 넓혀
///   '끊어진 테두리로 분리된 값 셀' 을 함께 담습니다.
///
///  반환: (mask, gate) — gate 는 mask 양수값의 평균
fn build_content_mask(heatmaps: &[CategoryHeatmap], n: usize) -> (Vec<f32>, f32) {
    let mut mask = vec![f32::MIN; n];
    for hm in heatmaps.iter() {
        let m = n.min(hm.scores.len());
        for i in 0..m {
            if hm.scores[i] > mask[i] {
                mask[i] = hm.scores[i];
            }
        }
    }
    let (mean, _, cnt) = positive_stats(&mask);
    let gate = if cnt < 4 { 0.0 } else { mean };
    (mask, gate)
}

/// 🌟 [OVERSIZED SPLIT] 면적 상한을 넘는 성분을 봉우리 단위로 재분할합니다.
///
///  cargo 처럼 앵커가 광범위하게 반응하는 카테고리는 게이트를 올려도
///  하나의 성분이 문서 대부분을 덮을 수 있습니다.
///  그 성분 '내부' 점수 분포로 임계를 다시 계산해 봉우리만 남깁니다.
fn split_oversized(
    comp: &Component,
    scores: &[f32],
    rows: usize,
    cols: usize,
) -> Vec<Component> {
    let n = rows * cols;
    let mut local = vec![f32::MIN; n];
    for &i in comp.indices.iter() {
        if i < n {
            local[i] = scores[i];
        }
    }
    let gate = core_threshold(&local);
    if gate <= 0.0 {
        return vec![comp.clone()];
    }
    let sub = extract_components(&local, rows, cols, gate);
    if sub.is_empty() {
        vec![comp.clone()]
    } else {
        sub
    }
}

/// 🌟 [ROW BAND EXPANSION] 라벨↔값 페어를 담기 위한 가로 확장.
///
///  문서 레이아웃의 보편 사실:
///    · 라벨과 값은 같은 행 밴드에 인쇄된다 (좌: 라벨 / 우: 값, 또는 그 반대)
///    · 셀 테두리가 끊겨 있어도 행 위치는 유지된다
///  성분의 행 범위를 고정한 채, 그 밴드 안에서 content 가 이어지는 열까지만 넓힙니다.
///  content 가 끊기면 즉시 멈추므로 이웃 블록을 삼키지 않습니다.
fn expand_row_band(
    comp: &Component,
    content: &[f32],
    gate: f32,
    cols: usize,
) -> (usize, usize, usize, usize) {
    let mut c_min = comp.c_min;
    let mut c_max = comp.c_max;

    let band_has = |c: usize| -> bool {
        (comp.r_min..=comp.r_max).any(|r| {
            let idx = r * cols + c;
            idx < content.len() && content[idx] > gate
        })
    };

    while c_min > 0 && band_has(c_min - 1) {
        c_min -= 1;
    }
    while c_max + 1 < cols && band_has(c_max + 1) {
        c_max += 1;
    }

    (comp.r_min, comp.r_max, c_min, c_max)
}

/// 🌟 [TABLE UNION] items / containers 전용. 표 전체를 하나의 박스로 잡습니다.
///
///  ── 왜 별도 처리인가 ──
///   items 는 배열 스키마입니다. 한 셀만 크롭하면 나머지 행을 통째로 잃습니다.
///   실측에서 items 는 단일 패치(r12,c12)만 배정받아 상품표(y 550~634)를
///   전혀 보지 못했습니다.
///
///  ── 판정 근거 ──
///   표 행은 '가로로 content 가 촘촘히 이어진다' 는 구조적 특징을 갖습니다.
///   격자 폭의 1/3 이상이 content 인 행을 '표 행' 으로 보고 위아래로 확장합니다.
fn table_union(
    comps: &[Component],
    content: &[f32],
    gate: f32,
    rows: usize,
    cols: usize,
) -> Option<(usize, usize, usize, usize)> {
    let peak = comps.iter().max_by(|a, b| {
        a.peak.partial_cmp(&b.peak).unwrap_or(std::cmp::Ordering::Equal)
    })?;

    let dense_cols = (cols / 3).max(2);
    let row_is_table = |r: usize| -> bool {
        (0..cols)
            .filter(|&c| {
                let idx = r * cols + c;
                idx < content.len() && content[idx] > gate
            })
            .count()
            >= dense_cols
    };

    let mut r0 = peak.r_min;
    let mut r1 = peak.r_max;
    while r0 > 0 && row_is_table(r0 - 1) {
        r0 -= 1;
    }
    while r1 + 1 < rows && row_is_table(r1 + 1) {
        r1 += 1;
    }

    // 표는 가로로 넓습니다. 밴드 안에서 content 가 있는 최좌·최우까지 잡습니다.
    let mut c0 = cols;
    let mut c1 = 0usize;
    for r in r0..=r1 {
        for c in 0..cols {
            let idx = r * cols + c;
            if idx < content.len() && content[idx] > gate {
                if c < c0 { c0 = c; }
                if c > c1 { c1 = c; }
            }
        }
    }
    if c0 > c1 {
        c0 = peak.c_min;
        c1 = peak.c_max;
    }

    Some((r0, r1, c0, c1))
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

    // 🌟 열 간격 허용치: 격자 폭의 1/6 (최소 2).
    //    실측 격자가 14열이라 1/8 = 1 이 되어 라벨 셀과 값 셀 사이의
    //    테두리 여백(2패치 = 약 114px)을 넘지 못했습니다.
    //    1/6 = 2 로 넓혀 '끊어진 테두리로 분리된 같은 행 셀' 이 병합되게 합니다.
    let gap_tol = (cols / 6).max(2);

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

/// 🌟 [PRESENCE GATE] 이 문서에 실제로 인쇄된 카테고리만 크롭 경쟁에 넣습니다.
///
///  ── 왜 '> 0' 게이트가 더는 안 통하는가 ──
///   score_patches_bank_neutral 이 √(2 ln N) 차감을 폐기하면서
///   점수의 0 은 더 이상 '극값 기대치' 가 아니라 '이중 센터링 후 평균' 이 되었습니다.
///   실측(CI 인보이스)에서 카테고리 11개 중 9개가 활성 패치 80% 이상을 기록했고
///   (HEATMAP FULL PAGE RISK), 상업송장에 존재하지도 않는 insurance / settlement 가
///   크롭을 받아 2B 모델이 빈 영역에서 값을 창작했습니다.
///     ✅ [insurance]  신규 1건 → {"INSURANCE": {"...ND CORRECT."}}
///     ✅ [settlement] 신규 1건 → {"SETTLEMENT": {"items": [...]}}
///   그 두 크롭이 차지한 영역은 header 가 가져갔어야 할 영역입니다.
///
///  ── 대체 기준 (무모수 / 임계치 없음) ──
///   "패치 하나라도 자기가 argmax 인가" 만 봅니다.
///   전 페이지 어느 패치도 그 카테고리로 가장 잘 설명되지 않는다면
///   그 축은 이 문서에 인쇄되어 있지 않습니다.
///   임계치가 없고 카테고리 수·뱅크 크기에도 무관합니다.
///
///  ⚠️ [HEADER EXEMPT] header 만은 예외입니다.
///     doc_number 는 문서 기본키라 잃으면 릴레이가 영구히 끊깁니다.
///     다른 필드는 다음 스캔에서 채울 수 있지만 기본키는 그럴 수 없습니다.
fn presence_gate(
    heatmaps: &[CategoryHeatmap],
    n: usize,
    emit: &dyn Fn(&str),
) -> std::collections::HashSet<String> {
    use std::collections::{HashMap, HashSet};
    let mut wins: HashMap<String, usize> = HashMap::new();
    for i in 0..n {
        let mut best = f32::MIN;
        let mut owner: Option<&str> = None;
        for hm in heatmaps.iter() {
            if i >= hm.scores.len() {
                continue;
            }
            if hm.scores[i] > best {
                best = hm.scores[i];
                owner = Some(hm.category.as_str());
            }
        }
        if let Some(o) = owner {
            if best > 0.0 {
                *wins.entry(o.to_string()).or_insert(0) += 1;
            }
        }
    }
    let mut out: HashSet<String> = HashSet::new();
    for hm in heatmaps.iter() {
        let w = wins.get(&hm.category).copied().unwrap_or(0);
        // header 는 문서 기본키를 담당하므로 게이트를 면제합니다.
        if w > 0 || hm.category == "header" {
            out.insert(hm.category.clone());
            if w == 0 {
                emit(&format!(
                    "    🪪 [PRESENCE GATE / HEADER EXEMPT] 'header' 는 argmax 패치가 0개지만 문서 기본키(doc_number)를 담당하므로 면제합니다. (Top: {:+.4})",
                    hm.top_score
                ));
            }
        } else {
            emit(&format!(
                "    ⚪ [PRESENCE GATE] '{}' 는 패치 {}개 중 단 한 곳에서도 최강 설명이 되지 못했습니다. 이 문서에 없는 축이므로 크롭하지 않습니다. (Top: {:+.4})",
                hm.category, n, hm.top_score
            ));
        }
    }
    out
}

/// 🌟 [IDENTITY BAND GUARANTEE] 'header' 는 반드시 문서 상단 식별 밴드를 점유합니다.
///
///  ── 왜 header 만 특별 취급하는가 ──
///   doc_number 는 다른 필드와 성격이 다릅니다.
///     · 값이 비면 resolve_trade_doc_identity 가 내용 합성 키로 폴백하고
///     · 그 키는 참조 번호가 아니므로 릴레이가 영구히 끊기며
///     · 저장 후에는 되돌릴 방법이 없습니다.
///   나머지 필드는 비어도 다음 스캔에서 채울 수 있지만 기본키는 그럴 수 없습니다.
///
///  ── 실측 사고 ──
///   확정된 11개 크롭의 y 최솟값이 286px 이었습니다.
///   INVOICE NUMBER 'CI-43726' 은 y≈181 이라 단 한 번도 전달되지 않았고,
///   header 는 4순위로 밀려 VAT/EORI 행(px 344~574)을 받아
///   라벨 텍스트 'EXPORTER VAT/EORI' 를 값으로 복사했습니다.
///
///  ── 밴드 정의 (레이아웃 구조 사실 / 어휘 사전 아님) ──
///   run_title_gate 가 상단 30% 를 제목 밴드로 보는 것과 같은 근거입니다.
///   서식 전문(제목) 아래 첫 정보 블록에 문서 자신의 번호가 인쇄됩니다.
///   '내용이 처음 나타나는 행의 다음 행 ~ 상단 40%' 를 식별 밴드로 확정합니다.
fn ensure_identity_band_crop(
    plans: &mut Vec<CropPlan>,
    heatmaps: &[CategoryHeatmap],
    content: &[f32],
    content_gate: f32,
    grid: &PatchGrid,
    emit: &dyn Fn(&str),
) {
    let rows = grid.grid_rows;
    let cols = grid.grid_cols;
    if rows < 4 || cols == 0 {
        return;
    }
    let hm = match heatmaps.iter().find(|h| h.category == "header") {
        Some(h) => h,
        None => return,
    };

    let row_has_content = |r: usize| -> bool {
        (0..cols).any(|c| {
            let i = r * cols + c;
            i < content.len() && content[i] > content_gate
        })
    };
    let mut title_row = 0usize;
    while title_row + 1 < rows && !row_has_content(title_row) {
        title_row += 1;
    }
    let band_start = (title_row + 1).min(rows - 1);
    let band_end = ((rows as f32 * 0.40) as usize)
        .max(band_start + 1)
        .min(rows - 1);

    let cw = grid.orig_width as f32 / cols as f32;
    let ch = grid.orig_height as f32 / rows as f32;
    let mut tot = 0usize;
    let mut cov = 0usize;
    for r in band_start..=band_end {
        for c in 0..cols {
            let i = r * cols + c;
            if i >= content.len() || content[i] <= content_gate {
                continue;
            }
            tot += 1;
            let cx = (c as f32 + 0.5) * cw;
            let cy = (r as f32 + 0.5) * ch;
            let hit = plans.iter().any(|p| {
                p.category == "header"
                    && cx >= p.bbox.0 as f32
                    && cx <= p.bbox.2 as f32
                    && cy >= p.bbox.1 as f32
                    && cy <= p.bbox.3 as f32
            });
            if hit {
                cov += 1;
            }
        }
    }
    if tot == 0 {
        emit("    ⚪ [IDENTITY BAND] 상단 식별 밴드에 내용 패치가 없어 추가 크롭하지 않습니다.");
        return;
    }
    if cov * 2 >= tot {
        emit(&format!(
            "    ✅ [IDENTITY BAND] 'header' 가 식별 밴드 r{}~{} 의 내용 {}/{} 를 이미 점유하고 있습니다.",
            band_start, band_end, cov, tot
        ));
        return;
    }

    let mut c0 = cols;
    let mut c1 = 0usize;
    for r in band_start..=band_end {
        for c in 0..cols {
            let i = r * cols + c;
            if i < content.len() && content[i] > content_gate {
                if c < c0 {
                    c0 = c;
                }
                if c > c1 {
                    c1 = c;
                }
            }
        }
    }
    if c0 > c1 {
        c0 = 0;
        c1 = cols - 1;
    }

    let raw = to_pixel_bbox((band_start, band_end, c0, c1), grid);
    let min_w = ((grid.orig_width as f32 * 0.12) as u32).max(64);
    let min_h = ((grid.orig_height as f32 * 0.06) as u32).max(48);
    let bbox = ensure_min_size(raw, grid.orig_width, grid.orig_height, min_w, min_h);

    let mut peak = f32::MIN;
    let m = (rows * cols).min(hm.scores.len());
    for r in band_start..=band_end {
        for c in c0..=c1 {
            let i = r * cols + c;
            if i < m && hm.scores[i] > peak {
                peak = hm.scores[i];
            }
        }
    }

    emit(&format!(
        "    🪪 [IDENTITY BAND GUARANTEE] 'header' 가 식별 밴드 내용을 {}/{} 밖에 못 담아 전용 크롭을 추가합니다. r{}~{} c{}~{} → px({},{})-({},{}) | Peak: {:+.4}",
        cov, tot, band_start, band_end, c0, c1, bbox.0, bbox.1, bbox.2, bbox.3,
        if peak == f32::MIN { 0.0 } else { peak }
    ));
    plans.push(CropPlan {
        category: "header".to_string(),
        bbox,
        score: if peak == f32::MIN { 0.0 } else { peak },
        margin: 0.0,
        patch_count: (band_end - band_start + 1) * (c1 - c0 + 1),
        top_field: "doc_number".to_string(),
    });
}

/// 🌟 [COVERAGE GUARANTEE] 어떤 크롭에도 담기지 않은 '내용 행 밴드' 를 구제합니다.
///
///  ── 왜 필요한가 (실측 사고) ──
///   확정된 11개 크롭의 y 최솟값이 286px 이었습니다. 즉 r0~r4(상단 27.7%)가
///   단 한 번도 LLM 에게 전달되지 않았고, 그 밴드의
///     INVOICE NUMBER CI-43726 / AIRWAYBILL 93763111837 / EXPORT REFERENCE ORD32829
///     DATE OF EXPORTATION Apr-19-2022 / EXPORTER / CONSIGNEE
///   가 통째로 소실되었습니다.
///   그 결과 doc_number 도 reference_* 도 없어 릴레이 키가 0개가 되었습니다.
///
///  ── 판정 근거 ──
///   build_content_mask 가 만든 '내용 패치'(전 카테고리 히트맵 최댓값 > 평균)는
///   "어떤 스키마 개념과도 관련이 있는 잉크" 의 근사입니다.
///   그 패치가 어느 크롭에도 안 들어갔다면 회수 불가능한 손실입니다.
///   행 단위로 커버리지를 재고, 절반도 못 담긴 행을 밴드로 묶어 구제합니다.
///
///  ── 왜 겹침을 허용하는가 ──
///   STARVATION RESCUE 와 같은 판단입니다.
///   크롭이 겹치는 비용은 LLM 호출 1회지만, 필드를 통째로 잃는 비용은 영구적입니다.
fn rescue_uncovered_bands(
    plans: &mut Vec<CropPlan>,
    heatmaps: &[CategoryHeatmap],
    content: &[f32],
    content_gate: f32,
    grid: &PatchGrid,
    emit: &dyn Fn(&str),
) {
    let rows = grid.grid_rows;
    let cols = grid.grid_cols;
    if rows == 0 || cols == 0 || heatmaps.is_empty() {
        return;
    }
    let cw = grid.orig_width as f32 / cols as f32;
    let ch = grid.orig_height as f32 / rows as f32;

    let mut need: Vec<bool> = vec![false; rows];
    let mut total_lost = 0usize;
    for r in 0..rows {
        let mut tot = 0usize;
        let mut cov = 0usize;
        for c in 0..cols {
            let i = r * cols + c;
            if i >= content.len() || content[i] <= content_gate {
                continue;
            }
            tot += 1;
            let cx = (c as f32 + 0.5) * cw;
            let cy = (r as f32 + 0.5) * ch;
            let hit = plans.iter().any(|p| {
                cx >= p.bbox.0 as f32
                    && cx <= p.bbox.2 as f32
                    && cy >= p.bbox.1 as f32
                    && cy <= p.bbox.3 as f32
            });
            if hit {
                cov += 1;
            }
        }
        if tot > 0 && cov * 2 < tot {
            need[r] = true;
            total_lost += tot - cov;
        }
    }
    if total_lost == 0 {
        emit("    ✅ [COVERAGE GUARANTEE] 모든 내용 행이 최소 하나의 크롭에 포함되어 있습니다.");
        return;
    }

    let mut bands: Vec<(usize, usize)> = Vec::new();
    let mut r = 0usize;
    while r < rows {
        if !need[r] {
            r += 1;
            continue;
        }
        let start = r;
        while r + 1 < rows && need[r + 1] {
            r += 1;
        }
        bands.push((start, r));
        r += 1;
    }

    let min_w = ((grid.orig_width as f32 * 0.12) as u32).max(64);
    let min_h = ((grid.orig_height as f32 * 0.06) as u32).max(48);

    for (r0, r1) in bands {
        let mut c0 = cols;
        let mut c1 = 0usize;
        for rr in r0..=r1 {
            for c in 0..cols {
                let i = rr * cols + c;
                if i < content.len() && content[i] > content_gate {
                    if c < c0 {
                        c0 = c;
                    }
                    if c > c1 {
                        c1 = c;
                    }
                }
            }
        }
        if c0 > c1 {
            c0 = 0;
            c1 = cols.saturating_sub(1);
        }

        let mut owner = String::new();
        let mut owner_field = String::new();
        let mut best = f32::MIN;
        for hm in heatmaps.iter() {
            let m = (rows * cols).min(hm.scores.len());
            for rr in r0..=r1 {
                for c in c0..=c1 {
                    let i = rr * cols + c;
                    if i >= m {
                        continue;
                    }
                    if hm.scores[i] > best {
                        best = hm.scores[i];
                        owner = hm.category.clone();
                        owner_field = hm.top_field.clone();
                    }
                }
            }
        }
        if owner.is_empty() {
            continue;
        }

        // 이미 같은 카테고리·같은 밴드로 추가된 식별 밴드 크롭과 중복되면 건너뜁니다.
        let raw = to_pixel_bbox((r0, r1, c0, c1), grid);
        let bbox = ensure_min_size(raw, grid.orig_width, grid.orig_height, min_w, min_h);
        let dup = plans.iter().any(|p| {
            p.category == owner
                && p.bbox.1 <= bbox.1
                && p.bbox.3 >= bbox.3
                && p.bbox.0 <= bbox.0
                && p.bbox.2 >= bbox.2
        });
        if dup {
            continue;
        }

        emit(&format!(
            "    🩹 [COVERAGE RESCUE] 미커버 행 밴드 r{}~{} (c{}~{}) 를 '{}' 소유로 추가 크롭합니다. → px({},{})-({},{}) | Peak: {:+.4}",
            r0, r1, c0, c1, owner, bbox.0, bbox.1, bbox.2, bbox.3, best
        ));
        plans.push(CropPlan {
            category: owner,
            bbox,
            score: best,
            margin: 0.0,
            patch_count: (r1 - r0 + 1) * (c1 - c0 + 1),
            top_field: owner_field,
        });
    }
}

// =====================================================================
// 공개 API
// =====================================================================

/// 🌟 [STEP 3] 히트맵 목록 → 카테고리별 크롭 계획.
///
///  ── 처리 순서 ──
///   ⓪ CONTENT MASK + PRESENCE GATE (이 문서에 없는 카테고리 제외)
///   ① 카테고리마다 활성 패치를 연결 성분으로 추출
///   ② 인접 조각 병합 (라벨-값 여백으로 쪼개진 성분 복구)
///   ③ 전 카테고리 성분을 하나의 풀로 모아 IoU dedup
///   ④ (카테고리 × 성분) 행렬 → 배타 배정 (1:1)
///   ⑤ 격자 박스 → 원본 픽셀 박스 + 컨텍스트 마진 + 최소 해상도 보장
///   ⑥ IDENTITY BAND GUARANTEE (header 가 문서 기본키 밴드를 반드시 점유)
///   ⑦ COVERAGE GUARANTEE (어떤 크롭에도 안 담긴 내용 행 밴드 구제)
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
    let n = rows * cols;

    // ── ⓪ CONTENT MASK ──
    //    라벨만 반응하고 값은 반응하지 않는 경우를 보정하기 위한 '잉크 지도' 입니다.
    let (content, content_gate) = build_content_mask(heatmaps, n);
    let content_cnt = content.iter().filter(|v| **v > content_gate).count();
    emit(&format!(
        "    🗺️ [CONTENT MASK] 내용 패치 {}/{} | gate {:+.4} (라벨↔값 행 밴드 확장 기준)",
        content_cnt, n, content_gate
    ));

    // 🌟 [PRESENCE GATE] BANK-NEUTRAL 이후 '> 0' 게이트가 의미를 잃었으므로
    //    "패치 하나라도 자기가 argmax 인가" 라는 무모수 기준으로 대체합니다.
    let present = presence_gate(heatmaps, n, emit);

    // 🌟 [AREA CAP] 한 카테고리가 문서 전체를 독점할 수 없다는 구조적 사실.
    //    전체 면적을 '실제로 경쟁하는' 카테고리 수로 나눈 값이 상한입니다.
    //    존재하지 않는 카테고리까지 분모에 넣으면 상한이 부당하게 낮아집니다.
    let area_cap = (n / present.len().max(1)).max(4);

    // 🌟 [LOG] 히트맵 → 크롭 전환 전 전체 상태 요약
    emit(&format!(
        "    📊 [PLAN_CROPS INPUT] 히트맵 {}개 (존재 판정 통과 {}개) | 격자 {}x{}={} | area_cap={} | content 활성 {}/{}",
        heatmaps.len(), present.len(), grid.grid_rows, grid.grid_cols, n, area_cap, content_cnt, n
    ));

    // 🌟 [LOG] 각 히트맵의 활성 패치 비율 — 잘림 감지
    for hm in heatmaps.iter() {
        let hot = hm.scores.iter().filter(|s| **s > 0.0).count();
        let ratio = if n > 0 { hot as f32 / n as f32 } else { 0.0 };
        if ratio > 0.60 {
            emit(&format!(
                "    ⚠️ [HEATMAP FULL PAGE RISK] '{}' 활성 패치 {}/{} ({:.0}%) — 전체 페이지 점유. 표 구조가 전체를 덮거나 히트맵 과잉 확산 가능.",
                hm.category, hot, n, ratio * 100.0
            ));
        }
        if hot > 0 && hot < 5 {
            emit(&format!(
                "    ⚠️ [HEATMAP SPARSE RISK] '{}' 활성 패치 {}개 — 패치 부족으로 크롭 정밀도 저하 가능.",
                hm.category, hot
            ));
        }
    }

    // ── ①② 카테고리별 성분 추출 → 거대 성분 재분할 → 인접 병합 → 행 밴드 확장 ──
    //    (category, top_field, gboxes, peaks, counts)
    let mut per_cat: Vec<(String, String, Vec<(usize, usize, usize, usize)>, Vec<f32>, Vec<usize>)> =
        Vec::new();

    for hm in heatmaps.iter() {
        // 🌟 [PRESENCE GATE] 이 문서에 인쇄되지 않은 축은 크롭 경쟁 자체에 넣지 않습니다.
        //    빈 영역을 2B 모델에게 보내면 반드시 무언가를 창작합니다.
        if !present.contains(&hm.category) {
            continue;
        }
        // ① 봉우리 게이트로 seed 확보. 봉우리가 없으면 기존 게이트(0.0)로 폴백.
        let gate = core_threshold(&hm.scores);
        let mut comps = extract_components(&hm.scores, rows, cols, gate);
        if comps.is_empty() {
            comps = extract_components(&hm.scores, rows, cols, 0.0);
        }
        if comps.is_empty() {
            emit(&format!(
                "    ⚪ [NO REGION] '{}' 는 활성 패치가 없어 크롭 대상에서 제외합니다.",
                hm.category
            ));
            continue;
        }

        // ② 거대 성분 재분할
        let mut split: Vec<Component> = Vec::new();
        for c in comps.into_iter() {
            if c.indices.len() > area_cap {
                let parts = split_oversized(&c, &hm.scores, rows, cols);
                emit(&format!(
                    "    ✂️ [OVERSIZED SPLIT] '{}' | 성분 {}패치 > 상한 {}패치 → 봉우리 {}개로 재분할",
                    hm.category,
                    c.indices.len(),
                    area_cap,
                    parts.len()
                ));
                split.extend(parts);
            } else {
                split.push(c);
            }
        }

        let comps = merge_adjacent(split, cols);
        if comps.is_empty() {
            continue;
        }

        // ③ 표 전용 카테고리는 행 밴드를 union 해 표 전체를 잡습니다.
        let is_table_cat = hm.category == "items" || hm.category == "containers";

        let mut gboxes: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut peaks: Vec<f32> = Vec::new();
        let mut counts: Vec<usize> = Vec::new();

        if is_table_cat {
            if let Some(tb) = table_union(&comps, &content, content_gate, rows, cols) {
                let area = (tb.1 - tb.0 + 1) * (tb.3 - tb.2 + 1);
                emit(&format!(
                    "    🧾 [TABLE UNION] '{}' | 표 밴드 r{}~{}, c{}~{} ({}패치) 로 통합",
                    hm.category, tb.0, tb.1, tb.2, tb.3, area
                ));
                gboxes.push(tb);
                peaks.push(comps[0].peak);
                counts.push(area);
            }
        }

        if gboxes.is_empty() {
            for comp in comps.iter() {
                // ④ 라벨↔값 행 밴드 확장
                let expanded = expand_row_band(comp, &content, content_gate, cols);
                if expanded.2 < comp.c_min || expanded.3 > comp.c_max {
                    emit(&format!(
                        "    ↔️ [ROW BAND] '{}' | c{}~{} → c{}~{} (같은 행 밴드의 값 셀 편입)",
                        hm.category, comp.c_min, comp.c_max, expanded.2, expanded.3
                    ));
                }
                gboxes.push(expanded);
                peaks.push(comp.peak);
                counts.push(comp.indices.len());
            }
        }

        emit(&format!(
            "    🧩 [COMPONENTS] '{}' | 영역 {}개 | Gate: {:+.4} | Top: {}({:+.4})",
            hm.category,
            gboxes.len(),
            gate,
            if hm.top_field.is_empty() { "-" } else { &hm.top_field },
            hm.top_score
        ));

        per_cat.push((
            hm.category.clone(),
            hm.top_field.clone(),
            gboxes,
            peaks,
            counts,
        ));
    }

    if per_cat.is_empty() {
        return Vec::new();
    }

    // ── ③ 성분 풀 구축 (IoU dedup) ──
    let mut pool = ComponentPool {
        boxes: Vec::new(),
        counts: Vec::new(),
    };
    let mut cat_pool_scores: Vec<Vec<(usize, f32)>> = Vec::with_capacity(per_cat.len());

    for (_, _, gboxes, peaks, counts) in per_cat.iter() {
        let mut mine: Vec<(usize, f32)> = Vec::new();
        for (gi, gbox) in gboxes.iter().enumerate() {
            let mut pool_idx: Option<usize> = None;
            for (pi, existing) in pool.boxes.iter().enumerate() {
                if grid_iou(*gbox, *existing) >= 0.5 {
                    pool_idx = Some(pi);
                    break;
                }
            }

            let pi = match pool_idx {
                Some(v) => v,
                None => {
                    pool.boxes.push(*gbox);
                    pool.counts.push(counts[gi]);
                    pool.boxes.len() - 1
                }
            };

            let peak = peaks[gi];
            if let Some(slot) = mine.iter_mut().find(|(i, _)| *i == pi) {
                if peak > slot.1 {
                    slot.1 = peak;
                }
            } else {
                mine.push((pi, peak));
            }
        }
        cat_pool_scores.push(mine);
    }

    let pool_n = pool.boxes.len();
    if pool_n == 0 {
        return Vec::new();
    }

    emit(&format!(
        "    🎯 [REGION POOL] 후보 영역 {}개 | 경쟁 카테고리 {}개 | 면적 상한 {}패치",
        pool_n,
        per_cat.len(),
        area_cap
    ));

    // ── ④ 배타 배정 ──
    let mut matrix: Vec<Vec<f32>> = vec![vec![-1.0f32; pool_n]; per_cat.len()];
    for (ci, mine) in cat_pool_scores.iter().enumerate() {
        for (pi, score) in mine.iter() {
            matrix[ci][*pi] = *score;
        }
    }

    let assign = exclusive_assign_by_score(&matrix, 0.0, 0.0);

    // ── ⑤ 픽셀 박스 확정 ──
    let min_w = ((grid.orig_width as f32 * 0.12) as u32).max(64);
    let min_h = ((grid.orig_height as f32 * 0.06) as u32).max(48);

    let mut plans: Vec<CropPlan> = Vec::new();

    for (ci, a) in assign.iter().enumerate() {
        // 🌟 [STARVATION RESCUE] 배정에 실패한 카테고리는 자기 히트맵의 최고 봉우리로
        //    독립 영역을 만들어 구제합니다.
        //    ── 왜 겹침을 허용하는가 ──
        //     실측에서 cargo 가 전면을 선점하자 header 가 UNASSIGNED 되었고,
        //     doc_number 를 잃어 task_id 폴백 → 재스캔마다 중복 문서가 쌓였습니다.
        //     크롭이 겹치는 비용은 LLM 호출 1회지만, 필드를 통째로 잃는 비용은 영구적입니다.
        let (gbox, score, margin, patch_count) = match a {
            Some((pi, score, margin)) => (pool.boxes[*pi], *score, *margin, pool.counts[*pi]),
            None => {
                let (_, _, gboxes, peaks, counts) = &per_cat[ci];
                let best = peaks
                    .iter()
                    .enumerate()
                    .max_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal));
                match best {
                    // 🌟 [RESCUE GATE] 자기 최고 봉우리가 극값 기대치(0)를 넘을 때만 구제합니다.
                    //
                    //  ── 왜 게이트가 필요해졌나 ──
                    //   카테고리가 8개에서 16개로 늘면서(customs / inspection / insurance /
                    //   hazmat / origin / compliance / settlement / charges),
                    //   그 문서에 아예 존재하지 않는 카테고리가 다수 생깁니다.
                    //   무조건 구제하면 CI 인보이스에 hazmat 크롭을 보내게 되고,
                    //   2B 모델은 빈 영역을 받아도 무언가를 반드시 채우려 듭니다.
                    //   (실측: 빈 BUYER 박스에서 "BUYER (IF NOT CONSIGNEE)" 를 값으로 읽음)
                    //   Qwen 호출 낭비이자 할루시네이션 유입구입니다.
                    //
                    //  ── 0 이 매직 상수가 아닌 이유 ──
                    //   surprisal = (max-μ)/σ - √(2 ln N) 이므로
                    //   0 은 'N개를 무작위로 뽑은 기대 최댓값과 같다' 는 뜻입니다.
                    //   즉 근거가 없다는 것을 극값이론이 정의한 지점입니다.
                    Some((bi, bscore)) if *bscore > 0.0 => {
                        let rescue_cat = per_cat[ci].0.clone();
                        let is_table_cat = rescue_cat == "items" || rescue_cat == "containers";
                        let cand_px = to_pixel_bbox(gboxes[bi], grid);
                        let dup = plans.iter().any(|p| {
                            let (ax0, ay0, ax1, ay1) = cand_px;
                            let (bx0, by0, bx1, by1) = p.bbox;
                            let ix0 = ax0.max(bx0);
                            let iy0 = ay0.max(by0);
                            let ix1 = ax1.min(bx1);
                            let iy1 = ay1.min(by1);
                            if ix0 >= ix1 || iy0 >= iy1 { return false; }
                            let inter = (ix1 - ix0) as f32 * (iy1 - iy0) as f32;
                            let aa = ((ax1 - ax0) as f32 * (ay1 - ay0) as f32).max(1.0);
                            let bb = ((bx1 - bx0) as f32 * (by1 - by0) as f32).max(1.0);
                            (inter / aa.min(bb)) > 0.5
                        });
                        if dup && !is_table_cat {
                            emit(&format!(
                                "    ⚪ [RESCUE SKIP] '{}' 의 최고 봉우리 영역이 이미 배정된 크롭과 절반 이상 겹칩니다. 빈 영역 할루시네이션 유입을 막기 위해 구제하지 않습니다.",
                                rescue_cat
                            ));
                            continue;
                        }
                        emit(&format!(
                            "    🛟 [STARVATION RESCUE] '{}' 는 영역을 선점당했지만 자기 최고 봉우리({:+.4})로 독립 크롭합니다.",
                            per_cat[ci].0, bscore
                        ));
                        (gboxes[bi], *bscore, 0.0f32, counts[bi])
                    }

                    Some((_, bscore)) => {
                        emit(&format!(
                            "    ⚪ [NOT PRESENT] '{}' 는 최고 봉우리가 {:+.4} 로 기대치 이하입니다. 이 문서에 없는 축이므로 크롭하지 않습니다.",
                            per_cat[ci].0, bscore
                        ));
                        continue;
                    }
                    None => {
                        emit(&format!(
                            "    ⚪ [UNASSIGNED] '{}' 는 후보 영역이 하나도 없어 크롭하지 않습니다.",
                            per_cat[ci].0
                        ));
                        continue;
                    }
                }
            }
        };

        let raw = to_pixel_bbox(gbox, grid);
        let bbox = ensure_min_size(raw, grid.orig_width, grid.orig_height, min_w, min_h);

        // 🌟 [LOG] 크롭이 히트맵 활성 패치를 얼마나 커버하는지 계산
        {
            let hm_opt = heatmaps.iter().find(|h| h.category == per_cat[ci].0);
            if let Some(hm) = hm_opt {
                let mut total_hot = 0usize;
                let mut covered_hot = 0usize;
                let cw = grid.orig_width as f32 / grid.grid_cols as f32;
                let ch = grid.orig_height as f32 / grid.grid_rows as f32;
                for idx in 0..hm.scores.len() {
                    if hm.scores[idx] <= 0.0 { continue; }
                    total_hot += 1;
                    let r = idx / grid.grid_cols;
                    let c = idx % grid.grid_cols;
                    let cx = (c as f32 + 0.5) * cw;
                    let cy = (r as f32 + 0.5) * ch;
                    if cx >= bbox.0 as f32 && cx <= bbox.2 as f32
                        && cy >= bbox.1 as f32 && cy <= bbox.3 as f32
                    {
                        covered_hot += 1;
                    }
                }
                let coverage = if total_hot > 0 {
                    covered_hot as f32 / total_hot as f32
                } else {
                    1.0
                };
                emit(&format!(
                    "    📊 [CROP COVERAGE] '{}' 히트맵 활성 {}개 중 {}개 커버 ({:.0}%)",
                    per_cat[ci].0, total_hot, covered_hot, coverage * 100.0
                ));
                if coverage < 0.70 && total_hot > 3 {
                    emit(&format!(
                        "    ⚠️ [CROP COVERAGE LOSS] '{}' 활성 패치의 {:.0}%가 크롭 밖 — 잘림 발생. 히트맵 확산 또는 크롭 마진 부족 가능.",
                        per_cat[ci].0, (1.0 - coverage) * 100.0
                    ));
                }
            }
        }

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
            patch_count,
            top_field: per_cat[ci].1.clone(),
        });
    }

    // ── ⑥ 문서 기본키 밴드 보장 ──
    //    doc_number 를 잃으면 릴레이가 영구히 끊기므로 header 만은 무조건 보장합니다.
    ensure_identity_band_crop(&mut plans, heatmaps, &content, content_gate, grid, emit);

    // ── ⑦ 미커버 내용 밴드 구제 ──
    //    "어떤 크롭에도 안 담긴 잉크" 는 회수 불가능한 손실입니다.
    rescue_uncovered_bands(&mut plans, heatmaps, &content, content_gate, grid, emit);

    // ── ⑦-B 동일 카테고리 겹침 크롭 병합 ──
    //
    //  ── 실측 사고 ──
    //   header 크롭이 두 개 생성되었습니다.
    //     [A] grid(r2~4, c4~13) → px(114,57)-(800,344)   ← 좌측 c0~c3 이 잘려 있음
    //     [B] IDENTITY BAND     → px(0,114)-(800,516)    ← 좌측 포함, 정답이 여기 있음
    //   순서를 바꿔도 두 크롭이 각각 LLM 을 타면서
    //   먼저 발언한 쪽이 [ALREADY CLAIMED] 로 뒤쪽을 막습니다.
    //   실측 로그:
    //     [A] → doc_number "93763111837" (AIRWAYBILL 번호)  ← 먼저 확정
    //     [B] → doc_number "CI-43726"    (정답)            ← '기존 유지' 로 폐기
    //   그 잘못된 번호가 그대로 entity_index 와 릴레이 25건의 키가 되었습니다.
    //
    //  ── 왜 순서가 아니라 병합인가 ──
    //   두 크롭은 같은 논리 블록(문서 식별 밴드)의 좌·우 조각입니다.
    //   조각을 순서대로 보여주면 각 조각이 자기 안에서만 답을 찾으므로,
    //   좌측 조각만 본 호출은 INVOICE NUMBER 를 볼 기회 자체가 없습니다.
    //   합집합으로 한 번에 보여주면 두 열이 같은 시야에 들어와
    //   모델이 라벨과 값을 좌우로 대조할 수 있습니다. LLM 호출도 하나 줄어듭니다.
    //
    //  ── 면적 상한 ──
    //   무제한 병합은 카테고리 하나가 페이지를 삼키는 역효과를 냅니다.
    //   합집합이 페이지의 60% 를 넘으면 병합하지 않고 원래대로 둡니다.
    //   (60% 는 '한 카테고리가 문서 과반을 넘게 차지할 수 없다' 는 구조적 상한이며,
    //    plan_crops 앞부분의 area_cap 과 같은 성격의 판정입니다)
    {
        let page_area = (grid.orig_width as f32) * (grid.orig_height as f32);
        let mut i = 0usize;
        while i < plans.len() {
            let mut j = i + 1;
            while j < plans.len() {
                if plans[i].category != plans[j].category {
                    j += 1;
                    continue;
                }
                let a = plans[i].bbox;
                let b = plans[j].bbox;
                let overlaps = a.0 < b.2 && b.0 < a.2 && a.1 < b.3 && b.1 < a.3;
                if !overlaps {
                    j += 1;
                    continue;
                }
                let merged = (
                    a.0.min(b.0),
                    a.1.min(b.1),
                    a.2.max(b.2),
                    a.3.max(b.3),
                );
                let area = ((merged.2 - merged.0) as f32) * ((merged.3 - merged.1) as f32);
                if area > page_area * 0.6 {
                    emit(&format!(
                        "    ⚪ [MERGE SKIP] '{}' 의 두 크롭을 합치면 페이지의 {:.0}% 를 차지해 병합하지 않습니다.",
                        plans[i].category,
                        area / page_area * 100.0
                    ));
                    j += 1;
                    continue;
                }
                emit(&format!(
                    "    🔗 [CROP MERGE] '{}' 의 겹치는 크롭 2개를 합칩니다. px({},{})-({},{}) + px({},{})-({},{}) → px({},{})-({},{})",
                    plans[i].category,
                    a.0, a.1, a.2, a.3,
                    b.0, b.1, b.2, b.3,
                    merged.0, merged.1, merged.2, merged.3
                ));
                plans[i].bbox = merged;
                plans[i].patch_count += plans[j].patch_count;
                if plans[j].score > plans[i].score {
                    plans[i].score = plans[j].score;
                    plans[i].top_field = plans[j].top_field.clone();
                }
                plans.remove(j);
            }
            i += 1;
        }
    }

    // ── ⑧ 식별 크롭 우선 정렬 ──
    //
    //  ── 실측 사고 ──
    //   header 크롭이 두 개 생성되었습니다.
    //     [1] grid(r2~4, c4~13) → px(114,57)-(800,344)   ← 좌측 c0~c3 이 잘려 있음
    //     [2] IDENTITY BAND     → px(0,114)-(800,516)    ← 좌측 포함, 정답이 여기 있음
    //   그런데 [2] 는 ⑥ 단계에서 벡터 맨 뒤에 push 되어 10번째로 처리되었습니다.
    //   STAGE-5 는 plans 순서대로 돌면서 [ALREADY CLAIMED] 로 앞선 확정값을 보호하므로,
    //     [1] 이 우측 AIRWAYBILL 번호 93763111837 을 doc_number 로 먼저 확정했고
    //     [2] 가 뒤늦게 읽어낸 정답 CI-43726 은 '기존 유지' 로 폐기되었습니다.
    //   그 잘못된 번호가 그대로 entity_index / 릴레이 25건의 키가 되었습니다.
    //
    //  ── 왜 순서만 바꾸면 되는가 ──
    //   두 크롭 모두 이미 계획에 들어 있고 둘 다 LLM 을 탑니다. 비용은 동일합니다.
    //   달라지는 것은 '누가 먼저 기본키를 확정하는가' 뿐입니다.
    //   문서 기본키는 되돌릴 수 없는 축이므로, 그것을 담당하는 크롭이 먼저 발언해야 합니다.
    //
    //  ── 판정 근거 ──
    //   top_field 가 "doc_number" 인 header 크롭이 식별 크롭입니다.
    //   ensure_identity_band_crop 이 그 값을 명시적으로 심어 두므로 어휘 판정이 아닙니다.
    //   sort_by_key 는 안정 정렬이라 나머지 크롭의 상대 순서는 그대로 보존됩니다.
    {
        let before: Vec<String> = plans.iter().map(|p| p.category.clone()).collect();
        plans.sort_by_key(|p| {
            if p.category == "header" && p.top_field == "doc_number" {
                0u8
            } else if p.category == "header" {
                1u8
            } else {
                2u8
            }
        });
        let after: Vec<String> = plans.iter().map(|p| p.category.clone()).collect();
        if before != after {
            emit(&format!(
                "    🥇 [IDENTITY FIRST ORDER] 문서 기본키 크롭을 선두로 재정렬했습니다. {:?} → {:?}",
                before, after
            ));
        }
    }

    plans
}

/// 🌟 [TEXT-AWARE UPSCALE] 확대 배율을 '글자 높이 → 비전 패치 1개' 기준으로 정합니다.
///
///  ── 왜 짧은 변 기준이 틀렸나 ──
///   현재 규칙은 크롭의 기하만 보고 글자 크기를 전혀 모릅니다.
///   가로로 긴 띠(515×86)는 6배까지 확대되어 글자가 2.5패치가 되고,
///   토큰만 4배 쓰면서 판독성은 그대로입니다.
///   실측: insurance 타일 2048×512, 판독가능 패치 0/10 → 값 창작
///
///  ── 배율 근거 ──
///   Qwen VL 은 14px 패치를 2×2 병합해 토큰 1개당 28×28px 를 봅니다.
///   글자 높이가 28px 에 도달하면 '한 글자 ≈ 한 토큰' 이 되고,
///   그 이상은 토큰만 늘고 판독성은 포화합니다.
///   행 투영 프로파일의 자기상관 주기가 곧 텍스트 라인 피치입니다.
///   (고전 문서영상 기법이며 어휘 사전이 아닙니다)
const VISION_PATCH_PX: f32 = 28.0;

fn estimate_text_height(img: &DynamicImage) -> Option<f32> {
    use image::GenericImageView;
    let g = img.to_luma8();
    let (w, h) = g.dimensions();
    if h < 16 || w < 16 { return None; }

    // ── 행별 잉크 밀도 프로파일 ──
    let mut prof: Vec<f32> = Vec::with_capacity(h as usize);
    for y in 0..h {
        let mut s = 0.0f32;
        for x in 0..w {
            s += 255.0 - g.get_pixel(x, y)[0] as f32;
        }
        prof.push(s / w as f32);
    }
    let mean: f32 = prof.iter().sum::<f32>() / prof.len() as f32;
    for v in prof.iter_mut() { *v -= mean; }

    // ── 자기상관 최대 주기 = 텍스트 라인 피치 ──
    let max_lag = ((h / 2) as usize).min(120);
    let mut best_lag = 0usize;
    let mut best = f32::MIN;
    for lag in 4..max_lag {
        let mut s = 0.0f32;
        for i in 0..(prof.len() - lag) {
            s += prof[i] * prof[i + lag];
        }
        let norm = s / (prof.len() - lag) as f32;
        if norm > best { best = norm; best_lag = lag; }
    }
    if best_lag == 0 || best <= 0.0 { return None; }
    // 라인 피치의 약 60% 가 실제 글자 높이(x-height + 어센더)
    Some(best_lag as f32 * 0.6)
}

pub fn crop_region(
    image: &DynamicImage,
    plan: &CropPlan,
    target_short: u32,
) -> DynamicImage {
    let (x0, y0, x1, y1) = plan.bbox;
    let w = x1.saturating_sub(x0).max(1);
    let h = y1.saturating_sub(y0).max(1);
    let cropped = image.crop_imm(x0, y0, w, h);

    // 🌟 글자 높이를 실측해 필요한 배율만 적용합니다.
    let factor = match estimate_text_height(&cropped) {
        Some(th) if th > 0.5 => {
            let f = (VISION_PATCH_PX / th).clamp(1.0, 4.0);
            println!(
                "    📏 [TEXT-AWARE UPSCALE] 추정 글자 높이 {:.1}px → 배율 {:.2}x (목표 {}px/글자)",
                th, f, VISION_PATCH_PX as u32
            );
            f
        }
        _ => {
            // 프로파일에서 주기를 못 찾음 = 텍스트가 거의 없음.
            // 기존 짧은 변 규칙으로 폴백하되 상한을 낮게 둡니다.
            let short = w.min(h) as f32;
            let f = (target_short as f32 / short).clamp(1.0, 2.0);
            println!(
                "    📏 [TEXT-AWARE UPSCALE] 라인 주기 미검출(텍스트 희소) → 보수적 배율 {:.2}x",
                f
            );
            f
        }
    };

    if factor <= 1.01 { return cropped; }

    let nw = (((w as f32 * factor).round() as u32).max(1)).min(2048);
    let nh = (((h as f32 * factor).round() as u32).max(1)).min(2048);
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

/// 🌟 [CROP AUDIT] 확정된 크롭이 정말 그 카테고리의 영역인지 재검증합니다.
///
///  ── 실측 사고 ──
///   logistics ← px(114,0)-(800,287)   = INVOICE NUMBER / AIRWAYBILL / DATE 영역
///   header    ← px(0,630)-(800,1032)  = 선언문 / 서명 / SIGNATORY 영역
///   두 카테고리가 서로의 영역을 통째로 바꿔 가졌습니다.
///   그 결과 header 는 doc_number 를 못 찾아 task_id 폴백이 확정되었고,
///   logistics 는 CI-43726 의 뒤 두 자리를 voyage_number 로 지어냈습니다.
///
///  ── 판정 원리 ──
///   이 카테고리 히트맵의 최댓값을 '크롭 내부' 와 '크롭 외부' 로 나눠
///   극값 기대치를 차감한 뒤 비교합니다.
///     surprisal_in  = (max_in  - μ)/σ - √(2 ln N_in)
///     surprisal_out = (max_out - μ)/σ - √(2 ln N_out)
///   외부가 이기면 그 크롭은 이 카테고리의 근거지가 아닙니다.
///
///  ── 교정 ──
///   서로를 이긴 두 카테고리가 있으면 크롭을 맞바꿉니다(swap).
///   맞바꿀 짝이 없으면 그 카테고리의 자체 최고 봉우리 영역으로 재배정합니다.
///   판정은 이미 계산된 히트맵만 사용하므로 추가 추론 비용이 0 입니다.
pub fn audit_crops(
    plans: &mut Vec<CropPlan>,
    heatmaps: &[CategoryHeatmap],
    grid: &PatchGrid,
    legibility: &crate::models::siglip2::legibility::LegibilityMap,
    emit: &dyn Fn(&str),
) {
    use crate::utils::ai_utils::gumbel_expected_z;

    if plans.is_empty() {
        return;
    }
    let rows = grid.grid_rows;
    let cols = grid.grid_cols;
    let n = rows * cols;

    let in_bbox = |idx: usize, bbox: (u32, u32, u32, u32)| -> bool {
        let r = idx / cols;
        let c = idx % cols;
        let cw = grid.orig_width as f32 / cols as f32;
        let ch = grid.orig_height as f32 / rows as f32;
        let cx = (c as f32 + 0.5) * cw;
        let cy = (r as f32 + 0.5) * ch;
        cx >= bbox.0 as f32 && cx <= bbox.2 as f32
            && cy >= bbox.1 as f32 && cy <= bbox.3 as f32
    };

    // (카테고리, bbox) → (surprisal_in, surprisal_out)
    let score_pair = |cat: &str, bbox: (u32, u32, u32, u32)| -> (f32, f32) {
        let hm = match heatmaps.iter().find(|h| h.category == cat) {
            Some(h) => h,
            None => return (0.0, 0.0),
        };
        let m = n.min(hm.scores.len());
        if m == 0 {
            return (0.0, 0.0);
        }
        let mean: f32 = hm.scores[..m].iter().sum::<f32>() / m as f32;
        let var: f32 = hm.scores[..m]
            .iter()
            .map(|s| (s - mean) * (s - mean))
            .sum::<f32>()
            / m as f32;
        let std = var.sqrt().max(1e-6);

        let (mut mx_in, mut mx_out) = (f32::MIN, f32::MIN);
        let (mut n_in, mut n_out) = (0usize, 0usize);
        for i in 0..m {
            // 판독 불가 패치는 근거가 될 수 없습니다.
            if !legibility.is_legible(i) {
                continue;
            }
            if in_bbox(i, bbox) {
                n_in += 1;
                if hm.scores[i] > mx_in { mx_in = hm.scores[i]; }
            } else {
                n_out += 1;
                if hm.scores[i] > mx_out { mx_out = hm.scores[i]; }
            }
        }
        let s_in = if n_in == 0 { f32::MIN }
            else { (mx_in - mean) / std - gumbel_expected_z(n_in) };
        let s_out = if n_out == 0 { f32::MIN }
            else { (mx_out - mean) / std - gumbel_expected_z(n_out) };
        (s_in, s_out)
    };

    // ── ① 자기 크롭에서 근거를 잃은 카테고리 수집 ──
    let mut suspects: Vec<usize> = Vec::new();
    for (pi, p) in plans.iter().enumerate() {
        let (s_in, s_out) = score_pair(&p.category, p.bbox);
        if s_out > s_in {
            emit(&format!(
                "    🔍 [CROP AUDIT] '{}' 의심 | in {:+.4} < out {:+.4} — 크롭 밖에 더 강한 근거가 있습니다.",
                p.category, s_in, s_out
            ));
            suspects.push(pi);
        }
    }
    if suspects.is_empty() {
        emit("    ✅ [CROP AUDIT] 전 크롭이 자기 카테고리의 최강 근거지를 점유하고 있습니다.");
        return;
    }

    // ── ② 상호 교환 후보 탐색 ──
    //    A 가 B 의 bbox 에서, B 가 A 의 bbox 에서 각각 더 높은 점수를 받으면 맞바꿉니다.
    let mut swapped: Vec<bool> = vec![false; plans.len()];
    for ai in 0..plans.len() {
        if swapped[ai] { continue; }
        for bi in (ai + 1)..plans.len() {
            if swapped[bi] { continue; }

            let a_here = score_pair(&plans[ai].category, plans[ai].bbox).0;
            let a_there = score_pair(&plans[ai].category, plans[bi].bbox).0;
            let b_here = score_pair(&plans[bi].category, plans[bi].bbox).0;
            let b_there = score_pair(&plans[bi].category, plans[ai].bbox).0;

            if a_there > a_here && b_there > b_here {
                emit(&format!(
                    "    🔁 [CROP SWAP] '{}' ↔ '{}' | {}: {:+.4}→{:+.4} | {}: {:+.4}→{:+.4}",
                    plans[ai].category, plans[bi].category,
                    plans[ai].category, a_here, a_there,
                    plans[bi].category, b_here, b_there
                ));
                let tmp_box = plans[ai].bbox;
                let tmp_cnt = plans[ai].patch_count;
                plans[ai].bbox = plans[bi].bbox;
                plans[ai].patch_count = plans[bi].patch_count;
                plans[ai].score = a_there;
                plans[bi].bbox = tmp_box;
                plans[bi].patch_count = tmp_cnt;
                plans[bi].score = b_there;
                swapped[ai] = true;
                swapped[bi] = true;
                break;
            }
        }
    }

    // ── ③ 짝이 없는 의심 크롭은 자체 최고 봉우리로 재배정 ──
    for &pi in suspects.iter() {
        if swapped[pi] { continue; }
        let cat = plans[pi].category.clone();
        let hm = match heatmaps.iter().find(|h| h.category == cat) {
            Some(h) => h,
            None => continue,
        };
        let m = n.min(hm.scores.len());
        let mut best = f32::MIN;
        let mut best_i = usize::MAX;
        for i in 0..m {
            if !legibility.is_legible(i) { continue; }
            if hm.scores[i] > best { best = hm.scores[i]; best_i = i; }
        }
        if best_i == usize::MAX { continue; }

        let r = best_i / cols;
        let c = best_i % cols;
        let mx = ((cols as f32 * 0.15) as usize).max(2);
        let my = ((rows as f32 * 0.06) as usize).max(1);
        let gbox = (
            r.saturating_sub(my),
            (r + my).min(rows - 1),
            c.saturating_sub(mx),
            (c + mx).min(cols - 1),
        );
        let raw = to_pixel_bbox(gbox, grid);
        let min_w = ((grid.orig_width as f32 * 0.12) as u32).max(64);
        let min_h = ((grid.orig_height as f32 * 0.06) as u32).max(48);
        let bbox = ensure_min_size(raw, grid.orig_width, grid.orig_height, min_w, min_h);

        emit(&format!(
            "    🎯 [CROP RELOCATE] '{}' → grid(r{}~{}, c{}~{}) px({},{})-({},{}) | 자체 최고 봉우리 {:+.4}",
            cat, gbox.0, gbox.1, gbox.2, gbox.3,
            bbox.0, bbox.1, bbox.2, bbox.3, best
        ));
        plans[pi].bbox = bbox;
        plans[pi].score = best;
    }
}

/// 🌟 [TILE PLAN] 겹치는 타일 분할.
///
///  ── 언제 발화하는가 (무지성 분할 금지) ──
///   T1 잘림 위험 : 크롭 bbox 밖에 그 카테고리의 활성 패치가 남아 있고,
///                 그 잔여의 surprisal 이 0 을 넘을 때. (근거가 잘려 나갔다는 뜻)
///   T2 배열 밀도 : items / containers 에서 '표 행' 으로 판정된 격자 행이
///                 2줄을 넘을 때. 한 번의 호출로 여러 행을 다 읽어내기 어렵습니다.
///   T3 해상도    : 실제 내용(판독가능 패치)이 크롭 면적의 소수에 불과할 때.
///                 실측 items 크롭은 800x746 인데 표는 84px(11%)뿐이라
///                 다운스케일 후 글자가 뭉개져 2행 중 1행만 읽혔습니다.
///
///  ── 겹침 비율 ──
///   사용자 요구대로 20~30% 를 씁니다. 값 자체는 '표 한 행이 두 타일에 걸쳐도
///   최소 한쪽에는 온전히 들어간다' 는 구조적 요구에서 나옵니다.
///   행 높이가 타일 높이의 25% 이하이면 겹침 25% 가 그 조건을 보장합니다.
///
///  ── 병합 ──
///   타일별 추출 결과는 호출부가 dedupe 합니다. (Part 18 참조)
#[derive(Debug, Clone)]
pub struct TilePlan {
    pub bbox: (u32, u32, u32, u32),
    pub index: usize,
    pub total: usize,
}

/// 세로 방향 겹침 분할. 무역 서식의 표는 가로로 넓고 세로로 쌓이므로
/// 세로 분할이 행 손실을 최소화합니다.
pub fn plan_overlap_tiles(
    bbox: (u32, u32, u32, u32),
    tile_count: usize,
    overlap_ratio: f32,
) -> Vec<TilePlan> {
    let (x0, y0, x1, y1) = bbox;
    if tile_count <= 1 || y1 <= y0 {
        return vec![TilePlan { bbox, index: 0, total: 1 }];
    }
    let h = (y1 - y0) as f32;
    // t = 타일 높이. n 타일이 겹침 r 로 전체를 덮으려면
    //   h = n*t - (n-1)*r*t  →  t = h / (n - (n-1)*r)
    let n = tile_count as f32;
    let denom = n - (n - 1.0) * overlap_ratio;
    if denom <= 0.0 {
        return vec![TilePlan { bbox, index: 0, total: 1 }];
    }
    let t = h / denom;
    let step = t * (1.0 - overlap_ratio);

    let mut out = Vec::with_capacity(tile_count);
    for i in 0..tile_count {
        let ty0 = y0 as f32 + step * i as f32;
        let ty1 = (ty0 + t).min(y1 as f32);
        if ty1 <= ty0 + 1.0 {
            continue;
        }
        out.push(TilePlan {
            bbox: (x0, ty0 as u32, x1, ty1 as u32),
            index: i,
            total: tile_count,
        });
    }
    if out.is_empty() {
        out.push(TilePlan { bbox, index: 0, total: 1 });
    }
    let total = out.len();
    for p in out.iter_mut() {
        p.total = total;
    }
    out
}

/// 🌟 [TILE DECISION] 이 크롭을 몇 개로 쪼갤지 '점수' 로 결정합니다.
///
///  반환 (타일 수, 사유). 1 이면 분할하지 않습니다.
pub fn decide_tile_count(
    plan: &CropPlan,
    heatmaps: &[CategoryHeatmap],
    grid: &PatchGrid,
    legibility: &crate::models::siglip2::legibility::LegibilityMap,
    emit: &dyn Fn(&str),
) -> (usize, String) {
    use crate::utils::ai_utils::gumbel_expected_z;

    let rows = grid.grid_rows;
    let cols = grid.grid_cols;
    let n = rows * cols;
    let cw = grid.orig_width as f32 / cols as f32;
    let ch = grid.orig_height as f32 / rows as f32;

    let inside = |i: usize| -> bool {
        let r = i / cols;
        let c = i % cols;
        let cx = (c as f32 + 0.5) * cw;
        let cy = (r as f32 + 0.5) * ch;
        cx >= plan.bbox.0 as f32 && cx <= plan.bbox.2 as f32
            && cy >= plan.bbox.1 as f32 && cy <= plan.bbox.3 as f32
    };

    // ── T1 : 잘림 위험 ──
    let mut t1 = false;
    if let Some(hm) = heatmaps.iter().find(|h| h.category == plan.category) {
        let m = n.min(hm.scores.len());
        if m > 0 {
            let mean: f32 = hm.scores[..m].iter().sum::<f32>() / m as f32;
            let var: f32 = hm.scores[..m].iter()
                .map(|s| (s - mean) * (s - mean)).sum::<f32>() / m as f32;
            let std = var.sqrt().max(1e-6);
            let (mut mx_out, mut n_out) = (f32::MIN, 0usize);
            for i in 0..m {
                if inside(i) || !legibility.is_legible(i) { continue; }
                n_out += 1;
                if hm.scores[i] > mx_out { mx_out = hm.scores[i]; }
            }
            if n_out > 0 {
                let s_out = (mx_out - mean) / std - gumbel_expected_z(n_out);
                if s_out > 0.0 { t1 = true; }
            }
        }
    }

    // ── T2 : 표 행 밀도 (배열 카테고리 전용) ──
    let is_array_cat = plan.category == "items" || plan.category == "containers";
    let dense_cols = (cols / 3).max(2);
    let mut table_rows = 0usize;
    if is_array_cat {
        for r in 0..rows {
            let cnt = (0..cols)
                .filter(|&c| {
                    let i = r * cols + c;
                    inside(i) && legibility.is_legible(i)
                })
                .count();
            if cnt >= dense_cols { table_rows += 1; }
        }
    }
    let t2 = is_array_cat && table_rows > 2;

    // ── T3 : 내용 희소 (해상도 손실) ──
    let (lg, _il, _bl) =
        legibility.count_in_bbox(plan.bbox, grid.orig_width, grid.orig_height);
    let total_in = (0..n).filter(|&i| inside(i)).count().max(1);
    let t3 = lg * 4 < total_in;

    if !t1 && !t2 && !t3 {
        return (1, String::new());
    }

    // ── [FIX-A] 표 카테고리: 판독가능 패치가 극소면 분할 무의미 ──
    //   표 행은 감지되었지만 읽을 텍스트가 거의 없으면
    //   타일을 나눠도 각 타일이 빈 영역만 받아 전부 null을 반환합니다.
    //   (실측: containers 판독가능 64/182 → 4타일 전부 null)
    //   판독가능 비율이 10% 미만이면 1타일로 축소합니다.
    if t2 && lg * 10 < total_in {
        emit(&format!(
            "    ⚪ [TILE SKIP / SPARSE TABLE] '{}' 표행 {}개 감지되었으나 \
             판독가능 패치 {}/{} ({:.0}%) 로 분할 무의미. 1타일로 축소합니다.",
            plan.category, table_rows, lg, total_in,
            lg as f32 / total_in as f32 * 100.0
        ));
        return (1, "표행감지_판독불가".to_string());
    }

    // ── [FIX-B] 비표 카테고리: 잘림위험도 내용희소도 아니면 1타일 ──
    //   기존: 사유 하나만 걸려도 무조건 2타일
    //   변경: T1(잘림) 또는 T3(희소) 중 실제 분할이 필요한 경우만 2타일
    //         내용희소(T3)만 걸린 경우: 크롭이 이미 좁아서
    //         나눠도 각 타일이 더 좁아질 뿐 → 1타일이 오히려 정확
    let count = if t2 {
        // ── [FIX-C] 표 상한 4 → 3 ──
        //   행 7개 이상이어도 3타일이면 타일당 ~2.3행으로 충분합니다.
        //   4타일이면 타일당 ~1.75행으로 겹침 영역만 늘어납니다.
        ((table_rows + 1) / 2).clamp(2, 3)
    } else if t1 {
        // 잘림 위험: 크롭 밖에 근거가 있으므로 2타일로 상/하 분리
        2
    } else {
        // T3(내용희소)만 남은 경우: 1타일
        // 크롭 자체가 좁으므로 분할하면 오히려 컨텍스트가 끊어집니다.
        1
    };

    let mut why: Vec<&str> = Vec::new();
    if t1 { why.push("잘림위험"); }
    if t2 { why.push("표행밀도"); }
    if t3 { why.push("내용희소"); }
    let reason = why.join("+");

    emit(&format!(
        "    🧱 [TILE PLAN] '{}' → {}타일 (겹침 25%) | 사유: {} | 표행 {} | 판독가능 {}/{}",
        plan.category, count, reason, table_rows, lg, total_in
    ));

    (count, reason)
}
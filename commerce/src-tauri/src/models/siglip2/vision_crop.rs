use image::DynamicImage;

use super::preprocessor::patch_index_to_bbox;
use super::vision_encoder::{CategoryHeatmap, PatchGrid};
use crate::utils::ai_utils::exclusive_assign_by_score;

/// 카테고리 하나에 대한 크롭 계획.
#[derive(Debug, Clone)]
pub struct CropPlan {
    pub category: String,
    pub bbox: (u32, u32, u32, u32),
    pub score: f32,
    pub margin: f32,
    pub patch_count: usize,
    pub top_field: String,
    pub owned_patches: usize,
    pub twin_of: String,
}

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

impl Component {
    fn area(&self) -> usize {
        (self.r_max - self.r_min + 1) * (self.c_max - self.c_min + 1)
    }
}

const COL_BAND_UP_ROWS: usize = 1;
const BAND_FOREIGN_RUN: usize = 2;

const CROP_SNAP_IOU: f32 = 0.85;
const CROP_TWIN_IOU: f32 = 0.90;

const RESCUE_MAX: usize = 5;
const RESCUE_MIN_CELLS: usize = 3;
const RESCUE_SPLIT_PASSES: usize = 4;
const RESCUE_OWNER_MIN_SHARE: f32 = 0.20;

const SPLIT_COVERAGE_FLOOR: f32 = 0.70;
const SPLIT_CROP_LIMIT: usize = 2;
const SPLIT_MIN_HOT: usize = 8;
const SPLIT_MIN_GAIN: usize = 4;

const CROP_MERGE_IOU: f32 = 0.25;
const MERGE_FILL_DROP: f32 = 0.65;
const MERGE_PAGE_RATIO: f32 = 0.55;

fn px_iou(a: (u32, u32, u32, u32), b: (u32, u32, u32, u32)) -> f32 {
    let x0 = a.0.max(b.0);
    let y0 = a.1.max(b.1);
    let x1 = a.2.min(b.2);
    let y1 = a.3.min(b.3);
    if x0 >= x1 || y0 >= y1 {
        return 0.0;
    }
    let inter = (x1 - x0) as f32 * (y1 - y0) as f32;
    let aa = (a.2.saturating_sub(a.0) as f32 * a.3.saturating_sub(a.1) as f32).max(1.0);
    let bb = (b.2.saturating_sub(b.0) as f32 * b.3.saturating_sub(b.1) as f32).max(1.0);
    let uni = aa + bb - inter;
    if uni <= 0.0 { 0.0 } else { inter / uni }
}

fn px_covers(outer: (u32, u32, u32, u32), inner: (u32, u32, u32, u32)) -> bool {
    outer.0 <= inner.0 && outer.1 <= inner.1 && outer.2 >= inner.2 && outer.3 >= inner.3
}

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

fn split_component_quantile(
    comp: &Component,
    field: &[f32],
    rows: usize,
    cols: usize,
    area_cap: usize,
) -> Vec<Component> {
    let n = rows * cols;
    let mut local = vec![-1.0f32; n];
    let mut vals: Vec<f32> = Vec::with_capacity(comp.indices.len());
    for &i in comp.indices.iter() {
        if i < n && i < field.len() {
            local[i] = field[i];
            vals.push(field[i]);
        }
    }
    if vals.len() < 2 {
        return vec![comp.clone()];
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut best: Vec<Component> = vec![comp.clone()];
    for k in 1..=4usize {
        let q = (0.35 + 0.15 * k as f32).min(0.95);
        let pos = (((vals.len() - 1) as f32) * q).round() as usize;
        let gate = vals[pos.min(vals.len() - 1)];
        let subs = extract_components(&local, rows, cols, gate);
        if subs.is_empty() {
            continue;
        }
        if subs.len() > best.len() {
            best = subs.clone();
        }
        if subs.len() > 1 && subs.iter().all(|s| s.area() <= area_cap) {
            return subs;
        }
    }
    best
}

fn split_until_fits(
    comps: Vec<Component>,
    field: &[f32],
    rows: usize,
    cols: usize,
    area_cap: usize,
    passes: usize,
) -> (Vec<Component>, usize) {
    let mut cur = comps;
    let mut done = 0usize;
    for _ in 0..passes {
        if !cur.iter().any(|c| c.area() > area_cap) {
            break;
        }
        let mut next: Vec<Component> = Vec::new();
        let mut progressed = false;
        for c in cur.into_iter() {
            if c.area() <= area_cap {
                next.push(c);
                continue;
            }
            let parts = split_component_quantile(&c, field, rows, cols, area_cap);
            if parts.len() > 1 {
                progressed = true;
                next.extend(parts);
            } else {
                next.push(c);
            }
        }
        cur = next;
        done += 1;
        if !progressed {
            break;
        }
    }
    (cur, done)
}

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

fn expand_col_band(
    gbox: (usize, usize, usize, usize),
    content: &[f32],
    gate: f32,
    rows: usize,
    cols: usize,
    max_height: usize,
    owned: Option<&[bool]>,
) -> ((usize, usize, usize, usize), String) {
    let (mut r_min, mut r_max, c_min, c_max) = gbox;
    let cap = if max_height == 0 { rows } else { max_height.max(1) };

    let band_has = |r: usize| -> bool {
        (c_min..=c_max).any(|c| {
            let idx = r * cols + c;
            idx < content.len() && content[idx] > gate
        })
    };
    let band_owned = |r: usize| -> bool {
        match owned {
            None => true,
            Some(map) => (c_min..=c_max).any(|c| {
                let idx = r * cols + c;
                idx < map.len() && map[idx]
            }),
        }
    };

    let mut note = String::new();
    let mut run = 0usize;
    while r_max + 1 < rows && (r_max - r_min + 1) < cap {
        let nr = r_max + 1;
        if !band_has(nr) {
            note = format!("아래 r{} 가 밴드 여백", nr);
            break;
        }
        if band_owned(nr) {
            run = 0;
        } else {
            run += 1;
            if run > BAND_FOREIGN_RUN {
                note = format!("아래 r{} 부터 남의 영토가 {}줄 연속", nr, run);
                break;
            }
        }
        r_max = nr;
    }
    if note.is_empty() && (r_max - r_min + 1) >= cap {
        note = format!("면적 상한(높이 {}행) 도달", cap);
    }

    let mut run = 0usize;
    let mut guaranteed = COL_BAND_UP_ROWS;
    while r_min > 0 && (r_max - r_min + 1) < cap {
        let nr = r_min - 1;
        if !band_has(nr) {
            if note.is_empty() { note = format!("위 r{} 가 밴드 여백", nr); }
            break;
        }
        if band_owned(nr) || guaranteed > 0 {
            run = 0;
        } else {
            run += 1;
            if run > BAND_FOREIGN_RUN {
                if note.is_empty() { note = format!("위 r{} 부터 남의 영토가 {}줄 연속", nr, run); }
                break;
            }
        }
        r_min = nr;
        guaranteed = guaranteed.saturating_sub(1);
    }

    if note.is_empty() {
        note = "격자 끝".to_string();
    }
    ((r_min, r_max, c_min, c_max), note)
}

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

fn merge_adjacent(mut comps: Vec<Component>, cols: usize) -> Vec<Component> {
    if comps.len() <= 1 {
        return comps;
    }

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

fn presence_gate(
    heatmaps: &[CategoryHeatmap],
    n: usize,
    identity_category: &str,
    legibility: &crate::models::siglip2::legibility::LegibilityMap,
    emit: &dyn Fn(&str),
) -> (std::collections::HashSet<String>, Vec<Option<usize>>) {
    use std::collections::{HashMap, HashSet};
    let mut wins: HashMap<String, usize> = HashMap::new();
    let mut blind: HashMap<String, usize> = HashMap::new();
    let mut owner_of: Vec<Option<usize>> = vec![None; n];
    for i in 0..n {
        let mut best = f32::MIN;
        let mut owner: Option<usize> = None;
        for (hi, hm) in heatmaps.iter().enumerate() {
            if i >= hm.scores.len() {
                continue;
            }
            if hm.scores[i] > best {
                best = hm.scores[i];
                owner = Some(hi);
            }
        }
        if let Some(o) = owner {
            if best > 0.0 {
                if legibility.is_legible(i) {
                    owner_of[i] = Some(o);
                    *wins.entry(heatmaps[o].category.clone()).or_insert(0) += 1;
                } else {
                    *blind.entry(heatmaps[o].category.clone()).or_insert(0) += 1;
                }
            }
        }
    }
    for hm in heatmaps.iter() {
        let lg = wins.get(&hm.category).copied().unwrap_or(0);
        let bl = blind.get(&hm.category).copied().unwrap_or(0);
        if bl == 0 {
            continue;
        }
        let ratio = lg as f32 / (lg + bl) as f32;
        crate::utils::score_dynamics::record_baseline("crop.territory.legible_ratio", ratio);
        emit(&format!(
            "    🕳️ [TERRITORY BLIND] '{}' 가 argmax 로 차지한 {}칸 중 글자가 있는 칸은 {}칸({:.0}%) 뿐입니다. 나머지 {}칸은 여백이라 어떤 축도 설명할 내용이 없으므로 영토에서 제외합니다. 여백을 영토로 남겨두면 세로 밴드 확장이 남의 여백에서 멈추고 구제 지분까지 갉아먹습니다.",
            hm.category, lg + bl, lg, ratio * 100.0, bl
        ));
    }
    let mut out: HashSet<String> = HashSet::new();
    for hm in heatmaps.iter() {
        let w = wins.get(&hm.category).copied().unwrap_or(0);
        let is_identity = !identity_category.is_empty() && hm.category == identity_category;
        if hm.absent && !is_identity {
            emit(&format!(
                "    ⚪ [PRESENCE GATE] '{}' 부재 — {}",
                hm.category,
                if hm.absent_reason.is_empty() {
                    "경쟁 영토 없음"
                } else {
                    &hm.absent_reason
                }
            ));
            continue;
        }
        if w > 0 || is_identity {
            out.insert(hm.category.clone());
            if w == 0 {
                emit(&format!(
                    "    🪪 [PRESENCE GATE / IDENTITY EXEMPT] '{}' 는 argmax 패치가 0개지만 문서 기본키를 담당하므로 면제합니다. (Top: {:+.4})",
                    hm.category, hm.top_score
                ));
            }
        } else {
            emit(&format!(
                "    ⚪ [PRESENCE GATE] '{}' 는 패치 {}개 중 단 한 곳에서도 최강 설명이 되지 못했습니다. 이 문서에 없는 축이므로 크롭하지 않습니다. (Top: {:+.4})",
                hm.category, n, hm.top_score
            ));
        }
    }
    let owned_cnt = owner_of.iter().filter(|o| o.is_some()).count();
    emit(&format!(
        "    🧭 [TERRITORY MAP] 패치 {}개 중 {}개가 소유자를 확정했습니다. 이 맵을 세로 밴드 확장의 정지 근거와 구제 지분 계산에 재사용합니다.",
        n, owned_cnt
    ));
    (out, owner_of)
}

fn ensure_identity_band_crop(
    plans: &mut Vec<CropPlan>,
    heatmaps: &[CategoryHeatmap],
    content: &[f32],
    content_gate: f32,
    grid: &PatchGrid,
    legibility: &crate::models::siglip2::legibility::LegibilityMap,
    identity_category: &str,
    identity_field: &str,
    emit: &dyn Fn(&str),
) {
    let rows = grid.grid_rows;
    let cols = grid.grid_cols;
    if rows < 4 || cols == 0 || identity_category.is_empty() {
        return;
    }
    let hm = match heatmaps.iter().find(|h| h.category == identity_category) {
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
    let band_end = identity_band_end_row(rows, band_start);

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
                p.category == identity_category
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
            "    ✅ [IDENTITY BAND] '{}' 가 식별 밴드 r{}~{} 의 내용 {}/{} 를 이미 점유하고 있습니다.",
            identity_category, band_start, band_end, cov, tot
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

    let (lg, il, bl) = legibility.count_in_bbox(bbox, grid.orig_width, grid.orig_height);
    if lg == 0 {
        emit(&format!(
            "    ⛔ [IDENTITY BAND SKIP] '{}' 식별 밴드 r{}~{} c{}~{} 는 판독 가능 패치가 0개입니다 (판독불가 {} / 여백 {}). 기본키가 이 밴드에 인쇄되어 있지 않으므로 빈 크롭을 추가하지 않습니다.",
            identity_category, band_start, band_end, c0, c1, il, bl
        ));
        return;
    }

    emit(&format!(
        "    🪪 [IDENTITY BAND GUARANTEE] '{}' 가 식별 밴드 내용을 {}/{} 밖에 못 담아 전용 크롭을 추가합니다. r{}~{} c{}~{} → px({},{})-({},{}) | Peak: {:+.4} | 판독 가능 {}",
        identity_category, cov, tot, band_start, band_end, c0, c1,
        bbox.0, bbox.1, bbox.2, bbox.3,
        if peak == f32::MIN { 0.0 } else { peak }, lg
    ));
    plans.push(CropPlan {
        category: identity_category.to_string(),
        bbox,
        score: if peak == f32::MIN { 0.0 } else { peak },
        margin: 0.0,
        patch_count: (band_end - band_start + 1) * (c1 - c0 + 1),
        top_field: identity_field.to_string(),
        owned_patches: 0,
        twin_of: String::new(),
    });
}

fn rescue_uncovered_cells(
    plans: &mut Vec<CropPlan>,
    heatmaps: &[CategoryHeatmap],
    owner_of: &[Option<usize>],
    content: &[f32],
    content_gate: f32,
    grid: &PatchGrid,
    legibility: &crate::models::siglip2::legibility::LegibilityMap,
    area_cap: usize,
    emit: &dyn Fn(&str),
) {
    let rows = grid.grid_rows;
    let cols = grid.grid_cols;
    let n = rows * cols;
    if n == 0 || heatmaps.is_empty() {
        return;
    }
    let cw = grid.orig_width as f32 / cols as f32;
    let ch = grid.orig_height as f32 / rows as f32;

    let mut hole = vec![false; n];
    let mut holes = 0usize;
    for i in 0..n {
        if i >= content.len() || content[i] <= content_gate {
            continue;
        }
        let r = i / cols;
        let c = i % cols;
        let cx = (c as f32 + 0.5) * cw;
        let cy = (r as f32 + 0.5) * ch;
        let covered = plans.iter().any(|p| {
            cx >= p.bbox.0 as f32
                && cx <= p.bbox.2 as f32
                && cy >= p.bbox.1 as f32
                && cy <= p.bbox.3 as f32
        });
        if !covered {
            hole[i] = true;
            holes += 1;
        }
    }
    if holes == 0 {
        emit("    ✅ [COVERAGE GUARANTEE] 모든 내용 칸이 최소 하나의 크롭에 포함되어 있습니다.");
        return;
    }

    let mut miss_rows: Vec<usize> = Vec::new();
    for i in 0..n {
        if hole[i] {
            let r = i / cols;
            if !miss_rows.contains(&r) {
                miss_rows.push(r);
            }
        }
    }
    emit(&format!(
        "    ⚠️ [COVERAGE GUARANTEE] 내용 칸 {}개가 어떤 크롭에도 없습니다 (행 {:?}). 행 단위로만 세면 같은 행의 다른 열이 통째로 빠져도 통과합니다 — 좌우로 떨어진 두 섬을 한 밴드로 묶으면 그 사이 여백까지 삼킵니다.",
        holes,
        miss_rows.iter().take(12).collect::<Vec<_>>()
    ));

    let (mut lo, mut hi) = (f32::MAX, f32::MIN);
    for i in 0..n.min(content.len()) {
        let v = content[i];
        if v == f32::MIN || !v.is_finite() {
            continue;
        }
        if v < lo { lo = v; }
        if v > hi { hi = v; }
    }
    if lo == f32::MAX {
        lo = 0.0;
        hi = 1.0;
    }
    let span = if hi > lo { hi - lo } else { 1.0 };

    let mut field = vec![-1.0f32; n];
    for i in 0..n {
        if !hole[i] {
            continue;
        }
        let v = if i < content.len() && content[i] != f32::MIN && content[i].is_finite() {
            content[i]
        } else {
            lo
        };
        field[i] = (v - lo) / span + 1e-3;
    }

    let blobs = extract_components(&field, rows, cols, 0.0);
    let before_n = blobs.len();
    let first_over = blobs.iter().filter(|c| c.area() > area_cap).count();
    let (mut blobs, passes) =
        split_until_fits(blobs, &field, rows, cols, area_cap, RESCUE_SPLIT_PASSES);
    if first_over > 0 {
        let left = blobs.iter().filter(|c| c.area() > area_cap).count();
        emit(&format!(
            "    ✂️ [RESCUE SPLIT] 미커버 덩이 {}개가 면적 상한 {}칸을 넘어 {}개 → {}개로 쪼갰습니다 ({}회 반복 / 잔여 초과 {}개). 한 번만 쪼개면 갈라진 조각이 다시 상한을 넘어도 그대로 버려지므로 더 갈라지지 않을 때까지 분위수 게이트를 올려 가며 반복합니다.",
            first_over, area_cap, before_n, blobs.len(), passes, left
        ));
    }

    blobs.sort_by(|a, b| b.indices.len().cmp(&a.indices.len()));

    let min_w = ((grid.orig_width as f32 * 0.12) as u32).max(64);
    let min_h = ((grid.orig_height as f32 * 0.06) as u32).max(48);
    let blob_n = blobs.len();
    let mut added = 0usize;

    for comp in blobs.iter() {
        if added >= RESCUE_MAX {
            break;
        }
        if comp.indices.len() < RESCUE_MIN_CELLS {
            continue;
        }
        if comp.area() > area_cap {
            emit(&format!(
                "    ⛔ [RESCUE SKIP] 미커버 덩이 grid(r{}~{}, c{}~{}) 은 {}칸으로 상한 {}칸을 넘습니다. 쪼개지지 않는 큰 여백이라 크롭하지 않습니다 — 이 자리에 통 크롭을 만들면 여러 카테고리 글자가 한 축으로 몰립니다.",
                comp.r_min, comp.r_max, comp.c_min, comp.c_max, comp.area(), area_cap
            ));
            continue;
        }

        let mut owner_hi: Option<usize> = None;
        let mut best_rank = f32::MIN;
        let mut best_share = 0.0f32;
        let mut best_score = f32::MIN;
        let mut near = String::new();
        let mut near_share = 0.0f32;

        for (hi, hm) in heatmaps.iter().enumerate() {
            let mut own = 0usize;
            let mut sum = 0.0f32;
            let mut peak = f32::MIN;
            for &i in comp.indices.iter() {
                if i >= hm.scores.len() || i >= owner_of.len() {
                    continue;
                }
                if owner_of[i] != Some(hi) {
                    continue;
                }
                own += 1;
                sum += hm.scores[i];
                if hm.scores[i] > peak {
                    peak = hm.scores[i];
                }
            }
            let share = own as f32 / comp.indices.len().max(1) as f32;
            if share > near_share {
                near_share = share;
                near = hm.category.clone();
            }
            if own == 0 || share < RESCUE_OWNER_MIN_SHARE {
                continue;
            }
            let rank = share * (sum / own as f32);
            if rank > best_rank {
                best_rank = rank;
                best_share = share;
                best_score = peak;
                owner_hi = Some(hi);
            }
        }

        let hi = match owner_hi {
            Some(v) => v,
            None => {
                emit(&format!(
                    "    ⛔ [RESCUE OWNER] 미커버 덩이 grid(r{}~{}, c{}~{}) 은 어느 카테고리도 지분이 {:.0}% 를 넘지 못합니다 (최고 '{}' {:.0}%). 지분에 평균 점수를 곱한 값을 지분 임계와 비교하면 한 칸짜리 봉우리가 점수만으로 문턱을 넘으므로, 문턱은 지분만 / 순위는 지분×평균으로 나눕니다.",
                    comp.r_min, comp.r_max, comp.c_min, comp.c_max,
                    RESCUE_OWNER_MIN_SHARE * 100.0,
                    if near.is_empty() { "-" } else { &near },
                    near_share * 100.0
                ));
                continue;
            }
        };

        let gbox = (comp.r_min, comp.r_max, comp.c_min, comp.c_max);
        let raw = to_pixel_bbox(gbox, grid);
        let bbox = ensure_min_size(raw, grid.orig_width, grid.orig_height, min_w, min_h);

        let (lg, il, bl) = legibility.count_in_bbox(bbox, grid.orig_width, grid.orig_height);
        if lg == 0 {
            emit(&format!(
                "    ⛔ [COVERAGE SKIP] 미커버 덩이 grid(r{}~{}, c{}~{}) 는 판독 가능 패치가 0개입니다 (판독불가 {} / 여백 {}). 로고 테두리나 도장이 내용 마스크를 통과한 자리이며, STEP 5 의 EMPTY CROP SKIP 이 같은 기준으로 거부하므로 크롭을 만들지 않습니다.",
                comp.r_min, comp.r_max, comp.c_min, comp.c_max, il, bl
            ));
            continue;
        }

        if plans.iter().any(|p| px_iou(p.bbox, bbox) >= CROP_SNAP_IOU) {
            continue;
        }

        emit(&format!(
            "    🩹 [COVERAGE RESCUE] 미커버 덩이 r{}~{} c{}~{} ({}칸) 를 '{}' 소유(지분 {:.0}%)로 전용 크롭합니다. → px({},{})-({},{}) | Peak: {:+.4} | 판독 가능 {} — 기존 크롭을 넓히지 않습니다.",
            comp.r_min, comp.r_max, comp.c_min, comp.c_max, comp.indices.len(),
            heatmaps[hi].category, best_share * 100.0,
            bbox.0, bbox.1, bbox.2, bbox.3,
            if best_score == f32::MIN { 0.0 } else { best_score }, lg
        ));
        plans.push(CropPlan {
            category: heatmaps[hi].category.clone(),
            bbox,
            score: if best_score == f32::MIN { 0.0 } else { best_score },
            margin: 0.0,
            patch_count: comp.indices.len(),
            top_field: heatmaps[hi].top_field.clone(),
            owned_patches: (best_share * comp.indices.len() as f32).round() as usize,
            twin_of: String::new(),
        });
        added += 1;
    }

    emit(&format!(
        "    🩹 [COVERAGE RESCUE] 미커버 칸 {}개를 덩이 {}개로 묶어 큰 것부터 {}건만 전용 크롭했습니다 (상한 {}건 — 크롭이 늘면 VLM 호출과 업스케일 버퍼가 함께 늘어납니다).",
        holes, blob_n, added, RESCUE_MAX
    ));
}

/// 🌟 [DOC TEXT HEIGHT BASELINE] 문서 전체의 글자 높이 중앙값과 MAD 를 한 번만 측정합니다.
///
///  ── 왜 문서 수준인가 ──
///   크롭 하나의 잉크 행 밴드가 2개뿐이면 "최솟값" 기준이 곧 "유일값" 이 되어
///   그 밴드가 글자 한 행인지 서명 획 덩어리인지 구분할 근거가 사라집니다.
///   (실측: 복구 창 7 이 44.0px 를 채택해 배율 1.00x → 70토큰 → 한 글자도 못 읽음.
///    같은 문서의 다른 22개 크롭은 8.0~14.4px, 중앙값 10.0px 였습니다)
///   이상치는 자기 안에서 보이지 않고 무리 안에서만 보이므로 기준선을 문서로 올립니다.
///
///  ── 왜 중앙값 + MAD 인가 ──
///   이 코드베이스가 SPATIAL RESIDUAL GATE 와 vision_encoder 의 robust z 에서
///   이미 쓰는 동일한 분포 판정입니다. 새 상수가 생기지 않습니다.
///
///  ── 반환 ──
///   (중앙값, MAD). 행 밴드가 3개 미만이면 판정이 불가능하므로 None.
pub fn measure_doc_text_height(
    img: &image::DynamicImage,
    emit: &dyn Fn(&str),
) -> Option<(f32, f32)> {
    let gray = img.to_luma8();
    let (w, h) = (gray.width() as usize, gray.height() as usize);
    if w == 0 || h < 8 { return None; }

    let mut row_ink: Vec<u32> = vec![0; h];
    let mut row_runs: Vec<u32> = vec![0; h];
    let mut row_longest: Vec<u32> = vec![0; h];
    for y in 0..h {
        let mut c = 0u32;
        let mut runs = 0u32;
        let mut cur = 0u32;
        let mut longest = 0u32;
        for x in 0..w {
            if gray.get_pixel(x as u32, y as u32).0[0] < 160 {
                c += 1;
                if cur == 0 { runs += 1; }
                cur += 1;
                if cur > longest { longest = cur; }
            } else {
                cur = 0;
            }
        }
        row_ink[y] = c;
        row_runs[y] = runs;
        row_longest[y] = longest;
    }

    let total: u64 = row_ink.iter().map(|v| *v as u64).sum();
    let mean = total as f32 / h as f32;
    if mean <= 0.0 { return None; }

    let is_text_row = |y: usize| -> bool {
        if (row_ink[y] as f32) <= mean { return false; }
        row_runs[y] >= 2 && (row_longest[y] as usize) * 2 < w
    };

    let mut bands: Vec<f32> = Vec::new();
    let mut rule_rows = 0usize;
    let mut run = 0usize;
    for y in 0..h {
        if is_text_row(y) {
            run += 1;
            continue;
        }
        if (row_ink[y] as f32) > mean { rule_rows += 1; }
        if run > 0 {
            bands.push(run as f32);
            run = 0;
        }
    }
    if run > 0 { bands.push(run as f32); }
    if bands.len() < 3 {
        emit(&format!(
            "  ⚪ [DOC TEXT HEIGHT] 괘선을 걷어내고 남은 글자 행 밴드가 {}개뿐이라 기준선을 세우지 못합니다. 클램프를 적용하지 않고 크롭별 추정을 그대로 씁니다.",
            bands.len()
        ));
        return None;
    }

    bands.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = bands[bands.len() / 2];

    let mut dev: Vec<f32> = bands.iter().map(|b| (b - median).abs()).collect();
    dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = dev[dev.len() / 2] * 1.4826;

    emit(&format!(
        "  📐 [DOC TEXT HEIGHT] 글자 행 밴드 {}개 | 중앙값 {:.1}px | MAD {:.1}px | 허용 상한 {:.1}px | 괘선으로 배제한 잉크 행 {}줄. 표 테두리는 행 전체가 하나의 연속 잉크 런이고 글자 행은 글자 사이 공백 때문에 런이 끊깁니다. 이 구분이 없으면 격자가 촘촘한 서식에서 1~2px 괘선이 밴드 다수를 차지해 중앙값이 글자 높이가 아니라 선 두께가 되고, 그러면 전 크롭이 상한 배율로 포화되어 추정이 사실상 상수가 됩니다.",
        bands.len(), median, mad, median + mad.max(1.0), rule_rows
    ));
    crate::utils::score_dynamics::record_baseline("vision.doc_text_height", median);
    crate::utils::score_dynamics::record_baseline(
        "vision.doc_rule_rows",
        rule_rows as f32 / h.max(1) as f32,
    );
    Some((median, mad))
}

/// 🌟 [TEXT HEIGHT CLAMP] 크롭 하나가 추정한 글자 높이를 문서 기준선으로 되돌립니다.
///
///  ── 왜 상한만 거는가 ──
///   과소 추정은 배율을 올려 토큰만 늘릴 뿐 판독을 해치지 않습니다.
///   과대 추정은 배율을 1.00x 로 떨어뜨려 판독 자체를 불가능하게 만듭니다.
///   손실이 비대칭이므로 상한만 강제합니다.
pub fn clamp_text_height(
    estimated: f32,
    baseline: Option<(f32, f32)>,
    tag: &str,
    emit: &dyn Fn(&str),
) -> f32 {
    let (median, mad) = match baseline { Some(v) => v, None => return estimated };
    let cap = median + mad.max(1.0);
    if estimated <= cap { return estimated; }
    emit(&format!(
        "    📐 [TEXT HEIGHT CLAMP / {}] 추정 {:.1}px 가 문서 상한 {:.1}px(중앙값 {:.1} + MAD {:.1})를 넘어 중앙값으로 되돌립니다. 이 크롭의 잉크 행 밴드가 서명 획이나 여러 행이 붙은 덩어리를 한 행으로 재고 있습니다. 그대로 두면 배율이 1.00x 로 떨어져 2B 모델이 한 글자도 읽지 못합니다.",
        tag, estimated, cap, median, mad
    ));
    crate::utils::score_dynamics::record_baseline("vision.text_height_clamp", 1.0);
    median
}

pub fn plan_crops(
    heatmaps: &[CategoryHeatmap],
    grid: &PatchGrid,
    legibility: &crate::models::siglip2::legibility::LegibilityMap,
    table_categories: &[&str],
    identity_category: &str,
    identity_field: &str,
    emit: &dyn Fn(&str),
) -> Vec<CropPlan> {
    if heatmaps.is_empty() || grid.len() == 0 {
        return Vec::new();
    }

    let rows = grid.grid_rows;
    let cols = grid.grid_cols;
    let n = rows * cols;

    let (content, content_gate) = build_content_mask(heatmaps, n);
    let content_cnt = content.iter().filter(|v| **v > content_gate).count();
    emit(&format!(
        "    🗺️ [CONTENT MASK] 내용 패치 {}/{} | gate {:+.4} (라벨↔값 행 밴드 확장 기준)",
        content_cnt, n, content_gate
    ));

    let legible_cnt = (0..n).filter(|&i| legibility.is_legible(i)).count();
    emit(&format!(
        "    🔎 [BLANK GATE] 판독 가능 패치 {}/{} | 교차 판독 가능 패치가 0개인 후보 영역은 크롭하지 않습니다. STEP 5 의 EMPTY CROP SKIP 이 정확히 같은 기준(count_in_bbox)으로 호출을 거부하므로, 계획만 무르게 두면 업스케일 버퍼를 잡았다가 한 글자도 못 읽고 버립니다.",
        legible_cnt, n
    ));

    let (present, owner_of) = presence_gate(heatmaps, n, identity_category, legibility, emit);

    let area_cap = (n / present.len().max(1)).max(4);

    let region_blank = |bbox: (u32, u32, u32, u32)| -> (bool, usize, usize, usize) {
        let (lg, il, bl) =
            legibility.count_in_bbox(bbox, grid.orig_width, grid.orig_height);
        (lg == 0, lg, il, bl)
    };

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

    for (hi, hm) in heatmaps.iter().enumerate() {
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
        let is_table_cat = table_categories.iter().any(|c| *c == hm.category.as_str());

        let owned_mask: Vec<bool> = owner_of
            .iter()
            .map(|o| matches!(o, Some(x) if *x == hi))
            .collect();

        let mut gboxes: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut peaks: Vec<f32> = Vec::new();
        let mut counts: Vec<usize> = Vec::new();
        let mut blank_skipped = 0usize;

        if is_table_cat {
            if let Some(tb) = table_union(&comps, &content, content_gate, rows, cols) {
                let area = (tb.1 - tb.0 + 1) * (tb.3 - tb.2 + 1);
                let px = to_pixel_bbox(tb, grid);
                let (blank, lg, il, bl) = region_blank(px);
                if blank {
                    blank_skipped += 1;
                    emit(&format!(
                        "    ⛔ [EMPTY REGION SKIP] '{}' 표 밴드 r{}~{}, c{}~{} 는 판독 가능 패치가 0개입니다 (판독불가 {} / 여백 {}). 히트맵 봉우리가 표 괘선이나 얼룩에 찍힌 것이므로 성분 단위 크롭으로 넘어갑니다.",
                        hm.category, tb.0, tb.1, tb.2, tb.3, il, bl
                    ));
                } else {
                    emit(&format!(
                        "    🧾 [TABLE UNION] '{}' | 표 밴드 r{}~{}, c{}~{} ({}패치 / 판독 가능 {}) 로 통합",
                        hm.category, tb.0, tb.1, tb.2, tb.3, area, lg
                    ));
                    gboxes.push(tb);
                    peaks.push(comps[0].peak);
                    counts.push(area);
                }
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

                let width = (expanded.3 - expanded.2 + 1).max(1);
                let cap_h = (area_cap / width).max(2);
                let (grown, stop) = expand_col_band(
                    expanded,
                    &content,
                    content_gate,
                    rows,
                    cols,
                    cap_h,
                    Some(&owned_mask),
                );
                if grown.0 != expanded.0 || grown.1 != expanded.1 {
                    emit(&format!(
                        "    ↕️ [COL BAND] '{}' | r{}~{} → r{}~{} (같은 열 밴드의 아래 값 셀 편입 — 이 서식은 라벨이 위, 값이 아래라 가로만 넓히면 값 행이 크롭 밖에 남습니다) | 정지: {}",
                        hm.category, expanded.0, expanded.1, grown.0, grown.1, stop
                    ));
                }

                let px = to_pixel_bbox(grown, grid);
                let (blank, _lg, il, bl) = region_blank(px);
                if blank {
                    blank_skipped += 1;
                    emit(&format!(
                        "    ⛔ [EMPTY REGION SKIP] '{}' 후보 grid(r{}~{}, c{}~{}) 는 판독 가능 패치가 0개입니다 (판독불가 {} / 여백 {}). 봉우리가 여백이나 판독불가 얼룩에 찍힌 것이므로 다음 후보 영역으로 넘어갑니다 — 빈 크롭을 VLM 에 보내면 없는 사실이 생성됩니다.",
                        hm.category, grown.0, grown.1, grown.2, grown.3, il, bl
                    ));
                    continue;
                }

                gboxes.push(grown);
                peaks.push(comp.peak);
                counts.push(comp.indices.len());
            }
        }

        if gboxes.is_empty() {
            emit(&format!(
                "    ⚪ [NO REGION] '{}' 는 후보 영역 {}개가 전부 판독 가능 패치 0개였습니다. 이 문서에 인쇄되지 않은 축으로 보고 크롭하지 않습니다.",
                hm.category, blank_skipped
            ));
            continue;
        }

        emit(&format!(
            "    🧩 [COMPONENTS] '{}' | 영역 {}개 (빈 영역 {}개 제외) | Gate: {:+.4} | Top: {}({:+.4})",
            hm.category,
            gboxes.len(),
            blank_skipped,
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
        let (gbox, score, margin, patch_count) = match a {
            Some((pi, score, margin)) => (pool.boxes[*pi], *score, *margin, pool.counts[*pi]),
            None => {
                let (_, _, gboxes, peaks, counts) = &per_cat[ci];
                let best = peaks
                    .iter()
                    .enumerate()
                    .max_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal));
                match best {
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

        {
            let (blank, _lg, il, bl) = region_blank(bbox);
            if blank {
                emit(&format!(
                    "    ⛔ [EMPTY CROP SKIP / PLAN] '{}' 최종 px({},{})-({},{}) 안에 판독 가능 패치가 0개입니다 (판독불가 {} / 여백 {}). ensure_min_size 가 여백 쪽으로 부풀린 결과이므로 계획 단계에서 버립니다.",
                    per_cat[ci].0, bbox.0, bbox.1, bbox.2, bbox.3, il, bl
                ));
                continue;
            }
        }

        let margin = {
            let terr = heatmaps
                .iter()
                .find(|h| h.category == per_cat[ci].0)
                .map(|h| h.mean_margin)
                .unwrap_or(0.0);
            if terr > margin { terr } else { margin }
        };

        let owned_patches = {
            let cw = grid.orig_width as f32 / cols as f32;
            let ch = grid.orig_height as f32 / rows as f32;
            match heatmaps.iter().position(|h| h.category == per_cat[ci].0) {
                Some(h) => (0..n)
                    .filter(|&i| {
                        if owner_of.get(i).copied().flatten() != Some(h) {
                            return false;
                        }
                        let r = i / cols;
                        let c = i % cols;
                        let cx = (c as f32 + 0.5) * cw;
                        let cy = (r as f32 + 0.5) * ch;
                        cx >= bbox.0 as f32
                            && cx <= bbox.2 as f32
                            && cy >= bbox.1 as f32
                            && cy <= bbox.3 as f32
                    })
                    .count(),
                None => 0,
            }
        };

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
                // 🌟 [SDS 계측] 커버리지 손실률을 남깁니다.
                //
                //  ── 실측 ──
                //   score_dynamics.json 의 spatial[*].coverage_loss 가 전부 n=0 입니다.
                //   Phase 0 에서 record_coverage_loss 를 정의만 하고
                //   호출부를 넣지 않았기 때문입니다.
                //   이 값이 없으면 V-2(크롭 재수립)의 기대 밴드를 유도할 수 없습니다.
                //
                //  ── 왜 경고 조건 밖인가 ──
                //   경고(coverage < 0.70)만 기록하면 '손실이 적은 정상 케이스' 가
                //   표본에서 빠져 분포가 한쪽으로 치우칩니다.
                //   V-2 는 '이 서식의 통상 손실률' 을 알아야 하므로 전량 기록합니다.
                crate::utils::score_dynamics::record_coverage_loss(
                    &per_cat[ci].0,
                    1.0 - coverage,
                );
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

        if owned_patches == 0 {
            emit(&format!(
                "    🧭 [TERRITORY TAG] '{}' 크롭은 최종 좌표 안에 자기 영토 패치를 한 칸도 담지 않았습니다. 이 크롭에서는 새 필드를 만들지 말고 명시된 라벨↔값만 읽어야 합니다.",
                per_cat[ci].0
            ));
        }

        plans.push(CropPlan {
            category: per_cat[ci].0.clone(),
            bbox,
            score,
            margin,
            patch_count,
            top_field: per_cat[ci].1.clone(),
            owned_patches,
            twin_of: String::new(),
        });
    }

    {
        let cw = grid.orig_width as f32 / cols as f32;
        let ch = grid.orig_height as f32 / rows as f32;
        let inside = |b: &(u32, u32, u32, u32), cx: f32, cy: f32| -> bool {
            cx >= b.0 as f32 && cx <= b.2 as f32 && cy >= b.1 as f32 && cy <= b.3 as f32
        };

        let mut split_total = 0usize;

        for (cat, field, gboxes, peaks, counts) in per_cat.iter() {
            let hm = match heatmaps.iter().find(|h| &h.category == cat) {
                Some(h) => h,
                None => continue,
            };
            let cur: Vec<(u32, u32, u32, u32)> = plans
                .iter()
                .filter(|p| &p.category == cat)
                .map(|p| p.bbox)
                .collect();
            if cur.is_empty() {
                continue;
            }

            let m = n.min(hm.scores.len());
            let mut hot = 0usize;
            let mut covered = 0usize;
            for i in 0..m {
                if hm.scores[i] <= 0.0 {
                    continue;
                }
                hot += 1;
                let cx = ((i % cols) as f32 + 0.5) * cw;
                let cy = ((i / cols) as f32 + 0.5) * ch;
                if cur.iter().any(|b| inside(b, cx, cy)) {
                    covered += 1;
                }
            }
            if hot < SPLIT_MIN_HOT {
                continue;
            }
            let ratio = covered as f32 / hot as f32;
            if ratio >= SPLIT_COVERAGE_FLOOR {
                continue;
            }

            let mut cands: Vec<(usize, usize, (u32, u32, u32, u32), usize)> = Vec::new();
            for (gi, gb) in gboxes.iter().enumerate() {
                let raw = to_pixel_bbox(*gb, grid);
                let bbox = ensure_min_size(raw, grid.orig_width, grid.orig_height, min_w, min_h);
                if plans.iter().any(|p| px_iou(p.bbox, bbox) >= CROP_SNAP_IOU) {
                    continue;
                }
                let (lg, _il, _bl) =
                    legibility.count_in_bbox(bbox, grid.orig_width, grid.orig_height);
                if lg == 0 {
                    continue;
                }
                let mut gain = 0usize;
                for i in 0..m {
                    if hm.scores[i] <= 0.0 {
                        continue;
                    }
                    let cx = ((i % cols) as f32 + 0.5) * cw;
                    let cy = ((i / cols) as f32 + 0.5) * ch;
                    if !inside(&bbox, cx, cy) {
                        continue;
                    }
                    if cur.iter().any(|b| inside(b, cx, cy)) {
                        continue;
                    }
                    gain += 1;
                }
                if gain < SPLIT_MIN_GAIN {
                    continue;
                }
                cands.push((gain, gi, bbox, lg));
            }
            cands.sort_by(|a, b| b.0.cmp(&a.0));

            let mut added = 0usize;
            let mut taken: Vec<(u32, u32, u32, u32)> = Vec::new();
            let mut acc = covered;

            for (gain, gi, bbox, lg) in cands.into_iter() {
                if added >= SPLIT_CROP_LIMIT {
                    break;
                }
                if taken.iter().any(|b| px_iou(*b, bbox) >= CROP_SNAP_IOU) {
                    continue;
                }
                acc += gain;
                emit(&format!(
                    "    ➕ [SPLIT CROP] '{}' 커버리지 {:.0}% < {:.0}% — 배정받지 못한 자기 영역 grid(r{}~{}, c{}~{}) 를 크롭으로 하나 더 만듭니다. → px({},{})-({},{}) | 신규 커버 {}칸 | 판독 가능 {} | 누적 {:.0}%",
                    cat,
                    ratio * 100.0,
                    SPLIT_COVERAGE_FLOOR * 100.0,
                    gboxes[gi].0, gboxes[gi].1, gboxes[gi].2, gboxes[gi].3,
                    bbox.0, bbox.1, bbox.2, bbox.3,
                    gain, lg,
                    acc as f32 / hot as f32 * 100.0
                ));
                plans.push(CropPlan {
                    category: cat.clone(),
                    bbox,
                    score: peaks[gi],
                    margin: 0.0,
                    patch_count: counts[gi],
                    top_field: field.clone(),
                    owned_patches: gain,
                    twin_of: String::new(),
                });
                taken.push(bbox);
                added += 1;
                split_total += 1;
            }

            if added == 0 {
                emit(&format!(
                    "    ⚪ [SPLIT SKIP] '{}' 커버리지 {:.0}% 이지만 추가할 영역이 없습니다. 남은 후보가 이미 배정된 크롭과 겹치거나 판독 가능 패치가 0개이거나 신규 커버가 {}칸 미만입니다. 히트맵이 페이지 전반에 퍼져 단일 영역으로 좁혀지지 않는 상태입니다.",
                    cat, ratio * 100.0, SPLIT_MIN_GAIN
                ));
            }
        }

        if split_total > 0 {
            emit(&format!(
                "    ➕ [SPLIT CROP] 커버리지 미달 카테고리에 크롭 {}건을 추가했습니다 (카테고리당 상한 {}건). 배타 배정은 카테고리당 영역 하나만 주므로, 확장된 큰 박스가 다른 카테고리에 선점되면 남는 것은 1×1 조각뿐입니다.",
                split_total, SPLIT_CROP_LIMIT
            ));
        }
    }

    ensure_identity_band_crop(
        &mut plans,
        heatmaps,
        &content,
        content_gate,
        grid,
        legibility,
        identity_category,
        identity_field,
        emit,
    );

    rescue_uncovered_cells(
        &mut plans,
        heatmaps,
        &owner_of,
        &content,
        content_gate,
        grid,
        legibility,
        area_cap,
        emit,
    );

    {
        let page_area = (grid.orig_width as f32) * (grid.orig_height as f32);
        let cw = grid.orig_width as f32 / cols as f32;
        let ch = grid.orig_height as f32 / rows as f32;

        let fill_of = |b: (u32, u32, u32, u32)| -> f32 {
            let mut tot = 0usize;
            let mut got = 0usize;
            for i in 0..n {
                let cx = ((i % cols) as f32 + 0.5) * cw;
                let cy = ((i / cols) as f32 + 0.5) * ch;
                if cx < b.0 as f32 || cx > b.2 as f32 || cy < b.1 as f32 || cy > b.3 as f32 {
                    continue;
                }
                tot += 1;
                if i < content.len() && content[i] > content_gate {
                    got += 1;
                }
            }
            if tot == 0 { 0.0 } else { got as f32 / tot as f32 }
        };

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
                let iou = px_iou(a, b);
                let contained = px_covers(a, b) || px_covers(b, a);
                if !contained && iou < CROP_MERGE_IOU {
                    emit(&format!(
                        "    ⚪ [MERGE SKIP] '{}' 의 두 크롭은 IoU {:.2} < {:.2} 로 서로 다른 지면입니다. px({},{})-({},{}) 와 px({},{})-({},{}) 를 따로 읽습니다 — 한 픽셀이라도 겹치면 합치면 SPLIT CROP 이 직후에 되삼켜져 통짜 크롭이 됩니다.",
                        plans[i].category, iou, CROP_MERGE_IOU,
                        a.0, a.1, a.2, a.3, b.0, b.1, b.2, b.3
                    ));
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
                if area > page_area * MERGE_PAGE_RATIO {
                    emit(&format!(
                        "    ⚪ [MERGE SKIP] '{}' 의 두 크롭을 합치면 페이지의 {:.0}% 를 차지해 병합하지 않습니다.",
                        plans[i].category,
                        area / page_area * 100.0
                    ));
                    j += 1;
                    continue;
                }
                let base_fill = fill_of(a).max(fill_of(b));
                let merged_fill = fill_of(merged);
                if base_fill > 0.0 && merged_fill < base_fill * MERGE_FILL_DROP {
                    emit(&format!(
                        "    ⚪ [MERGE SKIP] '{}' 병합 사각형의 내용 밀도가 {:.0}% 로 원본 {:.0}% 대비 급락합니다. 두 크롭 사이가 여백이라는 뜻이므로 합치지 않습니다.",
                        plans[i].category,
                        merged_fill * 100.0,
                        base_fill * 100.0
                    ));
                    j += 1;
                    continue;
                }
                emit(&format!(
                    "    🔗 [CROP MERGE] '{}' 의 겹치는 크롭 2개를 합칩니다 (IoU {:.2}{}). px({},{})-({},{}) + px({},{})-({},{}) → px({},{})-({},{}) | 내용 밀도 {:.0}%→{:.0}%",
                    plans[i].category, iou,
                    if contained { " / 완전 포함" } else { "" },
                    a.0, a.1, a.2, a.3,
                    b.0, b.1, b.2, b.3,
                    merged.0, merged.1, merged.2, merged.3,
                    base_fill * 100.0, merged_fill * 100.0
                ));
                plans[i].bbox = merged;
                plans[i].patch_count += plans[j].patch_count;
                plans[i].owned_patches += plans[j].owned_patches;
                if plans[j].score > plans[i].score {
                    plans[i].score = plans[j].score;
                    plans[i].top_field = plans[j].top_field.clone();
                }
                plans.remove(j);
            }
            i += 1;
        }
    }

    {
        let mut order: Vec<usize> = (0..plans.len()).collect();
        order.sort_by(|&a, &b| {
            plans[b]
                .score
                .partial_cmp(&plans[a].score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let is_table = |c: &str| -> bool { table_categories.iter().any(|t| *t == c) };

        let mut seen: Vec<((u32, u32, u32, u32), String)> = Vec::new();
        let mut snapped = 0usize;
        let mut twins: Vec<String> = Vec::new();
        let mut shared_tables: Vec<String> = Vec::new();

        for &pi in order.iter() {
            let cand = plans[pi].bbox;
            let cand_cat = plans[pi].category.clone();
            let mut owner: Option<String> = None;

            for (bx, cat) in seen.iter() {
                let same = *bx == cand;
                let iou = if same { 1.0 } else { px_iou(*bx, cand) };
                if !same && iou < CROP_SNAP_IOU {
                    continue;
                }
                let covers = same || px_covers(*bx, cand);
                if !same && !covers && iou < CROP_TWIN_IOU {
                    continue;
                }

                if is_table(&cand_cat) && is_table(cat) {
                    shared_tables.push(format!("{}↔{}", cand_cat, cat));
                    emit(&format!(
                        "    🧾 [SHARED TABLE EXEMPT] '{}' 와 '{}' 는 둘 다 표 카테고리이고 table_union 이 같은 표 밴드를 배정했습니다. 좌표가 같은 것이 정상이므로 쌍둥이로 묶지 않습니다 — 묶으면 점수가 낮은 쪽 표가 통째로 읽히지 않습니다.",
                        cand_cat, cat
                    ));
                    if !same {
                        plans[pi].bbox = *bx;
                        snapped += 1;
                    }
                    owner = None;
                    break;
                }

                if !same {
                    emit(&format!(
                        "    🔗 [CROP SNAP] '{}' px({},{})-({},{}) 를 '{}' px({},{})-({},{}) 에 맞춥니다 (IoU {:.2}{}). 격자 박스가 같은데 픽셀 좌표가 몇십 px 어긋나면 완전 일치 검사도 판독 원장도 놓쳐 같은 지면을 두 번 읽습니다.",
                        cand_cat, cand.0, cand.1, cand.2, cand.3,
                        cat, bx.0, bx.1, bx.2, bx.3, iou,
                        if covers { " / 완전 포함" } else { "" }
                    ));
                    plans[pi].bbox = *bx;
                    snapped += 1;
                }
                owner = Some(cat.clone());
                break;
            }

            match owner {
                None => seen.push((plans[pi].bbox, cand_cat)),
                Some(cat) => {
                    twins.push(cand_cat);
                    plans[pi].twin_of = cat;
                }
            }
        }

        if !shared_tables.is_empty() {
            emit(&format!(
                "    🧾 [SHARED TABLE EXEMPT] 표 밴드를 공유하는 쌍 {}건 ({}) 은 쌍둥이 판정에서 면제했습니다. 두 카테고리가 같은 지면을 각자의 필드 집합으로 읽습니다.",
                shared_tables.len(), shared_tables.join(", ")
            ));
        }
        if !twins.is_empty() {
            emit(&format!(
                "    👯 [TWIN CROP] 좌표가 같아진 크롭 {}건 ({}) — 좌표 스냅 {}건. 점수가 낮은 쪽은 배열을 만들지 않습니다.",
                twins.len(), twins.join(", "), snapped
            ));
        }
    }

    {
        let before: Vec<String> = plans.iter().map(|p| p.category.clone()).collect();
        plans.sort_by_key(|p| {
            if !identity_category.is_empty()
                && p.category == identity_category
                && p.top_field == identity_field
            {
                0u8
            } else if !identity_category.is_empty() && p.category == identity_category {
                1u8
            } else if table_categories.iter().any(|c| *c == p.category.as_str()) {
                2u8
            } else {
                3u8
            }
        });
        let after: Vec<String> = plans.iter().map(|p| p.category.clone()).collect();
        if before != after {
            emit(&format!(
                "    🥇 [IDENTITY FIRST ORDER] 문서 기본키 크롭을 선두로, 표 크롭을 스칼라 크롭보다 앞으로 재정렬했습니다. {:?} → {:?} — 표 행이 먼저 병합되어 있어야 뒤의 스칼라 크롭이 표 칸을 라벨↔값 쌍으로 다시 읽었을 때 TABLE CELL ECHO 가 그 값을 문서 총계로 올리지 않습니다. 표 행은 금지 목록(ALREADY CLAIMED)에 들어가지 않으므로, 순서를 앞당겨도 스칼라 크롭의 스키마 패스가 받는 금지 목록에 표 칸 값이 새로 실리지 않습니다.",
                before, after
            ));
        }
    }

    plans
}

const VISION_PATCH_PX: f32 = 28.0;

const TEXT_HEIGHT_MIN_COHERENCE: f32 = 0.15;
const TEXT_HEIGHT_FLOOR_PX: f32 = 8.0;
const TEXT_HEIGHT_CROP_FRACTION: f32 = 40.0;
const TEXT_HEIGHT_CEIL_FRACTION: f32 = 3.0;
const TEXT_HEIGHT_MIN_BANDS: usize = 4;

fn band_measured_text_height(img: &DynamicImage, w: u32, h: u32) -> Option<f32> {
    let mut band_h: Vec<f32> = text_row_bands(img, (0, 0, w, h))
        .iter()
        .map(|b| b.y1.saturating_sub(b.y0) as f32)
        .filter(|v| *v > 0.0)
        .collect();
    if band_h.is_empty() {
        println!("    📏 [TEXT HEIGHT / BAND MEASURED] 잉크 행 밴드가 0개입니다. 이 크롭에는 글자 행이 없으므로 추정을 포기합니다.");
        return None;
    }
    band_h.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = band_h.len();
    let picked = if n >= TEXT_HEIGHT_MIN_BANDS { band_h[n / 4] } else { band_h[0] };
    let floor = TEXT_HEIGHT_FLOOR_PX.max(h as f32 / TEXT_HEIGHT_CROP_FRACTION);
    let ceil = (h as f32 / TEXT_HEIGHT_CEIL_FRACTION).max(floor);
    let th = picked.max(floor).min(ceil);
    crate::utils::score_dynamics::record_baseline("crop.text_height.band_only", th);
    println!(
        "    📏 [TEXT HEIGHT / BAND MEASURED] 잉크 행 밴드 {}개 | 채택 기준 {} | 실측 {:.1}px → 채택 {:.1}px (허용 {:.1}~{:.1}px, 크롭 높이 {}px). 자기상관 주기가 성립하지 않는 소형 크롭에서는 실측한 행 밴드 높이 자체가 글자 높이입니다.",
        n,
        if n >= TEXT_HEIGHT_MIN_BANDS { "하위 1/4" } else { "최솟값" },
        picked, th, floor, ceil, h
    );
    Some(th)
}

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

    let var: f32 = prof.iter().map(|v| v * v).sum::<f32>() / prof.len() as f32;
    if var <= 1e-6 {
        println!("    📏 [TEXT HEIGHT REJECT] 행 프로파일 분산이 0 입니다. 잉크가 없는 영역이므로 추정을 기각합니다.");
        return None;
    }

    // ── 자기상관 최대 주기 = 텍스트 라인 피치 ──
    let max_lag = ((h / 2) as usize).min(120);
    let min_lag = ((TEXT_HEIGHT_FLOOR_PX / 0.6).ceil() as usize).max(4);
    if max_lag <= min_lag {
        println!(
            "    📏 [TEXT HEIGHT REJECT] 크롭 높이 {}px 로는 탐색 구간(lag {}~{})이 성립하지 않습니다. 실측 행 밴드로 폴백합니다.",
            h, min_lag, max_lag
        );
        return band_measured_text_height(img, w, h);
    }

    let mut best_lag = 0usize;
    let mut best = f32::MIN;
    for lag in min_lag..max_lag {
        let mut s = 0.0f32;
        for i in 0..(prof.len() - lag) {
            s += prof[i] * prof[i + lag];
        }
        let norm = s / (prof.len() - lag) as f32;
        if norm > best { best = norm; best_lag = lag; }
    }
    if best_lag == 0 || best <= 0.0 {
        println!("    📏 [TEXT HEIGHT REJECT] 자기상관 최댓값이 양수가 아닙니다. 실측 행 밴드로 폴백합니다.");
        return band_measured_text_height(img, w, h);
    }

    let coherence = best / var;
    if coherence < TEXT_HEIGHT_MIN_COHERENCE {
        println!(
            "    📏 [TEXT HEIGHT REJECT] 자기상관 응집도 {:.3} < {:.2} (lag {}) — 주기가 노이즈 수준입니다. 실측 행 밴드로 폴백합니다.",
            coherence, TEXT_HEIGHT_MIN_COHERENCE, best_lag
        );
        return band_measured_text_height(img, w, h);
    }

    let th_pitch = best_lag as f32 * 0.6;
    let floor = TEXT_HEIGHT_FLOOR_PX.max(h as f32 / TEXT_HEIGHT_CROP_FRACTION);
    let ceil = h as f32 / TEXT_HEIGHT_CEIL_FRACTION;
    if th_pitch < floor || th_pitch > ceil {
        println!(
            "    📏 [TEXT HEIGHT REJECT] 추정 글자 높이 {:.1}px 가 허용 범위 {:.1}~{:.1}px 밖입니다 (크롭 높이 {}px). 자기상관이 여백 줄무늬나 표 괘선을 글자 주기로 오인한 것이므로 실측 행 밴드로 폴백합니다.",
            th_pitch, floor, ceil, h
        );
        return band_measured_text_height(img, w, h);
    }
    let mut band_h: Vec<f32> = text_row_bands(img, (0, 0, w, h))
        .iter()
        .map(|b| b.y1.saturating_sub(b.y0) as f32)
        .filter(|v| *v > 0.0)
        .collect();
    if band_h.is_empty() {
        println!(
            "    📏 [TEXT HEIGHT / PITCH ONLY] 잉크 행 밴드가 0개라 실측으로 교차 확인할 수 없습니다. 지배 주기가 말하는 높이 {:.1}px 를 그대로 채택합니다.",
            th_pitch
        );
        return Some(th_pitch);
    }
    band_h.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n_band = band_h.len();
    let low = if n_band >= TEXT_HEIGHT_MIN_BANDS { band_h[n_band / 4] } else { band_h[0] };
    let th = low.min(th_pitch).max(floor);
    crate::utils::score_dynamics::record_baseline("crop.text_height.band_ratio", low / th_pitch.max(1e-6));
    if th < th_pitch {
        println!(
            "    📏 [TEXT HEIGHT / SMALLEST BAND] 잉크 행 밴드 {}개 | 채택 기준 {} 밴드 높이 {:.1}px | 지배 주기가 말하는 높이 {:.1}px → 채택 {:.1}px (하한 {:.1}px). 배율은 가장 작은 글자가 읽힐 때까지 올려야 하므로, 다수 본문의 주기만 보면 소형 라벨이 통째로 뭉개집니다.",
            n_band,
            if n_band >= TEXT_HEIGHT_MIN_BANDS { "하위 1/4" } else { "최솟값" },
            low, th_pitch, th, floor
        );
    }
    Some(th)
}

pub fn crop_region(
    img: &image::DynamicImage,
    plan: &CropPlan,
    min_side: u32,
) -> image::DynamicImage {
    crop_region_clamped(img, plan, min_side, None, &|_| {})
}

pub fn crop_region_clamped(
    img: &image::DynamicImage,
    plan: &CropPlan,
    min_side: u32,
    height_baseline: Option<(f32, f32)>,
    emit: &dyn Fn(&str),
) -> image::DynamicImage {
    let (x0, y0, x1, y1) = plan.bbox;
    let w = x1.saturating_sub(x0).max(1);
    let h = y1.saturating_sub(y0).max(1);
    let cropped = img.crop_imm(x0, y0, w, h);

    // 🌟 [D-1] 실측 글자 높이를 '문서 기준선으로 되돌린 뒤' 배율을 계산합니다.
    //
    //  ── 순서가 중요한 이유 ──
    //   배율을 먼저 확정하고 높이만 클램프하면 로그와 실제 전송 크기가 어긋나고,
    //   창 7(44.0px → 1.00x)의 사고가 그대로 재현됩니다.
    //   클램프된 높이로 배율을 다시 계산해야 640px 급 전송이 성립합니다.
    let factor = match estimate_text_height(&cropped) {
        Some(th) if th > 0.5 => {
            let est_h = clamp_text_height(th, height_baseline, &plan.category, emit);
            let f = if est_h > 0.5 {
                (VISION_PATCH_PX / est_h).clamp(1.0, 4.0)
            } else {
                (VISION_PATCH_PX / th).clamp(1.0, 4.0)
            };
            println!(
                "    📏 [TEXT-AWARE UPSCALE] 추정 글자 높이 {:.1}px(크롭 실측 {:.1}px) → 배율 {:.2}x (목표 {:.0}px/글자)",
                est_h, th, f, VISION_PATCH_PX
            );
            f
        }
        _ => {
            // 프로파일에서 주기를 못 찾음 = 텍스트가 거의 없음.
            // 기존 짧은 변 규칙으로 폴백하되 상한을 낮게 둡니다.
            let short = w.min(h) as f32;
            let f = (min_side as f32 / short).clamp(1.0, 2.0);
            println!(
                "    📏 [TEXT-AWARE UPSCALE] 라인 주기 미검출(텍스트 희소) → 보수적 배율 {:.2}x",
                f
            );
            f
        }
    };

    let mut factor = factor;
    let cap_w = CROP_MAX_SIDE_PX as f32 / w as f32;
    let cap_h = CROP_MAX_SIDE_PX as f32 / h as f32;
    let cap = if cap_w < cap_h { cap_w } else { cap_h };
    if cap > 1.0 && factor > cap {
        println!(
            "    📐 [ISOTROPIC CAP] 크롭 {}x{} 의 배율 {:.2}x 가 긴 변 상한 {}px 를 넘어 {:.2}x 로 낮춥니다. 변마다 따로 자르면 종횡비가 깨져 글자가 한쪽으로 늘어나고, 그 왜곡은 ViT 패치 격자와 어긋나 식별 축을 담은 크롭에서 특히 손해가 큽니다.",
            w, h, factor, CROP_MAX_SIDE_PX, cap
        );
        factor = cap;
    }

    if factor <= 1.01 { return cropped; }

    let nw = ((w as f32 * factor).round() as u32).max(1);
    let nh = ((h as f32 * factor).round() as u32).max(1);
    cropped.resize_exact(nw, nh, image::imageops::FilterType::Lanczos3)
}

pub fn crop_tile(
    image: &DynamicImage,
    plan: &CropPlan,
    tile: &TilePlan,
    target_short: u32,
) -> DynamicImage {
    let (hy0, hy1) = match tile.header_band {
        Some(h) => h,
        None => {
            let mut p = plan.clone();
            p.bbox = tile.bbox;
            return crop_region(image, &p, target_short);
        }
    };

    let (x0, ry0, x1, ry1) = tile.bbox;
    let w = x1.saturating_sub(x0).max(1);
    let hh = hy1.saturating_sub(hy0).max(1);
    let rh = ry1.saturating_sub(ry0).max(1);
    let total_h = hh + HEADER_STITCH_SEAM_PX + rh;

    let head = image.crop_imm(x0, hy0, w, hh).to_rgb8();
    let body = image.crop_imm(x0, ry0, w, rh).to_rgb8();

    let mut canvas = image::RgbImage::new(w, total_h);
    for px in canvas.pixels_mut() {
        *px = image::Rgb([255u8, 255u8, 255u8]);
    }
    let hw = head.width().min(w);
    let hd = head.height().min(hh);
    for y in 0..hd {
        for x in 0..hw {
            canvas.put_pixel(x, y, *head.get_pixel(x, y));
        }
    }
    let bw = body.width().min(w);
    let bd = body.height().min(rh);
    for y in 0..bd {
        for x in 0..bw {
            canvas.put_pixel(x, hh + HEADER_STITCH_SEAM_PX + y, *body.get_pixel(x, y));
        }
    }
    let stitched = DynamicImage::ImageRgb8(canvas);

    let mut factor = if tile.text_h > 0.5 {
        (VISION_PATCH_PX / tile.text_h).clamp(1.0, 4.0)
    } else {
        let short = w.min(total_h) as f32;
        (target_short as f32 / short).clamp(1.0, 2.0)
    };
    let cap_w = CROP_MAX_SIDE_PX as f32 / w as f32;
    let cap_h = CROP_MAX_SIDE_PX as f32 / total_h as f32;
    let cap = if cap_w < cap_h { cap_w } else { cap_h };
    if cap > 1.0 && factor > cap {
        factor = cap;
    }

    println!(
        "    🧷 [HEADER STITCH] 표 헤더 y{}~{} ({}px) 를 데이터 행 y{}~{} ({}px) 위에 붙여 {}x{} 합성 크롭을 만들었습니다. 추정 글자 높이 {:.1}px → 배율 {:.2}x (종횡비 유지)",
        hy0, hy1, hh, ry0, ry1, rh, w, total_h, tile.text_h, factor
    );

    if factor <= 1.01 {
        return stitched;
    }
    let nw = ((w as f32 * factor).round() as u32).max(1);
    let nh = ((total_h as f32 * factor).round() as u32).max(1);
    stitched.resize_exact(nw, nh, image::imageops::FilterType::Lanczos3)
}

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
            owned_patches: 0,
            twin_of: String::new(),
        })
        .collect()
}

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
            None => return (f32::MIN, f32::MIN),
        };
        let m = n.min(hm.scores.len());
        let live: Vec<usize> = (0..m).filter(|&i| hm.scores[i].is_finite()).collect();
        if live.len() < 2 {
            return (f32::MIN, f32::MIN);
        }
        let mean: f32 =
            live.iter().map(|&i| hm.scores[i]).sum::<f32>() / live.len() as f32;
        let var: f32 = live
            .iter()
            .map(|&i| (hm.scores[i] - mean) * (hm.scores[i] - mean))
            .sum::<f32>()
            / live.len() as f32;
        let std = var.sqrt().max(1e-6);

        let (mut mx_in, mut mx_out) = (f32::MIN, f32::MIN);
        let (mut n_in, mut n_out) = (0usize, 0usize);
        for &i in live.iter() {
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
    let mut swap_blocked = 0usize;
    for ai in 0..plans.len() {
        if swapped[ai] { continue; }
        for bi in (ai + 1)..plans.len() {
            if swapped[bi] { continue; }

            let a_here = score_pair(&plans[ai].category, plans[ai].bbox).0;
            let a_there = score_pair(&plans[ai].category, plans[bi].bbox).0;
            let b_here = score_pair(&plans[bi].category, plans[bi].bbox).0;
            let b_there = score_pair(&plans[bi].category, plans[ai].bbox).0;

            if a_there == f32::MIN || b_there == f32::MIN {
                swap_blocked += 1;
                continue;
            }

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

    if swap_blocked > 0 {
        emit(&format!(
            "    ⏭ [CROP SWAP SKIP] 교환 후보 {}쌍이 상대 영역에서 유한 점수를 갖지 못했습니다. arena 가 이미 패치를 배타 배정했으므로 교환은 구조적으로 성립하지 않습니다 — 잘못 착지한 크롭은 아래 재배정이 처리합니다.",
            swap_blocked
        ));
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
    pub header_band: Option<(u32, u32)>,
    pub text_h: f32,
}

/// 세로 방향 겹침 분할. 무역 서식의 표는 가로로 넓고 세로로 쌓이므로
/// 세로 분할이 행 손실을 최소화합니다.
const ROW_BAND_MIN_H: u32 = 4;
const ROW_BAND_MERGE_GAP: u32 = 4;
const ROW_BAND_NOISE_SIGMA: f32 = 3.0;
const ROW_TILE_PAD_PX: u32 = 6;
const ROW_TILE_MAX: usize = 6;
const ROW_TILE_MIN_BANDS: usize = 2;
const TABLE_BAND_MIN_COLS: usize = 4;
const TABLE_BAND_CLUSTER_TOL: usize = 1;
const COL_CLUSTER_GAP: u32 = 6;
const COL_CLUSTER_MIN_W: u32 = 4;
const HEADER_STITCH_SEAM_PX: u32 = 2;
const CROP_MAX_SIDE_PX: u32 = 2048;

#[derive(Debug, Clone, Copy)]
pub struct RowBand {
    pub y0: u32,
    pub y1: u32,
    pub col_clusters: usize,
}

pub fn text_row_bands(img: &DynamicImage, bbox: (u32, u32, u32, u32)) -> Vec<RowBand> {
    use image::GenericImageView;
    let (x0, y0, x1, y1) = bbox;
    let w = x1.saturating_sub(x0).max(1);
    let h = y1.saturating_sub(y0).max(1);
    if h < ROW_BAND_MIN_H * 2 {
        return Vec::new();
    }
    let g = img.crop_imm(x0, y0, w, h).to_luma8();
    let (gw, gh) = g.dimensions();

    let mut prof: Vec<f32> = Vec::with_capacity(gh as usize);
    for y in 0..gh {
        let mut s = 0.0f32;
        for x in 0..gw {
            s += 255.0 - g.get_pixel(x, y)[0] as f32;
        }
        prof.push(s / gw as f32);
    }

    let mut sorted = prof.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let floor = sorted[sorted.len() / 10];
    let peak = sorted[sorted.len() * 9 / 10];
    if peak - floor < 1.0 {
        return Vec::new();
    }
    let span_gate = floor + (peak - floor) * 0.25;
    let half = &sorted[..(sorted.len() / 2).max(1)];
    let hm: f32 = half.iter().sum::<f32>() / half.len() as f32;
    let hsd: f32 = (half.iter().map(|v| (v - hm) * (v - hm)).sum::<f32>() / half.len() as f32).sqrt();
    let noise_gate = hm + hsd * ROW_BAND_NOISE_SIGMA;
    let gate = if noise_gate < span_gate { noise_gate } else { span_gate };

    let mut runs: Vec<(u32, u32)> = Vec::new();
    let mut start: Option<u32> = None;
    for y in 0..gh {
        if prof[y as usize] > gate {
            if start.is_none() {
                start = Some(y);
            }
        } else if let Some(s) = start.take() {
            runs.push((s, y));
        }
    }
    if let Some(s) = start {
        runs.push((s, gh));
    }

    let mut merged: Vec<(u32, u32)> = Vec::new();
    for (bs, be) in runs.into_iter() {
        match merged.last_mut() {
            Some(last) if bs.saturating_sub(last.1) <= ROW_BAND_MERGE_GAP => {
                last.1 = be;
            }
            _ => merged.push((bs, be)),
        }
    }
    merged.retain(|(bs, be)| be.saturating_sub(*bs) >= ROW_BAND_MIN_H);

    merged
        .into_iter()
        .map(|(bs, be)| {
            let span = (be - bs).max(1) as f32;
            let mut colp: Vec<f32> = Vec::with_capacity(gw as usize);
            for x in 0..gw {
                let mut s = 0.0f32;
                for y in bs..be {
                    s += 255.0 - g.get_pixel(x, y)[0] as f32;
                }
                colp.push(s / span);
            }
            let mut cs = colp.clone();
            cs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let cfloor = cs[cs.len() / 10];
            let cpeak = cs[cs.len() * 9 / 10];
            let cgate = cfloor + (cpeak - cfloor).max(1.0) * 0.30;

            let mut cruns: Vec<(u32, u32)> = Vec::new();
            let mut cst: Option<u32> = None;
            for x in 0..gw {
                if colp[x as usize] > cgate {
                    if cst.is_none() {
                        cst = Some(x);
                    }
                } else if let Some(s) = cst.take() {
                    cruns.push((s, x));
                }
            }
            if let Some(s) = cst {
                cruns.push((s, gw));
            }

            let mut cmerged: Vec<(u32, u32)> = Vec::new();
            for (s, e) in cruns.into_iter() {
                match cmerged.last_mut() {
                    Some(last) if s.saturating_sub(last.1) <= COL_CLUSTER_GAP => {
                        last.1 = e;
                    }
                    _ => cmerged.push((s, e)),
                }
            }
            let clusters = cmerged
                .iter()
                .filter(|(s, e)| e.saturating_sub(*s) >= COL_CLUSTER_MIN_W)
                .count();

            RowBand {
                y0: y0 + bs,
                y1: y0 + be,
                col_clusters: clusters,
            }
        })
        .collect()
}

pub fn plan_row_tiles(
    img: &DynamicImage,
    bbox: (u32, u32, u32, u32),
    emit: &dyn Fn(&str),
) -> Option<Vec<TilePlan>> {
    let (x0, y0, x1, y1) = bbox;
    let bands = text_row_bands(img, bbox);
    if bands.len() < ROW_TILE_MIN_BANDS {
        emit(&format!(
            "    ⏭ [ROW TILE SKIP] 크롭 px({},{})-({},{}) 안에서 잉크 행 밴드를 {}개밖에 못 찾았습니다. 균등 분할로 되돌립니다.",
            x0, y0, x1, y1, bands.len()
        ));
        return None;
    }

    let table: Vec<RowBand> = bands
        .iter()
        .copied()
        .filter(|b| b.col_clusters >= TABLE_BAND_MIN_COLS)
        .collect();

    let diff = |a: usize, b: usize| -> usize { if a > b { a - b } else { b - a } };

    let picked: Vec<RowBand> = if table.len() >= ROW_TILE_MIN_BANDS {
        let mut mode_cols = 0usize;
        let mut mode_hits = 0usize;
        for b in table.iter() {
            let hits = table
                .iter()
                .filter(|o| diff(o.col_clusters, b.col_clusters) <= TABLE_BAND_CLUSTER_TOL)
                .count();
            if hits > mode_hits || (hits == mode_hits && b.col_clusters > mode_cols) {
                mode_hits = hits;
                mode_cols = b.col_clusters;
            }
        }
        let aligned: Vec<RowBand> = table
            .iter()
            .copied()
            .filter(|b| diff(b.col_clusters, mode_cols) <= TABLE_BAND_CLUSTER_TOL)
            .collect();
        let dropped: Vec<(u32, u32, usize)> = table
            .iter()
            .filter(|b| diff(b.col_clusters, mode_cols) > TABLE_BAND_CLUSTER_TOL)
            .map(|b| (b.y0, b.y1, b.col_clusters))
            .collect();
        if aligned.len() >= ROW_TILE_MIN_BANDS {
            emit(&format!(
                "    📏 [ROW TILE / GRID ALIGN] 잉크 행 밴드 {}개 중 열 뭉치 {}개 이상인 후보 {}개, 그중 최빈 열 수 {}±{} 로 정렬된 표 행 {}개만 남깁니다 (y {:?}). 비표 블록 {}개 제외: {:?}",
                bands.len(), TABLE_BAND_MIN_COLS, table.len(),
                mode_cols, TABLE_BAND_CLUSTER_TOL, aligned.len(),
                aligned.iter().map(|b| (b.y0, b.y1)).take(8).collect::<Vec<_>>(),
                dropped.len(),
                dropped.iter().take(6).collect::<Vec<_>>()
            ));
            aligned
        } else {
            emit(&format!(
                "    📏 [ROW TILE / TABLE BANDS] 최빈 열 수 {}±{} 로 정렬된 행이 {}개뿐이라 열 정렬 필터를 기각하고 후보 {}개를 그대로 씁니다.",
                mode_cols, TABLE_BAND_CLUSTER_TOL, aligned.len(), table.len()
            ));
            table
        }
    } else {
        emit(&format!(
            "    📏 [ROW TILE / ALL BANDS] 열 뭉치 {}개 이상인 행이 {}개뿐이라 표 행을 특정하지 못했습니다. 전체 밴드 {}개를 그대로 씁니다.",
            TABLE_BAND_MIN_COLS, table.len(), bands.len()
        ));
        bands
    };

    let header = picked[0];
    let hy0 = header.y0.saturating_sub(ROW_TILE_PAD_PX).max(y0);
    let hy1 = (header.y1 + ROW_TILE_PAD_PX).min(y1);

    let mut out: Vec<TilePlan> = Vec::new();
    if hy1 > hy0 + 1 {
        out.push(TilePlan {
            bbox: (x0, hy0, x1, hy1),
            index: 0,
            total: 0,
            header_band: None,
            text_h: (header.y1.saturating_sub(header.y0)).max(1) as f32,
        });
    }

    for b in picked.iter().skip(1) {
        if out.len() >= ROW_TILE_MAX {
            break;
        }
        let ty0 = b.y0.saturating_sub(ROW_TILE_PAD_PX).max(y0);
        let ty1 = (b.y1 + ROW_TILE_PAD_PX).min(y1);
        if ty1 <= ty0 + 1 {
            continue;
        }
        out.push(TilePlan {
            bbox: (x0, ty0, x1, ty1),
            index: out.len(),
            total: 0,
            header_band: Some((hy0, hy1)),
            text_h: (b.y1.saturating_sub(b.y0)).max(1) as f32,
        });
    }

    if out.len() < 2 {
        return None;
    }
    let total = out.len();
    for p in out.iter_mut() {
        p.total = total;
    }
    emit(&format!(
        "    📏 [ROW TILE] 크롭 px({},{})-({},{}) 를 표 행 {}개 기준으로 타일 {}개로 만들었습니다. 1번은 헤더 밴드 y{}~{} 단독, 나머지는 그 헤더를 데이터 행 위에 붙인 합성 크롭입니다. 데이터 행 단독 크롭은 열 대응 근거가 없어 2B 모델이 값을 엉뚱한 필드에 넣습니다.",
        x0, y0, x1, y1, picked.len(), total, hy0, hy1
    ));
    Some(out)
}

fn identity_band_end_row(rows: usize, band_start: usize) -> usize {
    ((rows as f32 * 0.40) as usize)
        .max(band_start + 1)
        .min(rows.saturating_sub(1))
}

pub fn identity_band_bottom_px(grid: &PatchGrid) -> u32 {
    let rows = grid.grid_rows.max(1);
    let band_end = identity_band_end_row(rows, 1);
    let ch = grid.orig_height as f32 / rows as f32;
    (((band_end + 1) as f32) * ch).round() as u32
}

pub fn table_row_evidence(
    img: &DynamicImage,
    bbox: (u32, u32, u32, u32),
) -> (usize, usize, usize, bool) {
    let bands = text_row_bands(img, bbox);
    let table = bands
        .iter()
        .filter(|b| b.col_clusters >= TABLE_BAND_MIN_COLS)
        .count();
    (table, bands.len(), TABLE_BAND_MIN_COLS, table >= ROW_TILE_MIN_BANDS)
}

pub fn plan_overlap_tiles(
    bbox: (u32, u32, u32, u32),
    tile_count: usize,
    overlap_ratio: f32,
) -> Vec<TilePlan> {
    let (x0, y0, x1, y1) = bbox;
    if tile_count <= 1 || y1 <= y0 {
        return vec![TilePlan { bbox, index: 0, total: 1, header_band: None, text_h: 0.0 }];
    }
    let h = (y1 - y0) as f32;
    // t = 타일 높이. n 타일이 겹침 r 로 전체를 덮으려면
    //   h = n*t - (n-1)*r*t  →  t = h / (n - (n-1)*r)
    let n = tile_count as f32;
    let denom = n - (n - 1.0) * overlap_ratio;
    if denom <= 0.0 {
        return vec![TilePlan { bbox, index: 0, total: 1, header_band: None, text_h: 0.0 }];
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
            header_band: None,
            text_h: 0.0,
        });
    }
    if out.is_empty() {
        out.push(TilePlan { bbox, index: 0, total: 1, header_band: None, text_h: 0.0 });
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
    table_categories: &[&str],
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
        let live: Vec<usize> = (0..m).filter(|&i| hm.scores[i].is_finite()).collect();
        if live.len() >= 2 {
            let mean: f32 =
                live.iter().map(|&i| hm.scores[i]).sum::<f32>() / live.len() as f32;
            let var: f32 = live
                .iter()
                .map(|&i| (hm.scores[i] - mean) * (hm.scores[i] - mean))
                .sum::<f32>()
                / live.len() as f32;
            let std = var.sqrt().max(1e-6);
            let (mut mx_out, mut n_out) = (f32::MIN, 0usize);
            for &i in live.iter() {
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
    let is_array_cat = table_categories.iter().any(|c| *c == plan.category.as_str());
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

    if t2 && lg * 10 < total_in {
        emit(&format!(
            "    ⚪ [TILE SKIP / SPARSE TABLE] '{}' 표행 {}개 감지되었으나 \
             판독가능 패치 {}/{} ({:.0}%) 로 분할 무의미. 1타일로 축소합니다.",
            plan.category, table_rows, lg, total_in,
            lg as f32 / total_in as f32 * 100.0
        ));
        return (1, "표행감지_판독불가".to_string());
    }

    let count = if t2 {
        ((table_rows + 1) / 2).clamp(2, 3)
    } else if t1 {
        2
    } else {
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
use image::{DynamicImage, GenericImageView};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatchLegibility {
    Blank,
    Illegible,
    Legible,
}

#[derive(Debug, Clone)]
pub struct LegibilityMap {
    pub rows: usize,
    pub cols: usize,
    pub ink: Vec<f32>,
    pub sharpness: Vec<f32>,
    pub verdict: Vec<PatchLegibility>,
    pub split_applied: bool,
    pub ink_gate: f32,
    pub sharp_gate: f32,
}

impl LegibilityMap {
    #[inline]
    pub fn at(&self, r: usize, c: usize) -> PatchLegibility {
        let i = r * self.cols + c;
        self.verdict.get(i).copied().unwrap_or(PatchLegibility::Legible)
    }

    #[inline]
    pub fn is_legible(&self, idx: usize) -> bool {
        matches!(self.verdict.get(idx), Some(PatchLegibility::Legible))
    }

    pub fn count_in_bbox(
        &self,
        bbox: (u32, u32, u32, u32),
        orig_w: u32,
        orig_h: u32,
    ) -> (usize, usize, usize) {
        let (mut lg, mut il, mut bl) = (0usize, 0usize, 0usize);
        if self.rows == 0 || self.cols == 0 || orig_w == 0 || orig_h == 0 {
            return (lg, il, bl);
        }
        let cw = orig_w as f32 / self.cols as f32;
        let ch = orig_h as f32 / self.rows as f32;
        for r in 0..self.rows {
            for c in 0..self.cols {
                let x0 = (c as f32 * cw) as u32;
                let y0 = (r as f32 * ch) as u32;
                let x1 = ((((c + 1) as f32 * cw).ceil()) as u32).max(x0 + 1);
                let y1 = ((((r + 1) as f32 * ch).ceil()) as u32).max(y0 + 1);
                if x1 <= bbox.0 || x0 >= bbox.2 || y1 <= bbox.1 || y0 >= bbox.3 {
                    continue;
                }
                match self.at(r, c) {
                    PatchLegibility::Legible => lg += 1,
                    PatchLegibility::Illegible => il += 1,
                    PatchLegibility::Blank => bl += 1,
                }
            }
        }
        (lg, il, bl)
    }
}

fn otsu_threshold(values: &[f32]) -> (f32, f32) {
    if values.len() < 8 {
        return (0.0, 0.0);
    }
    let lo = values.iter().cloned().fold(f32::MAX, f32::min);
    let hi = values.iter().cloned().fold(f32::MIN, f32::max);
    if !(hi > lo) {
        return (lo, 0.0);
    }

    const BINS: usize = 64;
    let mut hist = [0usize; BINS];
    for v in values {
        let t = ((v - lo) / (hi - lo) * (BINS as f32 - 1.0)).round() as usize;
        hist[t.min(BINS - 1)] += 1;
    }

    let n = values.len() as f32;
    let bin_val = |b: usize| lo + (hi - lo) * (b as f32) / (BINS as f32 - 1.0);

    let total_mean: f32 = (0..BINS).map(|b| bin_val(b) * hist[b] as f32).sum::<f32>() / n;
    let total_var: f32 = (0..BINS)
        .map(|b| {
            let d = bin_val(b) - total_mean;
            d * d * hist[b] as f32
        })
        .sum::<f32>()
        / n;
    if total_var <= 0.0 {
        return (total_mean, 0.0);
    }

    let mut w0 = 0.0f32;
    let mut sum0 = 0.0f32;
    let mut best_t = lo;
    let mut best_between = 0.0f32;

    for b in 0..BINS - 1 {
        w0 += hist[b] as f32 / n;
        sum0 += bin_val(b) * hist[b] as f32 / n;
        let w1 = 1.0 - w0;
        if w0 <= 0.0 || w1 <= 0.0 {
            continue;
        }
        let m0 = sum0 / w0;
        let m1 = (total_mean - sum0) / w1;
        let between = w0 * w1 * (m0 - m1) * (m0 - m1);
        if between > best_between {
            best_between = between;
            best_t = bin_val(b);
        }
    }

    (best_t, best_between / total_var)
}

pub fn build_legibility_map(
    img: &DynamicImage,
    grid_rows: usize,
    grid_cols: usize,
    emit: &dyn Fn(&str),
) -> LegibilityMap {
    let (w, h) = img.dimensions();
    let n = grid_rows * grid_cols;
    let mut ink = vec![0.0f32; n];
    let mut sharpness = vec![0.0f32; n];

    if w == 0 || h == 0 || n == 0 {
        return LegibilityMap {
            rows: grid_rows,
            cols: grid_cols,
            ink,
            sharpness,
            verdict: vec![PatchLegibility::Legible; n],
            split_applied: false,
            ink_gate: 0.0,
            sharp_gate: 0.0,
        };
    }

    let gray = img.to_luma8();
    let cw = w as f32 / grid_cols as f32;
    let ch = h as f32 / grid_rows as f32;

    for r in 0..grid_rows {
        for c in 0..grid_cols {
            let x0 = (c as f32 * cw).floor() as u32;
            let y0 = (r as f32 * ch).floor() as u32;
            let x1 = (((c + 1) as f32 * cw).ceil() as u32).min(w);
            let y1 = (((r + 1) as f32 * ch).ceil() as u32).min(h);
            if x1 <= x0 + 1 || y1 <= y0 + 1 {
                continue;
            }

            let mut sum = 0.0f64;
            let mut sq = 0.0f64;
            let mut cnt = 0.0f64;
            let mut grad = 0.0f64;
            let mut gcnt = 0.0f64;

            for y in y0..y1 {
                for x in x0..x1 {
                    let v = gray.get_pixel(x, y)[0] as f64;
                    sum += v;
                    sq += v * v;
                    cnt += 1.0;
                    if x + 1 < x1 {
                        grad += (gray.get_pixel(x + 1, y)[0] as f64 - v).abs();
                        gcnt += 1.0;
                    }
                    if y + 1 < y1 {
                        grad += (gray.get_pixel(x, y + 1)[0] as f64 - v).abs();
                        gcnt += 1.0;
                    }
                }
            }

            if cnt < 4.0 || gcnt < 1.0 {
                continue;
            }
            let mean = sum / cnt;
            let var = (sq / cnt - mean * mean).max(0.0);
            let std = var.sqrt() as f32;
            let g = (grad / gcnt) as f32;

            let idx = r * grid_cols + c;
            ink[idx] = std;
            sharpness[idx] = g / (std + 1e-3);
        }
    }

    // ── ① 여백 분리 ──
    let (ink_gate, ink_sep) = otsu_threshold(&ink);
    let inked: Vec<usize> = (0..n)
        .filter(|&i| ink_sep > 0.0 && ink[i] > ink_gate)
        .collect();
    let inked = if inked.is_empty() {
        (0..n).collect::<Vec<usize>>()
    } else {
        inked
    };

    // ── ② 판독불가 분리 ──
    let sharp_vals: Vec<f32> = inked.iter().map(|&i| sharpness[i]).collect();
    let (sharp_gate, sharp_sep) = otsu_threshold(&sharp_vals);
    let low_count = sharp_vals.iter().filter(|v| **v <= sharp_gate).count();
    // 문서는 대부분 판독 가능합니다. 저품질 클래스가 과반이면 분리를 기각합니다.
    let split_applied = sharp_sep > 0.0 && low_count * 2 < sharp_vals.len();

    let mut verdict = vec![PatchLegibility::Legible; n];
    for i in 0..n {
        if ink_sep > 0.0 && ink[i] <= ink_gate {
            verdict[i] = PatchLegibility::Blank;
        } else if split_applied && sharpness[i] <= sharp_gate {
            verdict[i] = PatchLegibility::Illegible;
        }
    }

    let bl = verdict.iter().filter(|v| **v == PatchLegibility::Blank).count();
    let il = verdict.iter().filter(|v| **v == PatchLegibility::Illegible).count();
    emit(&format!(
        "  🔎 [LEGIBILITY] 여백 {} | 판독불가 {} | 판독가능 {} / {} | ink_gate {:.2} | sharp_gate {:.3} | 분리 {}",
        bl,
        il,
        n - bl - il,
        n,
        ink_gate,
        sharp_gate,
        if split_applied { "성공" } else { "기각(단봉)" }
    ));

    LegibilityMap {
        rows: grid_rows,
        cols: grid_cols,
        ink,
        sharpness,
        verdict,
        split_applied,
        ink_gate,
        sharp_gate,
    }
}
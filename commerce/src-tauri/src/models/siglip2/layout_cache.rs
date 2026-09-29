use base64::Engine;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;
use std::sync::RwLock;

use super::legibility::LegibilityMap;
use super::vision_encoder::{CategoryHeatmap, DocTypeVerdict, PatchGrid};

const LAYOUT_RECIPE: &str = "layout-v1:siglip2-so400m-naflex/label-anchor-int8/raw-heatmap";
const MAX_TEMPLATES: usize = 64;
const ANCHOR_MAX: usize = 48;
const ANCHOR_MIN: usize = 6;
const PROBE_MEDIAN_FLOOR: f32 = 0.80;
const PROBE_WIN_FLOOR: f32 = 0.70;
const PROBE_WIN_MARGIN: f32 = 0.02;
const PROBE_INK_FLOOR: f32 = 0.50;
const AGREE_FLOOR: f32 = 0.85;
const TRUST_AGREEMENTS: u32 = 3;
const AUDIT_EVERY: u32 = 10;
const RETIRE_DEMOTIONS: u32 = 3;
const PEAK_RADIUS: usize = 1;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CachedVerdict {
    group: String,
    group_score: f32,
    group_margin: f32,
    code: String,
    code_score: f32,
    code_margin: f32,
    prejudice_dropped: usize,
    code_candidates: Vec<(String, f32)>,
    title_confirmed: bool,
    title_text: String,
    title_band: Vec<(String, f32)>,
    conflict_candidates: Vec<(String, f32)>,
}

impl CachedVerdict {
    fn of(v: &DocTypeVerdict) -> Self {
        Self {
            group: v.group.clone(),
            group_score: finite(v.group_score),
            group_margin: finite(v.group_margin),
            code: v.code.clone(),
            code_score: finite(v.code_score),
            code_margin: finite(v.code_margin),
            prejudice_dropped: v.prejudice_dropped,
            code_candidates: v.code_candidates.iter().map(|(c, s)| (c.clone(), finite(*s))).collect(),
            title_confirmed: v.title_confirmed,
            title_text: v.title_text.clone(),
            title_band: v.title_band.iter().map(|(c, s)| (c.clone(), finite(*s))).collect(),
            conflict_candidates: v.conflict_candidates.iter().map(|(c, s)| (c.clone(), finite(*s))).collect(),
        }
    }

    fn verdict(&self) -> DocTypeVerdict {
        DocTypeVerdict {
            group: self.group.clone(),
            group_score: self.group_score,
            group_margin: self.group_margin,
            code: self.code.clone(),
            code_score: self.code_score,
            code_margin: self.code_margin,
            prejudice_dropped: self.prejudice_dropped,
            code_candidates: self.code_candidates.clone(),
            title_confirmed: self.title_confirmed,
            title_text: self.title_text.clone(),
            title_band: self.title_band.clone(),
            conflict_candidates: self.conflict_candidates.clone(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct CachedHeatmap {
    category: String,
    scores: Vec<f32>,
    top_field: String,
    top_score: f32,
    territory: usize,
    mean_margin: f32,
    top_rival: String,
    absent: bool,
    absent_reason: String,
    field_peaks: Vec<(String, usize, f32)>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Anchor {
    idx: usize,
    scale: f32,
    q: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct YieldStat {
    n: u64,
    mean: f64,
    m2: f64,
}

impl YieldStat {
    fn push(&mut self, x: f64) {
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / self.n as f64;
        self.m2 += d * (x - self.mean);
    }

    fn sd(&self) -> f64 {
        if self.n < 2 {
            0.0
        } else {
            (self.m2 / (self.n - 1) as f64).sqrt()
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct LayoutTemplate {
    id: String,
    route_trade: bool,
    detected_type: String,
    grid_rows: usize,
    grid_cols: usize,
    aspect: f32,
    anchors: Vec<Anchor>,
    legible: Vec<bool>,
    verdict: Option<CachedVerdict>,
    heatmaps: Vec<CachedHeatmap>,
    observations: u32,
    agreements: u32,
    trusted: bool,
    reuse_count: u32,
    demotions: u32,
    retired: bool,
    yields: YieldStat,
    created_ms: i64,
    last_ms: i64,
}

#[derive(Serialize, Deserialize, Default)]
struct LayoutFile {
    recipe: String,
    templates: Vec<LayoutTemplate>,
}

pub struct LayoutHit {
    pub template_id: String,
    pub trusted: bool,
    pub audit: bool,
    pub median: f32,
    pub win_rate: f32,
    pub ink_overlap: f32,
    pub route_trade: bool,
    pub detected_type: String,
    verdict: Option<CachedVerdict>,
    heatmaps: Vec<CachedHeatmap>,
    grid_cols: usize,
}

impl LayoutHit {
    pub fn reuse(&self) -> bool {
        self.trusted && !self.audit
    }

    pub fn verdict(&self) -> Option<DocTypeVerdict> {
        self.verdict.as_ref().map(|v| v.verdict())
    }

    pub fn heatmaps_for(&self, detected_type: &str) -> Option<Vec<CategoryHeatmap>> {
        if self.detected_type != detected_type || self.heatmaps.is_empty() {
            return None;
        }
        Some(self.heatmaps.iter().map(restore).collect())
    }
}

fn finite(x: f32) -> f32 {
    if x.is_finite() { x } else { f32::MIN }
}

pub fn snapshot(heatmaps: &[CategoryHeatmap]) -> Vec<CachedHeatmap> {
    heatmaps
        .iter()
        .map(|h| CachedHeatmap {
            category: h.category.clone(),
            scores: h.scores.iter().map(|s| finite(*s)).collect(),
            top_field: h.top_field.clone(),
            top_score: finite(h.top_score),
            territory: h.territory,
            mean_margin: finite(h.mean_margin),
            top_rival: h.top_rival.clone(),
            absent: h.absent,
            absent_reason: h.absent_reason.clone(),
            field_peaks: h
                .field_peaks
                .iter()
                .map(|(f, p, s)| (f.clone(), *p, finite(*s)))
                .collect(),
        })
        .collect()
}

fn restore(c: &CachedHeatmap) -> CategoryHeatmap {
    CategoryHeatmap {
        category: c.category.clone(),
        scores: c.scores.clone(),
        top_field: c.top_field.clone(),
        top_score: c.top_score,
        territory: c.territory,
        mean_margin: c.mean_margin,
        top_rival: c.top_rival.clone(),
        absent: c.absent,
        absent_reason: c.absent_reason.clone(),
        field_peaks: c.field_peaks.clone(),
    }
}

fn quantize(idx: usize, v: &[f32]) -> Anchor {
    let scale = v.iter().fold(0.0f32, |m, x| m.max(x.abs())).max(1e-8);
    let bytes: Vec<u8> = v
        .iter()
        .map(|x| ((x / scale) * 127.0).round().clamp(-127.0, 127.0) as i8 as u8)
        .collect();
    Anchor {
        idx,
        scale,
        q: base64::engine::general_purpose::STANDARD.encode(bytes),
    }
}

fn dequant(a: &Anchor) -> Vec<f32> {
    base64::engine::general_purpose::STANDARD
        .decode(a.q.as_bytes())
        .map(|b| b.into_iter().map(|x| (x as i8) as f32 / 127.0 * a.scale).collect())
        .unwrap_or_default()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    let (mut d, mut na, mut nb) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..a.len() {
        d += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na <= 0.0 || nb <= 0.0 {
        0.0
    } else {
        d / (na.sqrt() * nb.sqrt())
    }
}

fn legible_mask(legibility: &LegibilityMap, n: usize) -> Vec<bool> {
    (0..n).map(|i| legibility.is_legible(i)).collect()
}

fn ink_overlap(a: &[bool], b: &[bool]) -> f32 {
    let (mut inter, mut uni) = (0usize, 0usize);
    for i in 0..a.len().min(b.len()) {
        if a[i] || b[i] {
            uni += 1;
            if a[i] && b[i] {
                inter += 1;
            }
        }
    }
    if uni == 0 { 0.0 } else { inter as f32 / uni as f32 }
}

fn align_score(t: &LayoutTemplate, grid: &PatchGrid) -> (f32, f32) {
    let rows = grid.grid_rows as i64;
    let cols = grid.grid_cols as i64;
    let mut aligned: Vec<f32> = Vec::with_capacity(t.anchors.len());
    let mut wins = 0usize;
    for a in t.anchors.iter() {
        if a.idx >= grid.patches.len() {
            continue;
        }
        let v = dequant(a);
        if v.len() != grid.patches[a.idx].len() {
            continue;
        }
        let s = cosine(&v, &grid.patches[a.idx]);
        let (r, c) = (a.idx as i64 / cols, a.idx as i64 % cols);
        let mut rival = f32::MIN;
        for (dr, dc) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
            let (rr, cc) = (r + dr, c + dc);
            if rr < 0 || cc < 0 || rr >= rows || cc >= cols {
                continue;
            }
            let sj = cosine(&v, &grid.patches[(rr * cols + cc) as usize]);
            if sj > rival {
                rival = sj;
            }
        }
        if rival == f32::MIN || s > rival + PROBE_WIN_MARGIN {
            wins += 1;
        }
        aligned.push(s);
    }
    if aligned.is_empty() {
        return (0.0, 0.0);
    }
    let mut sorted = aligned.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    (sorted[sorted.len() / 2], wins as f32 / aligned.len() as f32)
}

fn pearson(a: &[f32], b: &[f32]) -> Option<f32> {
    let pairs: Vec<(f32, f32)> = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite() && **x > -1e30 && **y > -1e30)
        .map(|(x, y)| (*x, *y))
        .collect();
    if pairs.len() < 4 {
        return None;
    }
    let n = pairs.len() as f32;
    let ma = pairs.iter().map(|p| p.0).sum::<f32>() / n;
    let mb = pairs.iter().map(|p| p.1).sum::<f32>() / n;
    let (mut sab, mut saa, mut sbb) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in pairs.iter() {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return None;
    }
    Some(sab / (saa.sqrt() * sbb.sqrt()))
}

fn heatmap_agreement(template: &[CachedHeatmap], fresh: &[CachedHeatmap], cols: usize) -> (f32, f32) {
    let cols = cols.max(1);
    let cheb = |p: usize, q: usize| -> usize {
        let (pr, pc) = (p / cols, p % cols);
        let (qr, qc) = (q / cols, q % cols);
        pr.abs_diff(qr).max(pc.abs_diff(qc))
    };
    let (mut peak_total, mut peak_hit) = (0usize, 0usize);
    let (mut corr_sum, mut corr_n) = (0.0f32, 0usize);
    for ht in template.iter() {
        let hf = match fresh.iter().find(|h| h.category == ht.category) {
            Some(h) => h,
            None => {
                peak_total += ht.field_peaks.len();
                continue;
            }
        };
        for (f, p, _) in ht.field_peaks.iter() {
            peak_total += 1;
            if hf.field_peaks.iter().any(|(g, q, _)| g == f && cheb(*p, *q) <= PEAK_RADIUS) {
                peak_hit += 1;
            }
        }
        if let Some(c) = pearson(&ht.scores, &hf.scores) {
            corr_sum += c.max(0.0);
            corr_n += 1;
        }
    }
    let peaks = if peak_total == 0 { 1.0 } else { peak_hit as f32 / peak_total as f32 };
    let corr = if corr_n == 0 { 0.0 } else { corr_sum / corr_n as f32 };
    (peaks, corr)
}

fn count_filled(v: &Value) -> usize {
    match v {
        Value::Null => 0,
        Value::Bool(_) | Value::Number(_) => 1,
        Value::String(s) => usize::from(!s.trim().is_empty()),
        Value::Array(a) => a.iter().map(count_filled).sum(),
        Value::Object(o) => o.values().map(count_filled).sum(),
    }
}

fn now_ms() -> i64 {
    chrono::Utc::now().timestamp_millis()
}

pub struct LayoutCache {
    path: PathBuf,
    file: RwLock<Option<LayoutFile>>,
}

impl LayoutCache {
    fn ensure_loaded(&self) {
        if self.file.read().map(|g| g.is_some()).unwrap_or(true) {
            return;
        }
        let mut g = match self.file.write() {
            Ok(g) => g,
            Err(_) => return,
        };
        if g.is_some() {
            return;
        }
        let loaded = std::fs::read_to_string(&self.path)
            .ok()
            .and_then(|s| serde_json::from_str::<LayoutFile>(&s).ok());
        let file = match loaded {
            Some(f) if f.recipe == LAYOUT_RECIPE => {
                println!(
                    "[LAYOUT CACHE] 서식 템플릿 {}개를 복원했습니다. (신뢰 {}개) | {:?}",
                    f.templates.len(),
                    f.templates.iter().filter(|t| t.trusted).count(),
                    self.path
                );
                f
            }
            Some(_) => {
                println!("[LAYOUT CACHE] 템플릿 형식이 현재 판정 방식과 달라 폐기하고 새로 쌓습니다.");
                let _ = std::fs::remove_file(&self.path);
                LayoutFile { recipe: LAYOUT_RECIPE.to_string(), templates: Vec::new() }
            }
            None => LayoutFile { recipe: LAYOUT_RECIPE.to_string(), templates: Vec::new() },
        };
        *g = Some(file);
    }

    fn persist(&self, file: &LayoutFile) {
        let text = match serde_json::to_string(file) {
            Ok(t) => t,
            Err(e) => {
                println!("[LAYOUT CACHE] 직렬화 실패(메모리 캐시는 유지): {}", e);
                return;
            }
        };
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let tmp = self.path.with_extension("json.tmp");
        if std::fs::write(&tmp, text).is_ok() {
            let _ = std::fs::remove_file(&self.path);
            let _ = std::fs::rename(&tmp, &self.path);
        }
    }

    pub fn probe(
        &self,
        grid: &PatchGrid,
        legibility: &LegibilityMap,
        allow_commerce: bool,
        emit: &dyn Fn(&str),
    ) -> Option<LayoutHit> {
        self.ensure_loaded();
        let mut g = self.file.write().ok()?;
        let file = g.as_mut()?;
        if file.templates.is_empty() {
            return None;
        }
        let n = grid.grid_rows * grid.grid_cols;
        let mask = legible_mask(legibility, n);
        let aspect = grid.orig_width as f32 / grid.orig_height.max(1) as f32;
        let mut best: Option<(usize, f32, f32, f32)> = None;
        let mut near: Option<(f32, f32, f32)> = None;
        for (ti, t) in file.templates.iter().enumerate() {
            if t.retired || (!allow_commerce && !t.route_trade) {
                continue;
            }
            if t.grid_rows != grid.grid_rows || t.grid_cols != grid.grid_cols {
                continue;
            }
            if (t.aspect - aspect).abs() > 0.03 * aspect.max(0.1) {
                continue;
            }
            let (median, win) = align_score(t, grid);
            let ink = ink_overlap(&t.legible, &mask);
            if near.map_or(true, |(m, _, _)| median > m) {
                near = Some((median, win, ink));
            }
            if median >= PROBE_MEDIAN_FLOOR && win >= PROBE_WIN_FLOOR && ink >= PROBE_INK_FLOOR {
                if best.map_or(true, |(_, m, _, _)| median > m) {
                    best = Some((ti, median, win, ink));
                }
            }
        }
        if let Some((m, w, k)) = near {
            crate::utils::score_dynamics::record_baseline("vision.layout_probe_median", m);
            crate::utils::score_dynamics::record_baseline("vision.layout_probe_win", w);
            if best.is_none() {
                emit(&format!(
                    "  🗂️ [LAYOUT CACHE / MISS] 같은 격자 크기의 서식 템플릿 중 가장 가까운 것도 라벨 앵커 정렬 코사인 중앙값 {:.3} (기준 {:.2}) · 제자리 우세율 {:.2} (기준 {:.2}) · 잉크 겹침 {:.2} (기준 {:.2}) 로 같은 서식이 아닙니다. 전체 분석을 수행합니다.",
                    m, PROBE_MEDIAN_FLOOR, w, PROBE_WIN_FLOOR, k, PROBE_INK_FLOOR
                ));
            }
        }
        let (ti, median, win, ink) = best?;
        let t = &mut file.templates[ti];
        t.last_ms = now_ms();
        let audit = if t.trusted {
            t.reuse_count += 1;
            t.reuse_count % AUDIT_EVERY == 0
        } else {
            false
        };
        let hit = LayoutHit {
            template_id: t.id.clone(),
            trusted: t.trusted,
            audit,
            median,
            win_rate: win,
            ink_overlap: ink,
            route_trade: t.route_trade,
            detected_type: t.detected_type.clone(),
            verdict: t.verdict.clone(),
            heatmaps: t.heatmaps.clone(),
            grid_cols: t.grid_cols,
        };
        let mode = if hit.reuse() {
            "재사용 — 서식 판정·열 히트맵 계산을 건너뜁니다"
        } else if audit {
            "정기 감사 — 이번 문서는 전체 분석을 돌려 템플릿과 대조합니다"
        } else {
            "관측 — 전체 분석을 돌리고 결과가 템플릿과 같은지만 기록합니다"
        };
        emit(&format!(
            "  🗂️ [LAYOUT CACHE / HIT] 템플릿 '{}' ({} · {}) | 라벨 앵커 {}개 정렬 코사인 중앙값 {:.3} · 제자리 우세율 {:.2} · 잉크 겹침 {:.2} | 합치 {}/{} · {}",
            t.id,
            if t.route_trade { "trade" } else { "commerce" },
            t.detected_type,
            t.anchors.len(),
            median,
            win,
            ink,
            t.agreements,
            TRUST_AGREEMENTS,
            mode
        ));
        Some(hit)
    }

    pub fn observe(
        &self,
        hit: &LayoutHit,
        detected_type: &str,
        fresh: &[CachedHeatmap],
        emit: &dyn Fn(&str),
    ) {
        let (peaks, corr) = heatmap_agreement(&hit.heatmaps, fresh, hit.grid_cols);
        let agreement = peaks.min(corr);
        let type_ok = hit.detected_type == detected_type;
        let agree = type_ok && agreement >= AGREE_FLOOR;
        crate::utils::score_dynamics::record_baseline("vision.layout_agreement", agreement);
        self.ensure_loaded();
        let mut g = match self.file.write() {
            Ok(g) => g,
            Err(_) => return,
        };
        let file = match g.as_mut() {
            Some(f) => f,
            None => return,
        };
        let t = match file.templates.iter_mut().find(|t| t.id == hit.template_id) {
            Some(t) => t,
            None => return,
        };
        t.observations += 1;
        let verdict_line = if agree {
            t.agreements += 1;
            if !t.trusted && t.agreements >= TRUST_AGREEMENTS {
                t.trusted = true;
                "합치 — 연속 합치가 기준에 도달해 다음 문서부터 재사용합니다"
            } else {
                "합치"
            }
        } else {
            t.agreements = 0;
            if t.trusted {
                t.trusted = false;
                t.demotions += 1;
                if t.demotions >= RETIRE_DEMOTIONS {
                    t.retired = true;
                }
            }
            if t.retired {
                "불일치 — 강등이 누적되어 이 템플릿을 은퇴시킵니다"
            } else {
                "불일치 — 신뢰를 초기화하고 전체 분석을 계속합니다"
            }
        };
        emit(&format!(
            "  🗂️ [LAYOUT CACHE / OBSERVE] 템플릿 '{}' | 서식 판정 {} ('{}' vs '{}') | 라벨 봉우리 일치 {:.2} · 히트맵 상관 {:.2} → 합치도 {:.2} (기준 {:.2}) | {} (합치 {}/{})",
            t.id,
            if type_ok { "일치" } else { "불일치" },
            hit.detected_type,
            detected_type,
            peaks,
            corr,
            agreement,
            AGREE_FLOOR,
            verdict_line,
            t.agreements,
            TRUST_AGREEMENTS
        ));
        let snapshot_file = LayoutFile { recipe: file.recipe.clone(), templates: file.templates.clone() };
        drop(g);
        self.persist(&snapshot_file);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn commit(
        &self,
        hit: Option<&LayoutHit>,
        fresh: Option<&Vec<CachedHeatmap>>,
        grid: &PatchGrid,
        legibility: &LegibilityMap,
        route_trade: bool,
        detected_type: &str,
        verdict: Option<&DocTypeVerdict>,
        extracted: &Value,
        emit: &dyn Fn(&str),
    ) {
        let filled = count_filled(extracted);
        crate::utils::score_dynamics::record_baseline("vision.layout_yield", filled as f32);
        self.ensure_loaded();
        let mut g = match self.file.write() {
            Ok(g) => g,
            Err(_) => return,
        };
        let file = match g.as_mut() {
            Some(f) => f,
            None => return,
        };
        match hit {
            Some(h) => {
                let t = match file.templates.iter_mut().find(|t| t.id == h.template_id) {
                    Some(t) => t,
                    None => return,
                };
                let (mean, sd, n) = (t.yields.mean, t.yields.sd(), t.yields.n);
                if h.reuse() {
                    let floor = if n >= 3 { mean - 2.0 * sd.max(1.0) } else { mean * 0.5 };
                    if filled == 0 || (n >= 1 && (filled as f64) < floor) {
                        t.trusted = false;
                        t.agreements = 0;
                        t.demotions += 1;
                        if t.demotions >= RETIRE_DEMOTIONS {
                            t.retired = true;
                        }
                        emit(&format!(
                            "  🗂️ [LAYOUT CACHE / DEMOTE] 템플릿 '{}' 재사용 문서의 추출 값 {}개가 이 서식의 평소 수준(평균 {:.1} · 하한 {:.1}) 에 못 미칩니다. 템플릿 신뢰를 거두고 다음 문서부터 전체 분석으로 돌아갑니다.{}",
                            t.id,
                            filled,
                            mean,
                            floor,
                            if t.retired { " 강등이 누적되어 템플릿을 은퇴시킵니다." } else { "" }
                        ));
                    }
                }
                t.yields.push(filled as f64);
                t.last_ms = now_ms();
            }
            None => {
                if filled == 0 {
                    return;
                }
                let fresh = match fresh {
                    Some(f) if !f.is_empty() => f,
                    _ => return,
                };
                let mut peaks: Vec<(usize, f32)> = Vec::new();
                for h in fresh.iter() {
                    for (_, p, s) in h.field_peaks.iter() {
                        if *p >= grid.patches.len() {
                            continue;
                        }
                        match peaks.iter_mut().find(|(q, _)| q == p) {
                            Some(e) => e.1 = e.1.max(*s),
                            None => peaks.push((*p, *s)),
                        }
                    }
                }
                peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                peaks.truncate(ANCHOR_MAX);
                if peaks.len() < ANCHOR_MIN {
                    emit(&format!(
                        "  🗂️ [LAYOUT CACHE / SKIP] 라벨 봉우리가 {}칸뿐이라 (최소 {}) 서식을 식별할 앵커가 부족합니다. 템플릿을 만들지 않습니다.",
                        peaks.len(),
                        ANCHOR_MIN
                    ));
                    return;
                }
                let now = now_ms();
                let id = format!(
                    "{}-{}x{}-{:x}",
                    detected_type,
                    grid.grid_rows,
                    grid.grid_cols,
                    now
                );
                let template = LayoutTemplate {
                    id: id.clone(),
                    route_trade,
                    detected_type: detected_type.to_string(),
                    grid_rows: grid.grid_rows,
                    grid_cols: grid.grid_cols,
                    aspect: grid.orig_width as f32 / grid.orig_height.max(1) as f32,
                    anchors: peaks.iter().map(|(p, _)| quantize(*p, &grid.patches[*p])).collect(),
                    legible: legible_mask(legibility, grid.grid_rows * grid.grid_cols),
                    verdict: verdict.map(CachedVerdict::of),
                    heatmaps: fresh.clone(),
                    observations: 0,
                    agreements: 0,
                    trusted: false,
                    reuse_count: 0,
                    demotions: 0,
                    retired: false,
                    yields: {
                        let mut y = YieldStat::default();
                        y.push(filled as f64);
                        y
                    },
                    created_ms: now,
                    last_ms: now,
                };
                file.templates.push(template);
                if file.templates.len() > MAX_TEMPLATES {
                    file.templates.sort_by(|a, b| b.last_ms.cmp(&a.last_ms));
                    file.templates.truncate(MAX_TEMPLATES);
                }
                emit(&format!(
                    "  🗂️ [LAYOUT CACHE / NEW] 서식 템플릿 '{}' 를 만들었습니다 | 라벨 앵커 {}칸 · 열 히트맵 {}개 · 추출 값 {}개. 같은 서식이 {}번 연속 같은 판정을 내면 그다음 문서부터 서식 판정·열 히트맵 계산을 건너뜁니다.",
                    id,
                    peaks.len(),
                    fresh.len(),
                    filled,
                    TRUST_AGREEMENTS
                ));
            }
        }
        let snapshot_file = LayoutFile { recipe: file.recipe.clone(), templates: file.templates.clone() };
        drop(g);
        self.persist(&snapshot_file);
    }

    pub fn clear_all(&self) {
        if let Ok(mut g) = self.file.write() {
            *g = Some(LayoutFile { recipe: LAYOUT_RECIPE.to_string(), templates: Vec::new() });
        }
        let _ = std::fs::remove_file(&self.path);
        println!("[LAYOUT CACHE] 서식 템플릿을 전량 삭제했습니다.");
    }
}

pub static LAYOUT_CACHE: Lazy<LayoutCache> = Lazy::new(|| LayoutCache {
    path: crate::utils::get_app_dir()
        .join("cache")
        .join("vision_layouts")
        .join("templates.json"),
    file: RwLock::new(None),
});
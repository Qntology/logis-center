use std::collections::{HashMap, HashSet};

use super::legibility::LegibilityMap;
use super::vision_encoder::{CategoryHeatmap, PatchGrid};

pub const ARENA_MAX_ROUNDS: usize = 4;
pub const ARENA_MARGIN_RATIO: f32 = 0.5;

#[derive(Debug, Clone)]
pub struct PatchVerdict {
    pub owner: Option<usize>,
    pub runner_up: Option<usize>,
    pub margin: f32,
}

#[derive(Debug, Clone)]
pub struct FieldTerritory {
    pub category: String,
    pub patches: Vec<usize>,
    pub legible: usize,
    pub mean_margin: f32,
    pub top_rival: String,
    pub absent: bool,
    pub absent_reason: String,
}

pub struct ArenaResult {
    pub verdicts: Vec<PatchVerdict>,
    pub territories: Vec<FieldTerritory>,
    pub rounds: usize,
    pub margin_gate: f32,
}

fn median_of(v: &mut Vec<f32>) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

pub fn compete_patches(
    heatmaps: &[CategoryHeatmap],
    active: &[bool],
    n: usize,
) -> Vec<PatchVerdict> {
    let mut out: Vec<PatchVerdict> = Vec::with_capacity(n);
    for i in 0..n {
        let mut best = f32::MIN;
        let mut best_hi: Option<usize> = None;
        let mut second = f32::MIN;
        let mut second_hi: Option<usize> = None;

        for (hi, hm) in heatmaps.iter().enumerate() {
            if !active[hi] || i >= hm.scores.len() {
                continue;
            }
            let v = hm.scores[i];
            if !v.is_finite() || v <= 0.0 {
                continue;
            }
            if v > best {
                second = best;
                second_hi = best_hi;
                best = v;
                best_hi = Some(hi);
            } else if v > second {
                second = v;
                second_hi = Some(hi);
            }
        }

        let margin = match (best_hi, second_hi) {
            (Some(_), Some(_)) => best - second,
            (Some(_), None) => best,
            _ => 0.0,
        };
        out.push(PatchVerdict {
            owner: best_hi,
            runner_up: second_hi,
            margin,
        });
    }
    out
}

fn territory_scores(
    heatmaps: &[CategoryHeatmap],
    verdicts: &[PatchVerdict],
    legibility: &LegibilityMap,
    n: usize,
) -> Vec<FieldTerritory> {
    let k = heatmaps.len();
    let mut out: Vec<FieldTerritory> = Vec::with_capacity(k);

    for hi in 0..k {
        let mut patches: Vec<usize> = Vec::new();
        let mut legible = 0usize;
        let mut msum = 0.0f32;
        let mut rivals: HashMap<usize, usize> = HashMap::new();

        for i in 0..n.min(verdicts.len()) {
            if verdicts[i].owner != Some(hi) {
                continue;
            }
            patches.push(i);
            msum += verdicts[i].margin;
            if legibility.is_legible(i) {
                legible += 1;
            }
            if let Some(r) = verdicts[i].runner_up {
                *rivals.entry(r).or_insert(0) += 1;
            }
        }

        let cnt = patches.len();
        let mean_margin = if cnt == 0 { 0.0 } else { msum / cnt as f32 };
        let mut rival_list: Vec<(usize, usize)> =
            rivals.iter().map(|(r, c)| (*r, *c)).collect();
        rival_list.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let top_rival = rival_list
            .first()
            .map(|(r, _)| heatmaps[*r].category.clone())
            .unwrap_or_default();

        out.push(FieldTerritory {
            category: heatmaps[hi].category.clone(),
            patches,
            legible,
            mean_margin,
            top_rival,
            absent: cnt == 0,
            absent_reason: String::new(),
        });
    }
    out
}

pub fn run_arena(
    heatmaps: &[CategoryHeatmap],
    grid: &PatchGrid,
    legibility: &LegibilityMap,
    protected: &[&str],
    emit: &dyn Fn(&str),
) -> ArenaResult {
    let n = grid.grid_rows * grid.grid_cols;
    let k = heatmaps.len();
    if n == 0 || k == 0 {
        return ArenaResult {
            verdicts: Vec::new(),
            territories: Vec::new(),
            rounds: 0,
            margin_gate: 0.0,
        };
    }

    let mut active = vec![true; k];
    let mut reasons: Vec<String> = vec![String::new(); k];

    let probe = compete_patches(heatmaps, &active, n);
    let mut margins: Vec<f32> = probe
        .iter()
        .filter(|v| v.owner.is_some() && v.runner_up.is_some())
        .map(|v| v.margin)
        .collect();
    let contested_n = margins.len();
    let margin_gate = if margins.len() < 4 {
        0.0
    } else {
        median_of(&mut margins) * ARENA_MARGIN_RATIO
    };

    emit(&format!(
        "    ⚔️ [ARENA] 패치 {}칸 · 카테고리 {}개로 배타 경쟁을 시작합니다. 2개 이상이 다투는 칸 {}개의 마진 중앙값 × {:.2} = 확정 임계 {:+.4}. 이 임계를 한 칸도 못 넘는 카테고리는 이 문서에 인쇄되지 않은 축으로 보고 탈락시킨 뒤, 그 칸을 남은 카테고리끼리 다시 경쟁시킵니다.",
        n, k, contested_n, ARENA_MARGIN_RATIO, margin_gate
    ));

    let mut rounds = 0usize;
    for _ in 0..ARENA_MAX_ROUNDS {
        rounds += 1;
        let v = compete_patches(heatmaps, &active, n);

        let mut firm = vec![0usize; k];
        for pv in v.iter() {
            if let Some(hi) = pv.owner {
                if pv.margin >= margin_gate {
                    firm[hi] += 1;
                }
            }
        }

        let mut removed: Vec<String> = Vec::new();
        for hi in 0..k {
            if !active[hi] || firm[hi] > 0 {
                continue;
            }
            if protected.iter().any(|c| *c == heatmaps[hi].category.as_str()) {
                emit(&format!(
                    "    🛡️ [ARENA PROTECT] '{}' 는 확정 패치가 0칸이지만 문서 기본키 또는 표 구조 축이라 탈락시키지 않습니다. 표 앵커는 여러 카테고리와 겹치도록 설계된 축이므로 경쟁만으로 지워지면 표 전체를 못 읽습니다.",
                    heatmaps[hi].category
                ));
                continue;
            }
            active[hi] = false;
            reasons[hi] = format!(
                "라운드 {} 에서 확정 임계 {:+.4} 를 넘는 패치를 한 칸도 얻지 못함",
                rounds, margin_gate
            );
            removed.push(heatmaps[hi].category.clone());
        }

        if removed.is_empty() {
            break;
        }
        emit(&format!(
            "    ⚔️ [ARENA ROUND {}] 탈락 {}개 ({}) — 이들이 쥐고 있던 칸을 남은 카테고리에 재경쟁시킵니다.",
            rounds,
            removed.len(),
            removed.join(", ")
        ));
    }

    let verdicts = compete_patches(heatmaps, &active, n);
    let mut territories = territory_scores(heatmaps, &verdicts, legibility, n);
    for hi in 0..k {
        if !active[hi] {
            territories[hi].absent = true;
            territories[hi].absent_reason = reasons[hi].clone();
            territories[hi].patches.clear();
            territories[hi].legible = 0;
        } else if territories[hi].absent {
            territories[hi].absent_reason =
                "경쟁에서 단 한 칸도 최강 설명이 되지 못함".to_string();
        }
    }

    let owned: usize = territories.iter().map(|t| t.patches.len()).sum();
    emit(&format!(
        "    ⚔️ [ARENA DONE] {}라운드 | 소유 확정 {}/{}칸 | 카테고리당 평균 {:.1}칸 (면적 상한과 같은 눈금입니다)",
        rounds,
        owned,
        n,
        owned as f32 / k.max(1) as f32
    ));
    for t in territories.iter() {
        if t.absent {
            emit(&format!(
                "    ⚪ [ARENA ABSENT] '{}' — {}",
                t.category, t.absent_reason
            ));
        } else {
            emit(&format!(
                "    🗺️ [ARENA TERRITORY] '{}' | 영토 {}칸 (판독 가능 {}) | 평균 마진 {:+.4} | 최대 경쟁자 {}",
                t.category,
                t.patches.len(),
                t.legible,
                t.mean_margin,
                if t.top_rival.is_empty() { "-" } else { &t.top_rival }
            ));
        }
    }

    ArenaResult {
        verdicts,
        territories,
        rounds,
        margin_gate,
    }
}

pub fn apply_arena(
    heatmaps: &mut Vec<CategoryHeatmap>,
    result: &ArenaResult,
    emit: &dyn Fn(&str),
) {
    if result.territories.len() != heatmaps.len() {
        emit("    ⚠️ [ARENA APPLY SKIP] 영토 수와 히트맵 수가 어긋나 마스킹을 건너뜁니다.");
        return;
    }

    let mut lines: Vec<String> = Vec::new();
    for (hi, hm) in heatmaps.iter_mut().enumerate() {
        let t = &result.territories[hi];
        let keep: HashSet<usize> = t.patches.iter().copied().collect();

        let before = hm
            .scores
            .iter()
            .filter(|s| s.is_finite() && **s > 0.0)
            .count();

        let mut top = f32::MIN;
        for i in 0..hm.scores.len() {
            if keep.contains(&i) {
                if hm.scores[i] > top {
                    top = hm.scores[i];
                }
            } else {
                hm.scores[i] = f32::MIN;
            }
        }

        let old_top = hm.top_score;
        hm.top_score = if top == f32::MIN { 0.0 } else { top };
        hm.territory = t.patches.len();
        hm.mean_margin = t.mean_margin;
        hm.top_rival = t.top_rival.clone();
        hm.absent = t.absent;
        hm.absent_reason = t.absent_reason.clone();

        lines.push(format!(
            "{}({}→{}, top {:+.4}→{:+.4})",
            hm.category,
            before,
            t.patches.len(),
            old_top,
            hm.top_score
        ));
    }

    lines.sort();
    emit(&format!(
        "    ⚔️ [ARENA APPLY] 카테고리가 자기 영토 밖에서 들고 있던 점수를 전부 내려놓았습니다. 활성 패치 변화: {}",
        lines.join(" | ")
    ));
    emit(
        "    ⚔️ [ARENA APPLY] 내용 마스크는 패치별 최댓값이라 소유자 점수가 그대로 남습니다. build_content_mask 결과와 content_gate 는 마스킹 전후가 동일합니다.",
    );
}
use candle_core::{DType, Device, Tensor};
use image::DynamicImage;

use super::preprocessor::{preprocess_image, PreprocessedImage};
use super::{Siglip2Config, Siglip2Model};
use crate::logic::TRADE_DOC_TITLES;
use crate::utils::ai_utils::{split_bias_phrases_full, surprisal_dual_scores};

pub const ERR_TEXT_ENCODER_REQUIRED: &str = "SIGLIP2_TEXT_ENCODER_REQUIRED";

pub struct AnchorBank {
    /// bias 축: 이 개념을 설명하는 구
    pub bias: Vec<(String, String, std::sync::Arc<Vec<f32>>)>,
    /// prejudice 축: 이 개념이 절대 아닌 구
    pub prejudice: Vec<(String, String, std::sync::Arc<Vec<f32>>)>,
}

impl AnchorBank {
    pub fn is_empty(&self) -> bool {
        self.bias.is_empty()
    }
}

/// L2 정규화. 코사인 계산의 전제입니다.
fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub fn encode_phrases_shared(
    model: &Siglip2Model,
    phrases: &[String],
) -> anyhow::Result<Vec<std::sync::Arc<Vec<f32>>>> {
    encode_phrases_persisted(model, phrases, true)
}

pub fn encode_phrases_ephemeral(
    model: &Siglip2Model,
    phrases: &[String],
) -> anyhow::Result<Vec<Vec<f32>>> {
    Ok(encode_phrases_persisted(model, phrases, false)?
        .into_iter()
        .map(|a| (*a).clone())
        .collect())
}

fn encode_phrases_persisted(
    model: &Siglip2Model,
    phrases: &[String],
    persist: bool,
) -> anyhow::Result<Vec<std::sync::Arc<Vec<f32>>>> {
    use crate::models::siglip2::phrase_cache::SIGLIP2_PHRASE_CACHE as CACHE;

    if phrases.is_empty() {
        return Ok(Vec::new());
    }

    // ── ① 캐시 조회 ──
    let mut slots: Vec<Option<std::sync::Arc<Vec<f32>>>> = Vec::with_capacity(phrases.len());
    let mut miss_idx: Vec<usize> = Vec::new();
    for (i, p) in phrases.iter().enumerate() {
        match CACHE.get(p) {
            Some(v) => slots.push(Some(v)),
            None => {
                slots.push(None);
                miss_idx.push(i);
            }
        }
    }

    if miss_idx.is_empty() {
        println!(
            "    ⚡ [PHRASE CACHE] 구 {}개 전량 히트 — 텍스트 인코더 순전파를 생략합니다. (약 {:.1} TFLOP 절감)",
            phrases.len(),
            phrases.len() as f64 * 26.0 / 1000.0
        );
        return Ok(slots.into_iter().map(|s| s.unwrap()).collect());
    }

    let text = model.text.as_ref().ok_or_else(|| {
        anyhow::anyhow!(
            "{}: text encoder is not loaded, and {} of {} anchor phrases are not cached.",
            ERR_TEXT_ENCODER_REQUIRED,
            miss_idx.len(),
            phrases.len()
        )
    })?;
    let tok = model
        .tokenizer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("SigLIP2 tokenizer is not loaded"))?;

    println!(
        "    🔤 [PHRASE CACHE] 구 {}개 중 {}개 히트 / {}개 신규 인코딩",
        phrases.len(),
        phrases.len() - miss_idx.len(),
        miss_idx.len()
    );

    let mut fresh: Vec<(String, std::sync::Arc<Vec<f32>>)> = Vec::with_capacity(miss_idx.len());

    // 배치 크기는 VRAM 상한을 고려해 32 로 고정합니다.
    // (64토큰 × 1152차원 × 27층이므로 32가 안전선입니다)
    for chunk in miss_idx.chunks(32) {
        let owned: Vec<String> = chunk.iter().map(|&i| phrases[i].clone()).collect();
        let batch = tok.encode_batch(&owned)?;
        let t = text.encode_batch(&batch, &model.device)?; // (b, D)
        let t = t.to_dtype(DType::F32)?;
        let rows: Vec<Vec<f32>> = t.to_vec2::<f32>()?;
        for (bi, mut r) in rows.into_iter().enumerate() {
            l2_normalize(&mut r);
            let arc = std::sync::Arc::new(r);
            let gi = chunk[bi];
            slots[gi] = Some(arc.clone());
            fresh.push((phrases[gi].clone(), arc));
        }
    }

    if persist {
        CACHE.put_batch(&fresh);
    } else {
        println!(
            "    🫧 [PHRASE EPHEMERAL] 새로 인코딩한 구 {}개를 캐시에 적재하지 않고 버립니다. 이 구들은 이 문서에서만 등장하는 값 문자열이라 다음 문서에서 다시 맞을 일이 없는데, 앵커 구와 같은 저장소를 쓰면 상한을 값이 잠식해 앵커 히트율이 무너집니다.",
            fresh.len()
        );
    }

    Ok(slots
        .into_iter()
        .map(|s| s.unwrap_or_else(|| std::sync::Arc::new(vec![0.0f32; model.config.text_hidden_size])))
        .collect())
}

/// 레거시 호환 래퍼. 소유 벡터가 필요한 호출부(encode_query_text)용입니다.
pub fn encode_phrases(
    model: &Siglip2Model,
    phrases: &[String],
) -> anyhow::Result<Vec<Vec<f32>>> {
    Ok(encode_phrases_shared(model, phrases)?
        .into_iter()
        .map(|a| (*a).clone())
        .collect())
}

pub fn phrases_all_cached(phrases: &[String]) -> bool {
    crate::models::siglip2::phrase_cache::SIGLIP2_PHRASE_CACHE.all_cached(phrases)
}

pub fn build_anchor_bank(
    model: &Siglip2Model,
    bias_defs: &[(String, String, String)],
    prej_defs: &[(String, String, String)],
) -> anyhow::Result<AnchorBank> {
    use std::collections::HashMap;

    let mut index: HashMap<&str, usize> = HashMap::new();
    let mut uniq: Vec<String> = Vec::new();
    for (_, _, p) in bias_defs.iter().chain(prej_defs.iter()) {
        if !index.contains_key(p.as_str()) {
            index.insert(p.as_str(), uniq.len());
            uniq.push(p.clone());
        }
    }

    let shared: Vec<std::sync::Arc<Vec<f32>>> = encode_phrases_shared(model, &uniq)?;
    let zero = std::sync::Arc::new(vec![0.0f32; model.config.text_hidden_size]);

    let lookup = |p: &str| -> std::sync::Arc<Vec<f32>> {
        match index.get(p) {
            Some(&i) => shared[i].clone(), // Arc clone = 참조 카운트 증가만
            None => zero.clone(),
        }
    };

    Ok(AnchorBank {
        bias: bias_defs
            .iter()
            .map(|(c, k, p)| (c.clone(), k.clone(), lookup(p)))
            .collect(),
        prejudice: prej_defs
            .iter()
            .map(|(c, k, p)| (c.clone(), k.clone(), lookup(p)))
            .collect(),
    })
}

pub struct PatchGrid {
    pub patches: Vec<Vec<f32>>,
    pub pooled: Vec<f32>,
    pub grid_rows: usize,
    pub grid_cols: usize,
    pub scale_x: f64,
    pub scale_y: f64,
    pub orig_width: u32,
    pub orig_height: u32,
    pub patch_size: usize,
}

impl PatchGrid {
    pub fn len(&self) -> usize {
        self.patches.len()
    }

    /// 패치 인덱스 → (row, col)
    pub fn rc(&self, idx: usize) -> (usize, usize) {
        (idx / self.grid_cols, idx % self.grid_cols)
    }
}

/// 이미지 → 패치 임베딩 격자.
///
/// preprocess → vision.forward → L2 정규화 까지 한 번에 수행합니다.
pub fn encode_image(
    model: &Siglip2Model,
    image: &DynamicImage,
) -> anyhow::Result<PatchGrid> {
    let vision = model
        .vision
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!(
            "SigLIP2 vision encoder is not loaded. Call ensure_siglip2 with needs_vision=true."
        ))?;

    let pre: PreprocessedImage = preprocess_image(image, &model.config, &model.device)?;
    let px = pre.pixel_values.to_dtype(model.dtype)?;
    let out = vision.forward(&px, pre.grid_rows, pre.grid_cols)?;

    let shared = out.patch_shared.squeeze(0)?.to_dtype(DType::F32)?; // (N, D)
    let mut patches: Vec<Vec<f32>> = shared.to_vec2::<f32>()?;
    for p in patches.iter_mut() {
        l2_normalize(p);
    }

    let pooled_t = out.pooled.to_dtype(DType::F32)?; // (1, D)
    let mut pooled: Vec<f32> = pooled_t.squeeze(0)?.to_vec1::<f32>()?;
    l2_normalize(&mut pooled);

    drop(out);
    drop(shared);
    drop(pooled_t);
    drop(px);

    Ok(PatchGrid {
        patches,
        pooled,
        grid_rows: pre.grid_rows,
        grid_cols: pre.grid_cols,
        scale_x: pre.scale_x,
        scale_y: pre.scale_y,
        orig_width: pre.orig_width,
        orig_height: pre.orig_height,
        patch_size: model.config.patch_size,
    })
}

pub fn encode_image_pooled(
    model: &Siglip2Model,
    image: &DynamicImage,
) -> anyhow::Result<Vec<f32>> {
    let vision = model
        .vision
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!(
            "SigLIP2 vision encoder is not loaded. Call ensure_siglip2 with needs_vision=true."
        ))?;

    let pre = preprocess_image(image, &model.config, &model.device)?;
    let px = pre.pixel_values.to_dtype(model.dtype)?;
    let pooled_t = vision
        .forward_pooled(&px, pre.grid_rows, pre.grid_cols)?
        .to_dtype(DType::F32)?;

    let mut pooled: Vec<f32> = pooled_t.squeeze(0)?.to_vec1::<f32>()?;
    l2_normalize(&mut pooled);
    Ok(pooled)
}

pub fn encode_image_and_release(
    model: &mut Siglip2Model,
    image: &DynamicImage,
) -> anyhow::Result<PatchGrid> {
    let grid = encode_image(model, image)?;
    println!(
        "    ♻️ [VISION RELEASE] 패치 {}개({}x{}) 를 호스트로 확보했습니다. 비전 가중치를 즉시 반납합니다.",
        grid.len(),
        grid.grid_rows,
        grid.grid_cols
    );
    model.detach_vision();
    Ok(grid)
}

/// 텍스트 한 줄을 SigLIP2 공유공간 벡터로 (검색 쿼리용).
pub fn encode_query_text(model: &Siglip2Model, text: &str) -> anyhow::Result<Vec<f32>> {
    let v = encode_phrases(model, &[text.to_string()])?;
    v.into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("SigLIP2 text encoding returned nothing"))
}

#[derive(Debug, Clone)]
pub struct DocTypeVerdict {
    pub group: String,
    pub group_score: f32,
    pub group_margin: f32,
    pub code: String,
    pub code_score: f32,
    pub code_margin: f32,
    pub prejudice_dropped: usize,
    pub code_candidates: Vec<(String, f32)>,
    pub title_confirmed: bool,
    pub title_text: String,
}

fn bank_neutral_key_scores(
    grid: &PatchGrid,
    bank: &AnchorBank,
    legible: Option<&crate::models::siglip2::legibility::LegibilityMap>,
) -> (Vec<(String, f32)>, usize) {
    let (keys, matrix) = score_patches_bank_neutral(grid, bank, legible);
    let mut out: Vec<(String, f32)> = keys
        .iter()
        .enumerate()
        .map(|(ki, k)| {
            let m = matrix[ki].iter().cloned().fold(f32::MIN, f32::max);
            (k.clone(), if m == f32::MIN { 0.0 } else { m })
        })
        .collect();
    out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let active = matrix.iter().flatten().filter(|&&v| v > 0.0).count();
    (out, active)
}

fn score_patches(
    grid: &PatchGrid,
    bank: &AnchorBank,
) -> (std::collections::HashMap<String, f32>, usize) {
    use std::collections::HashMap;

    let empty_names: Vec<String> = Vec::new();
    let empty_banks: Vec<Vec<Vec<f32>>> = Vec::new();
    let empty_skip: Vec<bool> = Vec::new();

    let mut best: HashMap<String, f32> = HashMap::new();
    let mut dropped = 0usize;

    // 🌟 [LOG] 패치별 채점 상세 추적을 위한 카운터
    let mut total_scored = 0usize;
    let mut zero_vec_skipped = 0usize;
    let mut empty_scores_skipped = 0usize;
    let mut prejudice_dropped_details: Vec<(usize, String, f32)> = Vec::new();
    let mut top_contributors: Vec<(usize, String, f32, usize, usize)> = Vec::new();

    for (patch_idx, p) in grid.patches.iter().enumerate() {
        if p.iter().all(|&v| v == 0.0) {
            zero_vec_skipped += 1;
            continue;
        }

        let (scores, _) = surprisal_dual_scores(
            p,
            &bank.bias,
            &bank.prejudice,
            &empty_names,
            &empty_banks,
            &empty_skip,
        );

        if scores.is_empty() {
            empty_scores_skipped += 1;
            continue;
        }

        total_scored += 1;

        let patch_idx_for_log = {
            let mut cnt = 0usize;
            for pp in grid.patches.iter() {
                if std::ptr::eq(pp, p) { break; }
                cnt += 1;
            }
            cnt
        };
        let log_this_patch = patch_idx_for_log < 12;

        if log_this_patch && !scores.is_empty() {
            let (r, c) = grid.rc(patch_idx_for_log);
            let top3: Vec<String> = scores.iter().take(3)
                .map(|s| format!("{}(cos:{:.4} sur:{:+.4} N:{})", s.key, s.max_cos, s.surprisal, s.n))
                .collect();
            println!("    🔬 [PATCH-COSINE r{}c{}] {}", r, c, top3.join(" | "));
        }

        // surprisal_dual_scores 는 이미 prejudice 를 상쇄해 반환합니다.
        // 최상위 점수가 0 이하라면 이 패치는 어떤 개념과도 무관합니다.
        if scores[0].surprisal <= 0.0 {
            dropped += 1;
            // 🌟 [LOG] 편견 우세로 탈락한 패치의 상세 (샘플)
            if log_this_patch {
                let (r, c) = grid.rc(patch_idx_for_log);
                println!(
                    "    🚫 [PREJUDICE r{}c{}] top: {} cos:{:.4} sur:{:+.4} ≤ 0 → 편견 우세 탈락",
                    r, c, scores[0].key, scores[0].max_cos, scores[0].surprisal
                );
            }
            // 🌟 [LOG] 편견 탈락 샘플 수집 (상위 20개)
            if prejudice_dropped_details.len() < 20 {
                let (r, c) = grid.rc(patch_idx);
                prejudice_dropped_details.push((patch_idx, format!("r{}c{} {}", r, c, scores[0].key), scores[0].surprisal));
            }
            continue;
        }

        for s in scores {
            let e = best.entry(s.key.clone()).or_insert(f32::MIN);
            if s.surprisal > *e {
                // 🌟 [LOG] 키별 최고점 갱신 시 원시 코사인 기록 (샘플)
                if log_this_patch {
                    let (r, c) = grid.rc(patch_idx_for_log);
                    println!(
                        "    ⬆️ [KEY-UPDATE r{}c{}] '{}' cos:{:.4} sur:{:+.4} (이전 {:+.4})",
                        r, c, s.key, s.max_cos, s.surprisal, *e
                    );
                }
                // 🌟 [LOG] 키별 최고점 갱신 패치 수집 (상위 30개)
                if top_contributors.len() < 30 {
                    let (r, c) = grid.rc(patch_idx);
                    top_contributors.push((patch_idx, s.key.clone(), s.surprisal, r, c));
                }
                *e = s.surprisal;
            }
        }
    }

    // 🌟 [LOG] score_patches 상세 요약 출력
    println!("    📊 [SCORE_PATCHES DETAIL] 총 패치 {} | 제로벡터 스킵 {} | 빈점수 스킵 {} | 편견탈락 {} | 유효채점 {}",
        grid.patches.len(), zero_vec_skipped, empty_scores_skipped, dropped, total_scored);

    if !prejudice_dropped_details.is_empty() {
        println!("    📊 [PREJUDICE DROP SAMPLE] (상위 {}개):", prejudice_dropped_details.len());
        for (idx, desc, sur) in prejudice_dropped_details.iter().take(10) {
            println!("      ↳ patch[{}] {} | surprisal: {:+.4}", idx, desc, sur);
        }
    }

    if !top_contributors.is_empty() {
        println!("    📊 [TOP CONTRIBUTORS] 키별 최고점 갱신 패치 (상위 {}개):", top_contributors.len());
        for (_, key, sur, r, c) in top_contributors.iter().take(15) {
            println!("      ↳ '{}' ← r{}c{} | surprisal: {:+.4}", key, r, c, sur);
        }
    }

    (best, dropped)
}

struct TitleGateVerdict {
    code: String,
    group: String,
    title: String,
    score: f32,
    margin: f32,
}

fn run_title_gate(
    model: &Siglip2Model,
    grid: &PatchGrid,
    chrome_phrases: &[String],
    emit: &dyn Fn(&str),
) -> Option<(TitleGateVerdict, Vec<(String, f32)>)> {
    let empty_names: Vec<String> = Vec::new();
    let empty_banks: Vec<Vec<Vec<f32>>> = Vec::new();
    let empty_skip: Vec<bool> = Vec::new();

    // ── 뱅크: bias = 자기 전문 1구, prejudice = 다른 전문 + 크롬 ──
    let mut t_bias: Vec<(String, String, String)> = Vec::new();
    let mut t_prej: Vec<(String, String, String)> = Vec::new();

    for (code, title) in TRADE_DOC_TITLES.iter() {
        t_bias.push(("title".to_string(), code.to_string(), title.to_string()));
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();

        for (other, other_title) in TRADE_DOC_TITLES.iter() {
            if other == code {
                continue;
            }
            if seen.insert(other_title) {
                t_prej.push(("title".to_string(), code.to_string(), other_title.to_string()));
            }
        }

        for p in chrome_phrases.iter() {
            if seen.insert(p.as_str()) {
                t_prej.push(("title".to_string(), code.to_string(), p.clone()));
            }
        }
    }

    let bank = match build_anchor_bank(model, &t_bias, &t_prej) {
        Ok(b) => b,
        Err(_) => return None,
    };

    // ── 상단 30% 행만 제목 밴드로 봅니다 (레이아웃 구조 사실) ──
    let title_rows = (grid.grid_rows * 3 / 10).max(1);
    let mut best: std::collections::HashMap<String, f32> = std::collections::HashMap::new();

    // 🌟 [LOG] 타이틀 게이트 스캔 범위 및 패치 카운터
    let mut scanned_patches = 0usize;
    let mut active_patches = 0usize;
    let mut positive_patches = 0usize;
    let mut patch_best: std::collections::HashMap<String, (usize, usize, usize, f32)> =
        std::collections::HashMap::new();

    emit(&format!(
        "     🔍 [TITLE GATE SCAN] 상단 밴드: {}행 / 전체 {}행 | 스캔 패치 범위: 0~{}",
        title_rows, grid.grid_rows, title_rows * grid.grid_cols
    ));

    for idx in 0..grid.len() {
        let (r, c) = grid.rc(idx);
        if r >= title_rows {
            continue;
        }

        scanned_patches += 1;

        let p = &grid.patches[idx];
        if p.iter().all(|&v| v == 0.0) {
            continue;
        }

        active_patches += 1;

        let (scores, _) = surprisal_dual_scores(
            p,
            &bank.bias,
            &bank.prejudice,
            &empty_names,
            &empty_banks,
            &empty_skip,
        );

        if scores.is_empty() {
            continue;
        }

        if scores[0].surprisal <= 0.0 {
            continue;
        }

        positive_patches += 1;

        for s in scores {
            let e = best.entry(s.key.clone()).or_insert(f32::MIN);
            if s.surprisal > *e {
                *e = s.surprisal;
                patch_best.insert(s.key.clone(), (idx, r, c, s.surprisal));
            }
        }
    }

    // 🌟 [LOG] 타이틀 게이트 스캔 요약
    emit(&format!(
        "     🔍 [TITLE GATE SCAN RESULT] 스캔 {} | 활성 {} | 양수 {} | 전문 키 {}개 발견",
        scanned_patches, active_patches, positive_patches, best.len()
    ));

    if !patch_best.is_empty() {
        let mut rows: Vec<(String, usize, usize, usize, f32)> = patch_best
            .iter()
            .map(|(k, (i, r, c, s))| (k.clone(), *i, *r, *c, *s))
            .collect();
        rows.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        let spread = {
            let mut uniq: Vec<usize> = rows.iter().map(|x| x.1).collect();
            uniq.sort_unstable();
            uniq.dedup();
            uniq.len()
        };
        emit(&format!(
            "     🔍 [TITLE GATE CONTRIBUTORS] 전문 {}개의 최종 최고점을 만든 서로 다른 패치 {}개. 한 패치가 전문 다수의 최고점을 독점하면 그 패치는 제목이 아니라 로고나 도장일 가능성이 큽니다.",
            rows.len(), spread
        ));
        for (key, idx, r, c, sur) in rows.iter().take(12) {
            emit(&format!(
                "       ↳ '{}' ← patch[{}] r{}c{} {:+.4}",
                key, idx, r, c, sur
            ));
        }
    }

    if best.is_empty() {
        emit("   ⚪ [TITLE GATE] 상단 밴드에 인쇄된 서식 전문이 없습니다. 벡터 판정에 위임합니다.");
        return None;
    }

    let mut sorted: Vec<(String, f32)> = best.into_iter().collect();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    for (c, s) in sorted.iter() {
        emit(&format!("     📐 [TITLE GATE] {} | Surprisal: {:+.4}", c, s));
    }

    // 🌟 [LOG] 타이틀 게이트 1위-2위 마진
    if sorted.len() >= 2 {
        let tg_margin = sorted[0].1 - sorted[1].1;
        emit(&format!(
            "     📐 [TITLE GATE MARGIN] 1위 '{}'({:+.4}) - 2위 '{}'({:+.4}) = {:+.4}",
            sorted[0].0, sorted[0].1, sorted[1].0, sorted[1].1, tg_margin
        ));
    }

    let (top_code, top_score) = sorted[0].clone();
    let margin = top_score - sorted.get(1).map(|x| x.1).unwrap_or(top_score);

    // 동명 서식은 마진 0 → 거부
    if margin <= 0.0 {
        emit(&format!(
            "   ⚪ [TITLE GATE] '{}' 와 2위 전문 점수가 동률(마진 {:+.4})이라 거부하고 벡터 판정에 위임합니다.",
            top_code, margin
        ));
        return None;
    }

    let title = TRADE_DOC_TITLES
        .iter()
        .find(|(c, _)| *c == top_code.as_str())
        .map(|(_, t)| t.to_string())
        .unwrap_or_default();

    let group = crate::logic::TRADE_GROUP_CODES
        .iter()
        .find(|(_, cs)| cs.iter().any(|c| *c == top_code.as_str()))
        .map(|(g, _)| g.to_string())
        .unwrap_or_else(|| "shipping".to_string());

    Some((
        TitleGateVerdict {
            code: top_code,
            group,
            title,
            score: top_score,
            margin,
        },
        sorted,
    ))
}
///
///  Depth 1 : 그룹 (contract / shipping / customs / inspection / legal / parcel)
///  Depth 2 : 코드 (그룹 소속 코드만 경쟁)
///
///  scheduler.rs STEP A 와 동일한 구조이며,
///  판정 대상만 PUG 라인 → 이미지 패치로 바뀝니다.
pub fn classify_doc_type(
    model: &Siglip2Model,
    grid: &PatchGrid,
    emit: &dyn Fn(&str),
) -> anyhow::Result<DocTypeVerdict> {
    // ── Depth 1 : 그룹 뱅크 ──
    //
    // 🌟 [SPLIT 1회 캐시] 구버전은 split_bias_phrases_full 을 그룹당 (그룹수)번
    //    재호출했습니다. 6그룹이면 36회, Part 20 확장 후 7그룹이면 49회입니다.
    //    같은 문자열을 매번 다시 쪼개고 HashSet 을 다시 만드는 순수 낭비입니다.
    let group_phrases: Vec<(&str, Vec<String>)> = crate::logic::TRADE_GROUPS
        .iter()
        .map(|(g, raw)| (*g, split_bias_phrases_full(raw)))
        .collect();
    let chrome_phrases: Vec<String> =
        split_bias_phrases_full(crate::logic::VISION_CHROME_ANCHOR);

    let mut g_bias: Vec<(String, String, String)> = Vec::new();
    let mut g_prej: Vec<(String, String, String)> = Vec::new();
    for (gname, phrases) in group_phrases.iter() {
        for p in phrases.iter() {
            g_bias.push(("group".to_string(), gname.to_string(), p.clone()));
        }
        for (other, other_phrases) in group_phrases.iter() {
            if other == gname {
                continue;
            }
            for p in other_phrases.iter() {
                g_prej.push(("group".to_string(), gname.to_string(), p.clone()));
            }
        }
        // 🌟 [VISUAL CHROME] 이미지에만 존재하는 노이즈(로고 / 스탬프 / 여백 / 표 괘선)를
        //    모든 그룹의 공통 편견으로 추가합니다.
        //    텍스트 트랙에는 없던 축이지만, 비전에서는 문서 면적의 상당수를 차지합니다.
        for p in chrome_phrases.iter() {
            g_prej.push(("group".to_string(), gname.to_string(), p.clone()));
        }
    }

    // 🌟 [LOG] 그룹 앵커 구 샘플 — 각 그룹에 어떤 구가 코사인 판정 기준인지 보여줍니다.
    {
        let mut group_names: Vec<&str> = Vec::new();
        for (_, k, _) in g_bias.iter() {
            if !group_names.contains(&k.as_str()) { group_names.push(k.as_str()); }
        }
        emit(&format!(
            "  📖 [GROUP ANCHOR BANK] 그룹 {}개 | 판정 구 {}개 | 편견 구 {}개",
            group_names.len(), g_bias.len(), g_prej.len()
        ));
        for gn in group_names.iter() {
            let phrases: Vec<&str> = g_bias.iter()
                .filter(|(_, k, _)| k == gn)
                .map(|(_, _, p)| p.as_str())
                .collect();
            let sample: Vec<&str> = phrases.iter().take(4).copied().collect();
            let prej_cnt = g_prej.iter().filter(|(_, k, _)| k == *gn).count();
            emit(&format!(
                "    📖 [GROUP '{}' ] 구 {}개 | 편견 {}개 | 샘플: {:?}",
                gn, phrases.len(), prej_cnt, sample
            ));
        }
    }

    let g_bank = build_anchor_bank(model, &g_bias, &g_prej)?;
    // 🌟 [BANK-NEUTRAL] √(2 ln N) 차감을 폐기합니다.
    //    실측: shipping(33구)은 2.644, customs(17구)은 2.380 을 차감받아
    //    앵커가 많은 그룹이 구조적으로 불리했습니다.
    let (mut g_scores, g_active) = bank_neutral_key_scores(grid, &g_bank, None);
    let dropped = grid.len().saturating_sub(g_active);

    if g_scores.is_empty() {
        g_scores.push(("shipping".to_string(), 0.0));
    }

    // 🌟 [LOG] 그룹 점수 분포 상세 — 1위~최하위 전부 + 1위-2위 마진
    emit(&format!(
        "  📐 [VISION GROUP DETAIL] 그룹 {}개 채점 완료 (BANK-NEUTRAL) | 활성 패치: {}/{}",
        g_scores.len(),
        g_active,
        grid.len()
    ));

    for (g, s) in g_scores.iter() {
        emit(&format!(
            "  📐 [VISION GROUP] {} | Surprisal(max over patches): {:+.4}",
            g, s
        ));
    }

    // 🌟 [LOG] 그룹 1위-2위 마진 상세
    if g_scores.len() >= 2 {
        let margin = g_scores[0].1 - g_scores[1].1;
        emit(&format!(
            "  📐 [VISION GROUP MARGIN] 1위 '{}'({:+.4}) - 2위 '{}'({:+.4}) = 마진 {:+.4}",
            g_scores[0].0, g_scores[0].1,
            g_scores[1].0, g_scores[1].1,
            margin
        ));
        // 마진이 1.0 미만이면 경쟁이 치열했다는 경고
        if margin < 1.0 {
            emit(&format!(
                "  ⚠️ [VISION GROUP LOW MARGIN] 그룹 간 마진 {:+.4} < 1.0 — 경쟁이 치열합니다. 패치 분포 확인 필요.",
                margin
            ));
        }
    }

    emit(&format!(
        "  🧹 [PREJUDICE DROP] 패치 {}개가 편견 우세로 판정에서 제외되었습니다. (전체 {}개)",
        dropped,
        grid.len()
    ));

    let mut best_group = g_scores[0].0.clone();
    let group_score = g_scores[0].1;
    let group_margin = group_score - g_scores.get(1).map(|x| x.1).unwrap_or(group_score);

    // 🌟 [TRACKING VETO] 텍스트 트랙과 동일한 구조 게이트.
    //    parcel 은 '택배 라벨' 전용이므로, 무역 서식 개념이 강하게 잡히면 거부합니다.
    if best_group == "parcel" {
        let trade_evidence = g_scores
            .iter()
            .filter(|(g, _)| g != "parcel")
            .map(|(_, s)| *s)
            .fold(f32::MIN, f32::max);
        if trade_evidence > 0.0 {
            if let Some((alt, alt_s)) = g_scores.iter().find(|(g, _)| g != "parcel").cloned() {
                emit(&format!(
                    "  🚫 [TRACKING VETO] 무역 개념이 {:+.4} 로 검출되어 parcel 을 거부하고 '{}'({:+.4}) 로 교체합니다.",
                    trade_evidence, alt, alt_s
                ));
                best_group = alt;
            }
        }
    }

    emit(&format!(
        "  👑 [VISION GROUP SELECTED] '{}' | Top: {:+.4} | Margin: {:+.4}",
        best_group, group_score, group_margin
    ));

    // ── Depth 2 : 코드 뱅크 (승리 그룹 + 증거가 있는 그룹의 합집합) ──
    let mut codes: Vec<&'static str> = crate::logic::TRADE_GROUP_CODES
        .iter()
        .find(|(g, _)| *g == best_group)
        .map(|(_, c)| c.to_vec())
        .unwrap_or_else(|| vec!["Unknown"]);

    for (g, s) in g_scores.iter() {
        if g == &best_group || *s <= 0.0 {
            continue;
        }
        if g == "parcel" {
            continue;
        }
        if let Some((_, extra)) = crate::logic::TRADE_GROUP_CODES.iter().find(|(gn, _)| gn == g) {
            for c in extra.iter() {
                if !codes.iter().any(|x| x == c) {
                    codes.push(c);
                }
            }
        }
    }
    emit(&format!("  🎯 [VISION CODE CANDIDATES] {:?}", codes));

    let code_phrases: Vec<(&str, Vec<String>)> = codes
        .iter()
        .map(|c| (*c, split_bias_phrases_full(crate::logic::trade_code_anchor(c))))
        .collect();

    let mut c_bias: Vec<(String, String, String)> = Vec::new();
    let mut c_prej: Vec<(String, String, String)> = Vec::new();
    for (c, phrases) in code_phrases.iter() {
        for p in phrases.iter() {
            c_bias.push(("code".to_string(), c.to_string(), p.clone()));
        }
        // 이 코드 하나에 대한 편견 집합. 중복 구는 한 번만 담습니다.
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for (other, other_phrases) in code_phrases.iter() {
            if other == c {
                continue;
            }
            for p in other_phrases.iter() {
                if seen.insert(p.as_str()) {
                    c_prej.push(("code".to_string(), c.to_string(), p.clone()));
                }
            }
        }
        for p in chrome_phrases.iter() {
            if seen.insert(p.as_str()) {
                c_prej.push(("code".to_string(), c.to_string(), p.clone()));
            }
        }
    }

    emit(&format!(
        "  📐 [VISION CODE BANK] 후보 {}개 | 코드 구 {}개 | 편견 구 {}개",
        codes.len(),
        c_bias.len(),
        c_prej.len()
    ));

    // 🌟 [LOG] 코드 앵커 구 샘플 — 55개 전부 출력하면 과다하므로 상위 8개만 상세
    {
        let mut code_names: Vec<&str> = Vec::new();
        for (_, k, _) in c_bias.iter() {
            if !code_names.contains(&k.as_str()) { code_names.push(k.as_str()); }
        }
        let sample_limit = code_names.len().min(8);
        for cn in code_names.iter().take(sample_limit) {
            let phrases: Vec<&str> = c_bias.iter()
                .filter(|(_, k, _)| k == cn)
                .map(|(_, _, p)| p.as_str())
                .collect();
            let sample: Vec<&str> = phrases.iter().take(3).copied().collect();
            let prej_cnt = c_prej.iter().filter(|(_, k, _)| k == *cn).count();
            emit(&format!(
                "    📖 [CODE '{}' ] 구 {}개 | 편견 {}개 | 샘플: {:?}",
                cn, phrases.len(), prej_cnt, sample
            ));
        }
        if code_names.len() > sample_limit {
            emit(&format!(
                "    📖 [CODE ...] 나머지 {}개 코드는 샘플 생략",
                code_names.len() - sample_limit
            ));
        }
    }

    let c_bank = build_anchor_bank(model, &c_bias, &c_prej)?;
    // 🌟 [BANK-NEUTRAL] 실측: ED(3구)은 1.482, CI(6구)은 1.893 을 차감받아
    //    앵커 적은 코드가 무조건 이기는 구조였습니다. 중립 채점으로 교체합니다.
    let (mut c_scores, _c_active) = bank_neutral_key_scores(grid, &c_bank, None);

    if c_scores.is_empty() {
        c_scores.push((codes[0].to_string(), 0.0));
    }

    // 🌟 [LOG] 코드 점수 상위 10개 + 하위 3개 출력 (55개 전부 출력하면 로그 과다)
    let top_n = c_scores.len().min(10);
    let bottom_start = if c_scores.len() > 13 { c_scores.len() - 3 } else { top_n };
    for (c, s) in c_scores.iter().take(top_n) {
        emit(&format!("    📐 [VISION CODE] {} | Surprisal: {:+.4}", c, s));
    }
    if c_scores.len() > top_n {
        emit(&format!("    📐 [VISION CODE] ... ({}개 생략 아님, 중간 {}개는 아래 요약 참조)", c_scores.len(), c_scores.len() - top_n - (c_scores.len() - bottom_start)));
        for (c, s) in c_scores.iter().skip(bottom_start) {
            emit(&format!("    📐 [VISION CODE TAIL] {} | Surprisal: {:+.4}", c, s));
        }
    }

    // 🌟 [LOG] 코드 1위-2위 마진 + 양수 점수 개수
    if c_scores.len() >= 2 {
        let code_margin = c_scores[0].1 - c_scores[1].1;
        let positive_count = c_scores.iter().filter(|(_, s)| *s > 0.0).count();
        emit(&format!(
            "  📐 [VISION CODE MARGIN] 1위 '{}'({:+.4}) - 2위 '{}'({:+.4}) = 마진 {:+.4} | 양수 점수 코드: {}/{}",
            c_scores[0].0, c_scores[0].1,
            c_scores[1].0, c_scores[1].1,
            code_margin,
            positive_count,
            c_scores.len()
        ));
        // 🌟 [SDS 계측] 55개 코드 점수 배열 전체의 감쇠 형상을 남깁니다.
        //
        //  ── 왜 마진만으로는 부족한가 ──
        //   실측 `마진 +0.0434 | 양수 점수 코드: 55/55` 는
        //   '1·2위가 붙었다' 가 아니라 '55개 코드 전부가 살아남았다',
        //   즉 분포 자체에 변별력이 없다는 뜻입니다.
        //   그런데 현재 판정은 1·2위 마진만 보고 LLM 재판정 여부를 정합니다.
        //   entropy_norm 과 positive_ratio 를 쌓아 두면 Phase 3 의 V-3 에서
        //   '1위가 감쇠 곡선의 이상치인가' 로 판정 축을 바꿀 수 있습니다.
        //
        //  ── 스코프가 없으면 조용히 무시됩니다 ──
        //   이 함수는 순수 함수라 team_id 를 갖지 않습니다.
        //   호출부(model/vision.rs)가 enter_scope 로 세운 전역 스코프를 읽으며,
        //   세워지지 않았으면 record_decay 가 즉시 반환합니다.
        let arr: Vec<f32> = c_scores.iter().map(|(_, s)| *s).collect();
        crate::utils::score_dynamics::record_decay("vision.doc_code", &arr);
        crate::utils::score_dynamics::record_baseline("vision.code_margin", code_margin);
    }

    // 🌟 [TITLE AXIS NMS INTEGRATION] 상단 밴드 전문 점수를 바디 점수와 합성합니다.
    //    기존 '사후 오버라이드' 는 임시방편이었으므로, 단일 점수 축으로 편입해
    //    NMS 자체가 결정론적으로 정답을 내게 합니다.
    //    가중치 2.0: 실측 제목 축 마진(1.3076)이 바디 축 최대 왜곡(약 1.1)보다 커지도록 한 값.
    let title_gate_result = run_title_gate(model, grid, &chrome_phrases, emit);
    let title_confirmed = title_gate_result.is_some();
    let title_text = title_gate_result
        .as_ref()
        .map(|(g, _)| g.title.clone())
        .unwrap_or_default();
    if let Some((_gate, title_sorted)) = title_gate_result {
        for (cname, cs) in c_scores.iter_mut() {
            if let Some((_, ts)) = title_sorted.iter().find(|(t, _)| t == cname) {
                *cs += 2.0 * ts;
            }
        }
        c_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    }

    let mut code = c_scores[0].0.clone();
    let mut code_score = c_scores[0].1;
    let mut code_margin = code_score - c_scores.get(1).map(|x| x.1).unwrap_or(code_score);
    let mut final_group = best_group.clone();
    let mut final_group_score = group_score;
    let mut final_group_margin = group_margin;

    // 🌟 [GROUP RE-ANCHOR] 그룹은 최종 코드가 속한 그룹으로 재확정합니다.
    //    그룹 단계가 customs 를 외쳐도 코드 단계가 CI 를 뽑으면 shipping 이 정답입니다.
    if let Some((g, _)) = crate::logic::TRADE_GROUP_CODES
        .iter()
        .find(|(_, cs)| cs.iter().any(|c| *c == code.as_str()))
    {
        final_group = g.to_string();
        final_group_score = g_scores
            .iter()
            .find(|(n, _)| n == g)
            .map(|(_, s)| *s)
            .unwrap_or(final_group_score);
        let rest = g_scores
            .iter()
            .filter(|(n, _)| n != g)
            .map(|(_, s)| *s)
            .fold(f32::MIN, f32::max);
        final_group_margin = final_group_score - rest;
    }


    emit(&format!(
        "  👑 [VISION CODE SELECTED] '{}' | Top: {:+.4} | Margin: {:+.4}",
        code, code_score, code_margin
    ));

    Ok(DocTypeVerdict {
        group: final_group,
        group_score: final_group_score,
        group_margin: final_group_margin,
        code,
        code_score,
        code_margin,
        prejudice_dropped: dropped,
        code_candidates: c_scores,
        title_confirmed,
        title_text,
    })
}

// =====================================================================
// 🌟 [STEP 2] Column Cosine Matching
// =====================================================================

/// 카테고리 하나에 대한 2D 코사인 히트맵.
pub struct CategoryHeatmap {
    pub category: String,
    /// grid_rows * grid_cols 길이. 각 패치의 순위 점수(surprisal).
    pub scores: Vec<f32>,
    /// 이 카테고리에서 가장 강하게 반응한 필드명 (진단용).
    pub top_field: String,
    pub top_score: f32,
    pub territory: usize,
    pub mean_margin: f32,
    pub top_rival: String,
    pub absent: bool,
    pub absent_reason: String,
    pub field_peaks: Vec<(String, usize, f32)>,
}

/// 🌟 [STEP 2] 스키마 카테고리별 히트맵을 만듭니다.
///
///  ── get_trade_doc_slice_config 대체 지점 ──
///   기존:  ("header", 0.00, 0.25) 처럼 좌표를 손으로 적어 둔 표
///   변경:  bias_schema 의 필드 semantic/bias 구를 SigLIP2 텍스트 공간에 올리고
///          패치와 코사인을 재서 '실제로 인쇄된 위치' 를 찾습니다.
///
///  ── 카테고리 정의 출처 ──
///   parsing.rs 의 get_trade_category_schema 가 소비하는 8개 카테고리
///   (header / parties / logistics / conditions / financials / cargo / items / containers)
///   를 그대로 씁니다. 저장 스키마와 히트맵 축이 어긋나지 않습니다.
pub fn build_column_heatmaps(
    model: &Siglip2Model,
    grid: &PatchGrid,
    doc_type: &str,
    doc_lang: &str,
    legibility: Option<&crate::models::siglip2::legibility::LegibilityMap>,
    title_prejudice: &[String],
    emit: &dyn Fn(&str),
) -> anyhow::Result<Vec<CategoryHeatmap>> {
    use std::collections::HashMap;

    // ── 1) 카테고리 × 필드 앵커 구 수집 ──
    //    필드 → 카테고리 매핑은 logic.rs 가 소유합니다.
    let schema_fields = crate::parsing::get_detail_schema_fields(doc_type, "", doc_lang);
    let mut bias_defs: Vec<(String, String, String)> = Vec::new();
    let mut field_to_cat: HashMap<String, String> = HashMap::new();

    // 🌟 [SELF-REFERENCE ANCHOR DROP] 자기 자신을 가리키는 참조 축은 존재할 수 없습니다.
    //
    //  ── 실측 사고 ──
    //   CI 인보이스에서 header 히트맵의 top_field 가 doc_number 가 아니라
    //   reference_invoice(+4.3688) 였습니다.
    //   'invoice number' 라는 라벨 하나를 두고 doc_number 와 reference_invoice 의
    //   앵커가 사실상 동일하기 때문입니다.
    //   그런데 CI 는 정의상 자기 자신을 참조하지 않으므로 reference_invoice 는
    //   이 문서에 존재할 수 없는 축입니다. 존재할 수 없는 축이 정체성 축을 이겼고,
    //   그 결과 header 크롭이 VAT/EORI 행으로 착지해 doc_number 가 null 이 되었습니다.
    //   doc_number 가 비면 릴레이 키가 사라져 문서 그래프가 통째로 끊깁니다.
    //
    //  ── 왜 여기서 자르는가 ──
    //   scheduler.rs 의 [SELF-REFERENCE DROP] 은 '저장 직전' 에 자기 참조를 지웁니다.
    //   그 시점에는 이미 크롭이 잘못 착지한 뒤라 되돌릴 수 없습니다.
    //   같은 판정을 히트맵 단계로 앞당깁니다. 사전은 logic.rs 것을 그대로 씁니다.
    let self_ref_field = crate::logic::trade_reference_field_of(doc_type).unwrap_or("");
    for (fname, _, bias_target, _) in schema_fields.iter() {
        let cat = crate::logic::trade_field_category(fname);
        if cat.is_empty() {
            continue;
        }
        if !self_ref_field.is_empty() && fname == self_ref_field {
            emit(&format!(
                "  🧹 [SELF-REFERENCE ANCHOR DROP] '{}' 는 '{}' 문서가 자기 자신을 가리키는 축입니다. doc_number 와 라벨이 완전히 겹쳐 정체성 축을 잠식하므로 히트맵 경쟁에서 제외합니다.",
                fname, doc_type
            ));
            continue;
        }
        // 🌟 [DOC TYPE ANCHOR DROP] doc_type 은 STEP 1 이 이미 확정한 축입니다.
        //
        //  ── 왜 히트맵에서도 빼야 하는가 ──
        //   doc_type 의 앵커 구는 "commercial invoice" / "document kind code" 처럼
        //   문서 제목 그 자체입니다. 그래서 제목 행 패치에 가장 강하게 반응하고,
        //   header 카테고리의 봉우리를 제목 쪽으로 끌어당깁니다.
        //   실측에서 header 크롭이 grid(r2~4, c4~13) 로 잡혀 좌측 식별 블록(c0~c3)을
        //   통째로 잘라먹었고, 그 결과 doc_number 가 INVOICE NUMBER 가 아니라
        //   우측의 AIRWAYBILL 번호(93763111837)로 확정되었습니다.
        //
        //   TITLE ROW SUPPRESSION 이 상단 2행을 억제하지만, doc_type 앵커는
        //   그 아래 행까지 제목 어휘로 반응하므로 억제만으로는 부족합니다.
        //   물어보지 않는 축이면 위치도 찾을 필요가 없습니다.
        if fname == "doc_type" {
            emit(&format!(
                "  🧹 [DOC TYPE ANCHOR DROP] 'doc_type' 은 STEP 1 비전 판정이 '{}' 로 이미 확정한 축입니다. 앵커 구가 제목 행에 반응해 header 봉우리를 제목 쪽으로 끌어당기므로 히트맵 경쟁에서 제외합니다.",
                doc_type
            ));
            continue;
        }
        field_to_cat.insert(fname.clone(), cat.to_string());

        // semantic 앵커 (필드의 정체성 문구)
        // 🌟 [LABEL + VALUE 이중 축]
        //  ── 왜 값 예시를 되살리는가 ──
        //   텍스트 트랙에서 값 예시를 배제하는 것은 옳습니다. PUG 라인은
        //   라벨과 값이 이미 '|' 로 분리되어 있어 라벨만 맞히면 되기 때문입니다.
        //   비전은 반대입니다. 우리가 찾는 것은 '값이 인쇄된 픽셀 영역' 이고,
        //   라벨 앵커만 두면 히트맵이 캡션에만 반응합니다.
        //   실측: pol="AIRWAYBILL / BILL OF LADING", recipient_name="BUYER (IF NOT CONSIGNEE)"
        //        — 네 건 전부 값이 아니라 라벨을 읽은 결과입니다.
        //   라벨 축과 값 축을 모두 두면 두 봉우리가 생기고,
        //   expand_row_band 가 같은 행 밴드에서 둘을 이어 붙입니다.
        let sem = crate::utils::ai_utils::semantic_anchor_text(doc_lang, doc_type, fname);
        let mut label_cnt = 0usize;
        let mut value_cnt = 0usize;
        let mut dup_cnt = 0usize;
        for p in split_bias_phrases_full(&sem) {
            // 🌟 [DUP ANCHOR DROP] 같은 (카테고리, 필드) 안의 중복 구를 제거합니다.
            //    실측: reference_bl(2구): ["reference bl", "reference bl"]
            //    score_patches_bank_neutral 은 μ_k(그 뱅크가 이 문서에서 보이는 평균 반응)를
            //    기준선으로 빼기 때문에, 같은 구가 두 번 들어가면 그 필드의 기준선만
            //    치우쳐 다른 필드와의 경쟁이 불공정해집니다.
            //    아래 bias_target 루프에는 이미 같은 dedup 이 있는데 이 루프에만 없었습니다.
            if bias_defs.iter().any(|(c, k, e)| c == cat && k == fname && e == &p) {
                dup_cnt += 1;
                continue;
            }
            if crate::utils::ai_utils::is_value_example_phrase(&p) {
                value_cnt += 1;
            } else {
                label_cnt += 1;
            }
            bias_defs.push((cat.to_string(), fname.clone(), p));
        }
        if dup_cnt > 0 {
            emit(&format!(
                "  🧹 [DUP ANCHOR DROP] '{}' 의 중복 앵커 구 {}개 제거 (BANK-NEUTRAL 기준선 왜곡 방지)",
                fname, dup_cnt
            ));
        }
        if value_cnt > 0 {
            emit(&format!(
                "  🏷️ [LABEL+VALUE] '{}' | 라벨 구 {}개 + 값 예시 구 {}개 (비전은 값이 인쇄된 위치를 찾아야 합니다)",
                fname, label_cnt, value_cnt
            ));
        }
        // bias 구 (동의어 나열)
        for p in split_bias_phrases_full(bias_target) {
            if crate::utils::ai_utils::is_value_example_phrase(&p) {
                continue;
            }
            if bias_defs
                .iter()
                .any(|(c, k, e)| c == cat && k == fname && e == &p)
            {
                continue;
            }
            bias_defs.push((cat.to_string(), fname.clone(), p));
        }
    }

    // 🌟 [TABLE STRUCTURE ANCHOR 편입]
    //  items / containers 는 스키마 필드가 각각 hs_code / container_number 정도뿐이라
    //  앵커 밀도가 다른 카테고리의 1/10 수준입니다.
    //  실측에서 items 히트맵이 최하위(+1.3126)로 밀려 상품 표를 놓쳤습니다.
    //  '표' 라는 시각 구조 자체를 앵커로 세워 위치 신호를 복원합니다.
    {
        let table_axes: [(&str, &str, &str); 2] = [
            ("items", "__table_structure__", crate::logic::TRADE_TABLE_STRUCTURE_ANCHOR),
            ("containers", "__container_table__", crate::logic::TRADE_CONTAINER_TABLE_ANCHOR),
        ];
        for (cat, pseudo_field, anchor) in table_axes.iter() {
            // 그 카테고리에 실제 스키마 필드가 하나라도 있을 때만 편입합니다.
            let has_field = field_to_cat.values().any(|c| c == cat);
            if !has_field {
                continue;
            }
            field_to_cat.insert(pseudo_field.to_string(), cat.to_string());
            let mut added = 0usize;
            for p in split_bias_phrases_full(anchor) {
                if bias_defs
                    .iter()
                    .any(|(c, k, e)| c == cat && k == pseudo_field && e == &p)
                {
                    continue;
                }
                bias_defs.push((cat.to_string(), pseudo_field.to_string(), p));
                added += 1;
            }
            if added > 0 {
                emit(&format!(
                    "  🧾 [TABLE ANCHOR] '{}' 카테고리에 표 구조 앵커 {}구 편입 (스키마 필드만으로는 표 위치를 못 잡습니다)",
                    cat, added
                ));
            }
        }
    }

    if bias_defs.is_empty() {
        emit(&format!(
            "  ⚪ [VISION COLUMN] doc_type='{}' 에 대응하는 스키마 필드가 없어 히트맵을 만들지 않습니다.",
            doc_type
        ));
        return Ok(Vec::new());
    }

    let cats: Vec<String> = {
        let mut v: Vec<String> = Vec::new();
        for (c, _, _) in bias_defs.iter() {
            if !v.iter().any(|x| x == c) {
                v.push(c.clone());
            }
        }
        v
    };

    let mut prej_defs: Vec<(String, String, String)> = Vec::new();
    {
        let mut global: Vec<String> = Vec::new();
        for p in split_bias_phrases_full(crate::logic::VISION_CHROME_ANCHOR) {
            if !global.iter().any(|e| e == &p) {
                global.push(p);
            }
        }
        for p in title_prejudice.iter() {
            if !global.iter().any(|e| e == p) {
                global.push(p.clone());
            }
        }
        for cat in cats.iter() {
            for p in global.iter() {
                prej_defs.push((cat.clone(), cat.clone(), p.clone()));
            }
        }
        emit(&format!(
            "  🧹 [PREJUDICE SCOPE v3] 교차 카테고리 편견 폐기(열 센터링이 이미 수행) | 크롬 {}구 + 제목 {}구 = 공통 편견 {}구 × 카테고리 {}개 = {}항목",
            global.len().saturating_sub(title_prejudice.len()),
            title_prejudice.len(),
            global.len(),
            cats.len(),
            prej_defs.len()
        ));
    }

    // 🌟 [LOG] 카테고리별 필드 구 개수 상세
    {
        let mut cat_phrase_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for (c, _, _) in bias_defs.iter() {
            *cat_phrase_counts.entry(c.as_str()).or_insert(0) += 1;
        }
        let mut cat_detail: Vec<String> = Vec::new();
        for (c, cnt) in cat_phrase_counts.iter() {
            cat_detail.push(format!("{}({})", c, cnt));
        }
        cat_detail.sort();
        emit(&format!(
            "  📐 [VISION COLUMN BANK DETAIL] 카테고리별 필드 구: {}",
            cat_detail.join(" | ")
        ));
    }

    // 🌟 [LOG] 카테고리별 편견 구 개수 상세
    {
        let mut cat_prej_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for (c, _, _) in prej_defs.iter() {
            *cat_prej_counts.entry(c.as_str()).or_insert(0) += 1;
        }
        let mut prej_detail: Vec<String> = Vec::new();
        for (c, cnt) in cat_prej_counts.iter() {
            prej_detail.push(format!("{}({})", c, cnt));
        }
        prej_detail.sort();
        emit(&format!(
            "  📐 [VISION COLUMN PREJ DETAIL] 카테고리별 편견 구: {}",
            prej_detail.join(" | ")
        ));
    }

    emit(&format!(
        "  📐 [VISION COLUMN BANK] 카테고리 {}개 | 필드 구 {}개 | 편견 구 {}개 (카테고리 단위 축약) | 패치 {}개",
        cats.len(),
        bias_defs.len(),
        prej_defs.len(),
        grid.len()
    ));

    // 🌟 [LOG] 카테고리별 필드 앵커 구 샘플
    {
        let mut cat_list: Vec<&str> = Vec::new();
        for (c, _, _) in bias_defs.iter() {
            if !cat_list.contains(&c.as_str()) { cat_list.push(c.as_str()); }
        }
        emit(&format!(
            "  📖 [FIELD ANCHOR BANK] 카테고리 {}개 | 필드 구 {}개 | 편견 구 {}개",
            cat_list.len(), bias_defs.len(), prej_defs.len()
        ));
        for cat in cat_list.iter() {
            let fields: Vec<&str> = bias_defs.iter()
                .filter(|(c, _, _)| c == cat)
                .map(|(_, f, _)| f.as_str())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            let phrase_cnt = bias_defs.iter().filter(|(c, _, _)| c == *cat).count();
            let prej_cnt = prej_defs.iter().filter(|(c, _, _)| c == *cat).count();
            // 필드별 구 수와 샘플 구 2개씩 (카테고리당 최대 3필드만)
            let mut field_detail: Vec<String> = Vec::new();
            for f in fields.iter().take(3) {
                let f_phrases: Vec<&str> = bias_defs.iter()
                    .filter(|(c, k, _)| c == cat && k == f)
                    .map(|(_, _, p)| p.as_str())
                    .collect();
                let sample: Vec<&str> = f_phrases.iter().take(2).copied().collect();
                field_detail.push(format!("{}({}구): {:?}", f, f_phrases.len(), sample));
            }
            emit(&format!(
                "    📖 [CAT '{}' ] 필드 {}개 | 구 {}개 | 편견 {}개 | {}",
                cat, fields.len(), phrase_cnt, prej_cnt, field_detail.join(" ; ")
            ));
            if fields.len() > 3 {
                emit(&format!(
                    "    📖 [CAT '{}' ...] 나머지 필드: {:?}",
                    cat,
                    fields.iter().skip(3).collect::<Vec<_>>()
                ));
            }
        }
    }

    let bank = build_anchor_bank(model, &bias_defs, &prej_defs)?;

    let n = grid.len();
    let mut cat_scores: HashMap<String, Vec<f32>> = HashMap::new();
    let mut cat_top: HashMap<String, (String, f32)> = HashMap::new();
    for c in cats.iter() {
        cat_scores.insert(c.clone(), vec![f32::MIN; n]);
        cat_top.insert(c.clone(), (String::new(), f32::MIN));
    }

    // 🌟 [BANK-NEUTRAL] 필드별 √(2 ln N) 차감을 폐기하고
    //    행/열 이중 센터링으로 뱅크 크기·응집도 편향을 제거합니다.
    //    (실측: reference_sr 1구가 status 19구보다 2.4점 공짜 우위)
    let (keys, matrix) = score_patches_bank_neutral(grid, &bank, legibility);
    let mut field_peaks_by_cat: HashMap<String, Vec<(String, usize, f32)>> = HashMap::new();
    let mut peak_cands: Vec<(String, String, Vec<(usize, f32)>)> = Vec::new();
    for (ki, fname) in keys.iter().enumerate() {
        if fname.starts_with("__") { continue; }
        let cat = match field_to_cat.get(fname) {
            Some(c) => c.clone(),
            None => continue,
        };
        let row = match matrix.get(ki) {
            Some(r) => r,
            None => continue,
        };
        let vals: Vec<(usize, f32)> = row
            .iter()
            .enumerate()
            .filter(|(_, v)| **v != f32::MIN)
            .map(|(i, v)| (i, *v))
            .collect();
        if vals.len() < 3 { continue; }
        let cnt = vals.len() as f32;
        let mean = vals.iter().map(|(_, v)| *v).sum::<f32>() / cnt;
        let sd = (vals.iter().map(|(_, v)| (*v - mean) * (*v - mean)).sum::<f32>() / cnt).sqrt();
        if sd <= 1e-6 { continue; }
        let gate = crate::utils::ai_utils::gumbel_expected_z(vals.len());
        let mut ranked: Vec<(usize, f32)> = vals
            .iter()
            .map(|(i, v)| (*i, (*v - mean) / sd))
            .filter(|(_, z)| *z > gate)
            .collect();
        if ranked.is_empty() { continue; }
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        peak_cands.push((cat, fname.clone(), ranked));
    }
    peak_cands.sort_by(|a, b| {
        a.2.len()
            .cmp(&b.2.len())
            .then(b.2[0].1.partial_cmp(&a.2[0].1).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut taken: Vec<(usize, String)> = Vec::new();
    let mut displaced: Vec<String> = Vec::new();
    let mut shared: Vec<String> = Vec::new();
    for (cat, fname, ranked) in peak_cands.into_iter() {
        if let Some((_, owner)) = taken.iter().find(|(q, _)| *q == ranked[0].0) {
            crate::utils::score_dynamics::record_confusion(owner, &fname, 0.0);
        }
        let top_row = grid.rc(ranked[0].0).0;
        let hit = ranked
            .iter()
            .find(|(p, _)| grid.rc(*p).0 == top_row && !taken.iter().any(|(q, _)| q == p))
            .map(|(p, z)| (*p, *z));
        match hit {
            Some((p, z)) => {
                if p != ranked[0].0 {
                    displaced.push(format!("{}({}→{}, z {:+.2})", fname, ranked[0].0, p, z));
                }
                taken.push((p, fname.clone()));
                field_peaks_by_cat.entry(cat).or_default().push((fname, p, z));
            }
            None => {
                let (p, z) = ranked[0];
                let owner = taken
                    .iter()
                    .find(|(q, _)| *q == p)
                    .map(|(_, o)| o.clone())
                    .unwrap_or_else(|| "-".to_string());
                crate::utils::score_dynamics::record_baseline("vision.peak_shared_z", z);
                shared.push(format!("{}↔{}(패치 {}, z {:+.2})", fname, owner, p, z));
                field_peaks_by_cat.entry(cat).or_default().push((fname, p, z));
            }
        }
    }
    if !displaced.is_empty() {
        emit(&format!(
            "    🔀 [PEAK NMS] 1순위 패치를 더 제약이 큰 필드에 내주고 같은 행의 차순위로 내려간 필드 {}개: {:?} — 라벨과 값은 같은 텍스트 행에 인쇄되므로, 행을 넘어간 차순위는 같은 필드의 다른 위치가 아니라 다른 레코드입니다.",
            displaced.len(), displaced.iter().take(8).collect::<Vec<_>>()
        ));
    }
    if !shared.is_empty() {
        emit(&format!(
            "    🤝 [PEAK SHARED] 같은 행에 빈 칸이 없어 1순위 봉우리를 공유로 보유한 필드 {}개: {:?} — 봉우리를 버리면 그 필드는 PEAK VERIFY 후보에서 통째로 사라져 잘린 값이 그대로 확정됩니다. 한 칸을 가리켜도 복구 윈도우는 필드마다 따로 열리므로 중복 판독이 일어나지 않습니다. 충돌 쌍은 혼동 사전에 기록했습니다.",
            shared.len(), shared.iter().take(8).collect::<Vec<_>>()
        ));
    }
    emit(&format!(
        "    📍 [FIELD PEAKS] 자기 분포에서 √(2lnN) 을 넘는 필드 봉우리 {}개 (카테고리 {}개) — 같은 행 안에서만 배타 배정하고, 자리가 없으면 공유로 보유했습니다. 빈 필드 복구의 위치 근거로 넘깁니다.",
        field_peaks_by_cat.values().map(|v| v.len()).sum::<usize>(),
        field_peaks_by_cat.len()
    ));

    const FIELD_COUNT_NEUTRAL_WEIGHT: f32 = 1.0;

    let cat_pos = |c: &str| -> Option<usize> { cats.iter().position(|x| x == c) };
    // 🌟 [PHRASE-COUNT NEUTRAL] 필드별 앵커 구 수를 bias_defs 에서 그대로 셉니다.
    //
    //  ── 왜 필드 수가 아니라 구 수인가 ──
    //   score_patches_bank_neutral 의 Max-Pool 은 '구' 단위로 최댓값을 취합니다.
    //   따라서 기대 최댓값 √(2 ln N) 의 N 은 실제 경쟁에 참여한 표본 수,
    //   즉 구 수여야 합니다.
    //   실측: settlement 는 필드 1개라 √(2 ln 1) = 0.000 을 차감했는데
    //   그 한 필드가 구 17개를 갖고 있어 사실상 무보정이었습니다.
    //   결과는 영토 36칸(최대) · 판독 가능 19%(최저) · 스키마 밖 폐기 7건입니다.
    let mut field_phrases: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    for (_, f, _) in bias_defs.iter() {
        *field_phrases.entry(f.clone()).or_insert(0) += 1;
    }
    let mut cat_raw: Vec<Vec<f32>> = vec![vec![f32::MIN; n]; cats.len()];
    let mut cat_arg: Vec<Vec<usize>> = vec![vec![usize::MAX; n]; cats.len()];
    let mut cat_fields: Vec<usize> = vec![0usize; cats.len()];
    let mut cat_phrases: Vec<usize> = vec![0usize; cats.len()];
    let mut mapped_keys = 0usize;
    for (ki, fname) in keys.iter().enumerate() {
        let cat = match field_to_cat.get(fname) { Some(c) => c.clone(), None => continue };
        let ci = match cat_pos(&cat) { Some(v) => v, None => continue };
        mapped_keys += 1;
        cat_fields[ci] += 1;
        cat_phrases[ci] += field_phrases.get(fname).copied().unwrap_or(1);
        for i in 0..n {
            let v = matrix[ki][i];
            if v == f32::MIN { continue; }
            if v > cat_raw[ci][i] {
                cat_raw[ci][i] = v;
                cat_arg[ci][i] = ki;
            }
        }
    }
    // ── ① Max-Pool 표본 수 보정 ──
    {
        let mut detail: Vec<String> = Vec::new();
        for ci in 0..cats.len() {
            // 🌟 구 수를 N 으로 씁니다. 집계 실패 시에만 필드 수로 폴백합니다.
            let n_eff = cat_phrases[ci].max(cat_fields[ci]).max(1);
            let base = crate::utils::ai_utils::gumbel_expected_z(n_eff) * FIELD_COUNT_NEUTRAL_WEIGHT;
            detail.push(format!(
                "{}({}구/{}필드 −{:.3})",
                cats[ci], cat_phrases[ci], cat_fields[ci], base
            ));
            crate::utils::score_dynamics::record_baseline(
                &format!("vision.neutral_n.{}", cats[ci]),
                n_eff as f32,
            );
            if base <= 0.0 { continue; }
            for i in 0..n {
                if cat_raw[ci][i] != f32::MIN { cat_raw[ci][i] -= base; }
            }
        }
        detail.sort();
        emit(&format!(
            "    ⚖️ [CATEGORY-NEUTRAL] max-pool 표본 수(구 수) 편향 보정: {} — 기대 최댓값을 정하는 것은 필드 수가 아니라 Max-Pool 이 실제로 뽑는 드로잉 수입니다. 필드 1개에 구 17개를 몰아넣은 카테고리는 필드 수 기준으로 보정하면 차감이 0 이 되어 공짜 우위를 얻습니다.",
            detail.join(" | ")
        ));
    }
    // ── ② 카테고리 축 센터링 ──
    for i in 0..n {
        let mut s = 0.0f32;
        let mut c = 0usize;
        for ci in 0..cats.len() {
            if cat_raw[ci][i] == f32::MIN { continue; }
            s += cat_raw[ci][i];
            c += 1;
        }
        if c < 2 { continue; }
        let mean = s / c as f32;
        for ci in 0..cats.len() {
            if cat_raw[ci][i] == f32::MIN { continue; }
            cat_raw[ci][i] -= mean;
        }
    }
    // =====================================================================
    // 🌟 [V-1 / 공간 베이스필드 잔차화]
    // ---------------------------------------------------------------------
    //  ── 무엇이 남아 있었나 ──
    //   실측: 🔥 [HEATMAP] header | 활성 패치 249/252 (98.8%)
    //         ⚠️ [HEATMAP FULL PAGE RISK] 'insurance' 활성 패치 214/252 (85%)
    //
    //   ① CATEGORY-NEUTRAL 과 ② 카테고리 축 센터링은 둘 다 '카테고리 축' 을
    //   따라 작동합니다. ②가 제거하는 것은 '어떤 패치가 모든 카테고리에서 높은가'
    //   입니다. 그런데 위 증상은 정반대로 '어떤 카테고리가 모든 패치에서 높은가'
    //   이므로, ②는 이 편향을 구조적으로 건드리지 못합니다.
    //   오히려 그 카테고리가 패치마다 평균을 끌어올려 다른 카테고리를 음수로 밀어냅니다.
    //   즉 공간 축(패치 간) 보정이 통째로 비어 있었습니다.
    //
    //  ── 제거되지 않던 성분 ──
    //   표 괘선의 가로 확산      → 행 방향으로 균일
    //   컬럼 구조의 세로 확산    → 열 방향으로 균일
    //   워터마크·배경 톤         → 전역 균일
    //   스캔 잉크 편치           → 저주파 그라디언트
    //   header 앵커('문서번호·발행일·참조번호')가 표 격자 전반과 약하게 공명하는데,
    //   그 공명이 행/열 방향으로 퍼져 있어 249/252 가 나옵니다.
    //
    //  ── 왜 double centering 인가 ──
    //   새 알고리즘이 아닙니다. ai_utils::double_center_matrix 가
    //   라벨×필드 행렬에서 뱅크 크기 편향을 제거하려고 쓰는 바로 그 연산자를
    //   패치 격자(행×열)에 그대로 적용하는 것입니다.
    //     관측 = 전체 평균 + 행 효과 + 열 효과 + 잔차
    //   워터마크는 μ, 괘선 가로 확산은 행 효과, 컬럼 확산은 열 효과로 분해되고,
    //   '실제 필드 블록' 만 잔차로 남습니다.
    //   TS 로 치면 차분(differencing) 에 대응하는 공간 자기회귀 잔차화입니다.
    //
    //  ── 왜 적용 대상을 고르는 임계값이 없는가 ──
    //   이 연산자는 자기적응적입니다.
    //     이미 좁은 히트맵(containers 30/252) → 행/열 효과가 작음 → 잔차 ≈ 원본
    //     페이지를 덮는 히트맵(header 249/252) → 행/열 효과가 큼 → 대량 제거
    //   따라서 전 카테고리에 균일 적용하면 되고, 새 상수가 생기지 않습니다.
    //
    //  ── 리스크 R7 방어 ──
    //   MSDS 처럼 페이지 전면이 실제로 hazmat 인 문서는 진짜 신호가 공간적으로
    //   균일하므로 잔차가 0 이 됩니다. 그 경우 히트맵이 통째로 죽습니다.
    //   방어는 임계값이 아니라 이진 사실로 둡니다.
    //     '잔차화 후 양수 패치가 하나도 없는데 원본에는 있었다' → 되돌림
    //   상수 없이 '히트맵이 죽었다' 는 관측만으로 판정합니다.
    // =====================================================================
    {
        // 격자 열 수는 grid.rc() 로 역산합니다. PatchGrid 의 내부 필드에
        // 의존하지 않기 위해서이며, NaFlex 로 종횡비가 달라져도 안전합니다.
        let grid_cols = {
            let mut mx = 0usize;
            for i in 0..n {
                let (_, c) = grid.rc(i);
                if c > mx { mx = c; }
            }
            mx + 1
        };
        let grid_rows = grid.grid_rows.max(1);

        // 🌟 [V-1 수정 / 과확산 선별] 전 카테고리 균일 적용을 철회합니다.
        //
        //  ── 실측 역효과 ──
        //   균일 적용 결과: cargo(66→105) conditions(99→116) containers(71→103)
        //   financials(106→115) items(45→115) logistics(70→113)
        //   other_parties(109→123) parties(84→117)
        //   10개 중 8개가 오히려 퍼졌고, 크롭 커버리지 손실이
        //   header 1개에서 8개 카테고리로 확대되었습니다.
        //
        //  ── 왜 그렇게 되는가 ──
        //   잔차 = v − row_mean − col_mean + μ 의 총합은 정확히 0 입니다.
        //   따라서 양수 개수가 구조적으로 전체의 절반(252/2 ≈ 126)으로 수렴합니다.
        //   실제 결과가 전부 103~133 에 몰린 것이 그 증거입니다.
        //   즉 이 연산자는 모든 카테고리를 같은 확산도로 '균질화' 하며,
        //   뾰족한 히트맵(items 45개)을 강제로 평평하게 만듭니다.
        //   '좁은 히트맵은 행/열 효과가 작아 잔차 ≈ 원본' 이라는 제 전제가 틀렸습니다.
        //
        //  ── 교정 ──
        //   '이 문서 안에서 유독 퍼진 카테고리' 에만 적용합니다.
        //   기준은 카테고리별 활성 비율의 중앙값과 절대편차 중앙값(MAD)이며,
        //   둘 다 이 문서의 분포에서 유도되므로 새 상수가 없습니다.
        //   MAD 는 표준편차와 달리 극단값(settlement 240)에 끌려가지 않아
        //   '나머지가 정상일 때 하나만 튄' 상황을 정확히 잡습니다.
        let spread_gate: Option<f32> = {
            let mut ratios: Vec<f32> = Vec::with_capacity(cats.len());
            for ci in 0..cats.len() {
                let hot = cat_raw[ci]
                    .iter()
                    .filter(|v| **v != f32::MIN && **v > 0.0)
                    .count();
                ratios.push(hot as f32 / n.max(1) as f32);
            }
            if ratios.len() < 3 {
                None
            } else {
                let mut s = ratios.clone();
                s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let med = s[s.len() / 2];
                let mut dev: Vec<f32> = ratios.iter().map(|r| (r - med).abs()).collect();
                dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mad = dev[dev.len() / 2];
                if mad <= 1e-6 {
                    // 전 카테고리가 같은 확산도면 '유독 퍼진 것' 이 존재하지 않습니다.
                    None
                } else {
                    Some(med + mad)
                }
            }
        };
        match spread_gate {
            Some(g) => emit(&format!(
                "    🧭 [SPATIAL RESIDUAL GATE] 이 문서의 카테고리 확산도 중앙값+MAD = {:.3} 를 넘는 카테고리에만 잔차화를 적용합니다. (균일 적용은 뾰족한 히트맵까지 평평하게 만듭니다)",
                g
            )),
            None => emit(
                "    ⏭️ [SPATIAL RESIDUAL SKIP] 카테고리 확산도가 고르거나 표본이 부족해 '유독 퍼진 카테고리' 를 특정할 수 없습니다. 잔차화를 건너뜁니다.",
            ),
        }

        let mut applied = 0usize;
        let mut reverted: Vec<String> = Vec::new();
        let mut skipped: Vec<String> = Vec::new();
        let mut detail: Vec<String> = Vec::new();

        for ci in 0..cats.len() {
            // 🌟 [V-1 수정] 게이트를 넘지 않는 카테고리는 원본을 그대로 둡니다.
            let gate = match spread_gate { Some(g) => g, None => break };
            {
                let hot_now = cat_raw[ci]
                    .iter()
                    .filter(|v| **v != f32::MIN && **v > 0.0)
                    .count();
                let ratio_now = hot_now as f32 / n.max(1) as f32;
                if ratio_now <= gate {
                    skipped.push(format!("{}({:.0}%)", cats[ci], ratio_now * 100.0));
                    continue;
                }
            }
            // ── 유효 패치만으로 전체/행/열 평균을 구합니다 ──
            let mut sum_all = 0.0f64;
            let mut cnt_all = 0usize;
            let mut row_sum = vec![0.0f64; grid_rows];
            let mut row_cnt = vec![0usize; grid_rows];
            let mut col_sum = vec![0.0f64; grid_cols];
            let mut col_cnt = vec![0usize; grid_cols];

            for i in 0..n {
                let v = cat_raw[ci][i];
                if v == f32::MIN || !v.is_finite() { continue; }
                let (r, c) = grid.rc(i);
                if r >= grid_rows || c >= grid_cols { continue; }
                sum_all += v as f64;
                cnt_all += 1;
                row_sum[r] += v as f64;
                row_cnt[r] += 1;
                col_sum[c] += v as f64;
                col_cnt[c] += 1;
            }
            // 격자 구조가 성립하지 않으면(패치가 너무 적음) 잔차화를 건너뜁니다.
            if cnt_all < 4 || grid_rows < 2 || grid_cols < 2 { continue; }

            let mu = (sum_all / cnt_all as f64) as f32;
            let row_mean: Vec<f32> = (0..grid_rows)
                .map(|r| if row_cnt[r] == 0 { mu } else { (row_sum[r] / row_cnt[r] as f64) as f32 })
                .collect();
            let col_mean: Vec<f32> = (0..grid_cols)
                .map(|c| if col_cnt[c] == 0 { mu } else { (col_sum[c] / col_cnt[c] as f64) as f32 })
                .collect();

            // ── 잔차 산출. 원본은 되돌림에 대비해 보존합니다 ──
            let before = cat_raw[ci].clone();
            let hot_before = before.iter().filter(|v| **v != f32::MIN && **v > 0.0).count();

            let mut residual = before.clone();
            for i in 0..n {
                let v = before[i];
                if v == f32::MIN || !v.is_finite() { continue; }
                let (r, c) = grid.rc(i);
                if r >= grid_rows || c >= grid_cols { continue; }
                // 관측 = μ + 행효과 + 열효과 + 잔차
                //      = μ + (row_mean[r] − μ) + (col_mean[c] − μ) + 잔차
                // 따라서 잔차 = v − row_mean[r] − col_mean[c] + μ
                residual[i] = v - row_mean[r] - col_mean[c] + mu;
            }
            let hot_after = residual.iter().filter(|v| **v != f32::MIN && **v > 0.0).count();

            // ── 리스크 R7 방어 (강화) ──
            //   ① 히트맵이 통째로 죽으면(양수 0) 되돌립니다.
            //      '페이지 전면이 실제로 이 카테고리인 문서'(MSDS 전면 hazmat 등)는
            //      진짜 신호가 공간적으로 균일하므로 잔차가 0 이 되는 것이 정상이고,
            //      그 문서에서 히트맵을 없애면 크롭 자체가 불가능해집니다.
            //
            //   ② 🌟 [신규] 잔차화 후 오히려 퍼지면 되돌립니다.
            //      실측에서 items(45→115) 처럼 2.6배 퍼지는 사례가 확인되었습니다.
            //      잔차의 총합이 0 이라 양수가 전체의 절반으로 수렴하기 때문입니다.
            //      V-1 의 목적은 '좁히는 것' 이므로, 넓어졌다면 그 자체가 실패입니다.
            //      게이트를 통과한 카테고리에서도 이 일이 일어날 수 있으므로
            //      최종 방어선으로 둡니다.
            if hot_after == 0 && hot_before > 0 {
                reverted.push(format!("{}({}→0 소멸)", cats[ci], hot_before));
                continue;
            }
            if hot_after >= hot_before {
                reverted.push(format!("{}({}→{} 확산)", cats[ci], hot_before, hot_after));
                continue;
            }

            cat_raw[ci] = residual;
            applied += 1;
            detail.push(format!(
                "{}({}→{})",
                cats[ci], hot_before, hot_after
            ));

            // 🌟 [SDS 계측] 잔차화 전후 확산도를 남깁니다.
            //    'V-1 이 실제로 확산을 줄였는가' 가 수용 기준이고,
            //    doc_type 별로 얼마나 줄었는지가 Phase 2 의 크롭 재수립(V-2) 입력입니다.
            crate::utils::score_dynamics::record_baseline(
                &format!("vision.spread_before.{}", cats[ci]),
                hot_before as f32 / n.max(1) as f32,
            );
            crate::utils::score_dynamics::record_baseline(
                &format!("vision.spread_after.{}", cats[ci]),
                hot_after as f32 / n.max(1) as f32,
            );
        }

        detail.sort();
        emit(&format!(
            "    🧭 [SPATIAL RESIDUAL] 공간 베이스필드(전역 톤 + 행 확산 + 열 확산)를 분해해 잔차만 남겼습니다. 격자 {}행×{}열 | 적용 {}개 카테고리 | 활성 패치 변화: {}",
            grid_rows, grid_cols, applied,
            if detail.is_empty() { "-".to_string() } else { detail.join(" | ") }
        ));
        if !reverted.is_empty() {
            reverted.sort();
            emit(&format!(
                "    ↩️ [SPATIAL RESIDUAL REVERT] 잔차화 후 양수 패치가 0 이 된 카테고리를 원본으로 되돌렸습니다: {} — 페이지 전면이 실제로 그 카테고리인 문서(예: MSDS 전면 hazmat)에서 정상입니다.",
                reverted.join(" | ")
            ));
        }
    }
    // ── ③ 결과 반영. top_field 는 최종 봉우리 패치의 argmax 필드입니다. ──
    let mut positive_by_cat: HashMap<String, usize> = HashMap::new();
    for (ci, c) in cats.iter().enumerate() {
        let mut best = f32::MIN;
        let mut best_i = usize::MAX;
        let mut pos = 0usize;
        for i in 0..n {
            let v = cat_raw[ci][i];
            if v == f32::MIN { continue; }
            if v > 0.0 { pos += 1; }
            if v > best { best = v; best_i = i; }
        }
        positive_by_cat.insert(c.clone(), pos);
        if let Some(slot) = cat_scores.get_mut(c) {
            *slot = cat_raw[ci].clone();
        }
        if let Some(t) = cat_top.get_mut(c) {
            let f = if best_i != usize::MAX && cat_arg[ci][best_i] != usize::MAX {
                keys[cat_arg[ci][best_i]].clone()
            } else {
                String::new()
            };
            *t = (f, best);
        }
    }
    emit(&format!(
        "    📊 [HEATMAP SCORING SUMMARY] 채점 키 {}개 (카테고리 매핑 성공 {}) | 패치 {}개",
        keys.len(), mapped_keys, n
    ));
    {
        let mut pf_detail: Vec<String> = Vec::new();
        for (cat, cnt) in positive_by_cat.iter() {
            pf_detail.push(format!("{}({})", cat, cnt));
        }
        pf_detail.sort();
        emit(&format!(
            "    📊 [HEATMAP POSITIVE PATCHES] 카테고리별 양수 패치 수 (카테고리 축 센터링 후): {}",
            pf_detail.join(" | ")
        ));
    }

    let mut out: Vec<CategoryHeatmap> = Vec::with_capacity(cats.len());
    // 🌟 [정정] 카테고리 인덱스를 되살립니다.
    //    cat_fields[ci] 는 ① 필드 수 보정 블록이 이미 쓰는 '이 카테고리의 앵커 필드 수' 이며,
    //    CATEGORY-NEUTRAL 이 가정한 드로잉 수 N 그 자체입니다.
    //    N_eff 캘리브레이션(기획 T-4)의 입력이 되므로 그대로 넘겨야 합니다.
    for (ci, c) in cats.iter().enumerate() {
        let scores = cat_scores.remove(c).unwrap_or_else(|| vec![f32::MIN; n]);
        let (top_field, top_score) = cat_top
            .remove(c)
            .unwrap_or_else(|| (String::new(), f32::MIN));

        let hot = scores.iter().filter(|s| **s > 0.0).count();

        // 🌟 [LOG] 히트맵 점수 분포 상세 — 양수 패치의 평균/최대/최소 + 행별 활성 분포
        let positive_vals: Vec<f32> = scores.iter().filter(|s| **s > 0.0).copied().collect();
        let (p_mean, p_max, p_min) = if positive_vals.is_empty() {
            (0.0f32, 0.0f32, 0.0f32)
        } else {
            let mean = positive_vals.iter().sum::<f32>() / positive_vals.len() as f32;
            let max = positive_vals.iter().cloned().fold(f32::MIN, f32::max);
            let min = positive_vals.iter().cloned().fold(f32::MAX, f32::min);
            (mean, max, min)
        };

        // 🌟 [LOG] 행별 활성 패치 수 — 잘림(상단/하단 편중) 감지용
        let mut row_active: Vec<usize> = vec![0; grid.grid_rows];
        for idx in 0..n {
            if scores[idx] > 0.0 {
                let (r, _) = grid.rc(idx);
                row_active[r] += 1;
            }
        }
        let active_rows: Vec<String> = row_active.iter().enumerate()
            .filter(|(_, cnt)| **cnt > 0)
            .map(|(r, cnt)| format!("r{}({})", r, cnt))
            .collect();

        emit(&format!(
            "    🔥 [HEATMAP] {} | 활성 패치 {}/{} | Top: {}({:+.4})",
            c,
            hot,
            n,
            if top_field.is_empty() { "-" } else { &top_field },
            top_score
        ));
        // 🌟 [SDS 계측] 카테고리별 히트맵 확산도를 남깁니다.
        //
        //  ── 표적 증상 ──
        //   실측 `활성 패치 249/252`(98.8%), `FULL PAGE RISK 85%`.
        //   score_patches_bank_neutral 이 기준선 패치의 μ_k 와 카테고리 센터링만
        //   차감하므로, 페이지 전역 베이스필드(워터마크·표 괘선 확산·잉크 편치·
        //   로고 반향)가 제거되지 않아 히트맵이 페이지를 통째로 덮습니다.
        //
        //  ── 이 관측이 Phase 1 에서 쓰이는 방식 ──
        //   V-1 공간 잔차화를 적용한 뒤 이 값이 실제로 내려갔는지가 수용 기준이고,
        //   잔차화가 정상 신호까지 지운 경우(리스크 R7)를 감지해
        //   원본 히트맵으로 폴백하는 판정의 기준선이 됩니다.
        crate::utils::score_dynamics::record_spatial(c, hot, n);
        crate::utils::score_dynamics::record_category_max(c, cat_fields[ci], top_score);

        // 🌟 [LOG] 히트맵 분포 + 잘림 체크
        emit(&format!(
            "    🔥 [HEATMAP DIST] {} | 양수 평균={:+.4} 최대={:+.4} 최소={:+.4} | 활성행: [{}]",
            c, p_mean, p_max, p_min,
            active_rows.join(", ")
        ));

        // 🌟 [LOG] 잘림 감지: 활성 패치가 전체 행의 20% 미만에 몰려 있으면 경고
        if !active_rows.is_empty() && hot > 0 {
            let total_rows = grid.grid_rows;
            let active_row_count = active_rows.len();
            let coverage = active_row_count as f32 / total_rows as f32;
            if coverage < 0.20 && hot > 3 {
                emit(&format!(
                    "    ⚠️ [HEATMAP TRUNCATION RISK] '{}' 활성 패치가 {}행/{}행({:.0}%)에만 집중 — 크롭 잘림 가능성 확인 필요",
                    c, active_row_count, total_rows, coverage * 100.0
                ));
            }
            // 상단 2행 또는 하단 2행에 80% 이상 몰려 있으면 편중 경고
            let top2: usize = row_active.iter().take(2).sum();
            let bottom2: usize = row_active.iter().rev().take(2).sum();
            if hot > 0 && (top2 * 5 > hot * 4 || bottom2 * 5 > hot * 4) {
                emit(&format!(
                    "    ⚠️ [HEATMAP EDGE BIAS] '{}' 활성 패치 {}개 중 상단2행={} 하단2행={} — 가장자리 편중 감지",
                    c, hot, top2, bottom2
                ));
            }
        }

        out.push(CategoryHeatmap {
            category: c.clone(),
            scores,
            top_field,
            top_score,
            territory: 0,
            mean_margin: 0.0,
            top_rival: String::new(),
            absent: false,
            absent_reason: String::new(),
            field_peaks: field_peaks_by_cat.remove(c).unwrap_or_default(),
        });
    }

    // 강한 카테고리부터 크롭 경쟁에 들어가도록 정렬합니다.
    out.sort_by(|a, b| {
        b.top_score
            .partial_cmp(&a.top_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(out)
}

// =====================================================================
// 진단 헬퍼
// =====================================================================

/// 히트맵을 터미널에 ASCII 로 그립니다. (디버깅 전용, 격자가 작을 때만)
pub fn render_heatmap_ascii(hm: &CategoryHeatmap, grid: &PatchGrid) -> String {
    if grid.grid_cols > 48 || grid.grid_rows > 48 {
        return String::new();
    }
    let mut s = format!("    [{}]\n", hm.category);
    for r in 0..grid.grid_rows {
        s.push_str("      ");
        for c in 0..grid.grid_cols {
            let v = hm.scores[r * grid.grid_cols + c];
            let ch = if v <= 0.0 {
                '.'
            } else if v < 0.5 {
                '-'
            } else if v < 1.0 {
                '+'
            } else if v < 2.0 {
                '#'
            } else {
                '@'
            };
            s.push(ch);
        }
        s.push('\n');
    }
    s
}

/// `Device` / `DType` 는 상위 모듈에서만 쓰이므로 미사용 경고를 방지합니다.
#[allow(dead_code)]
fn _unused_marker(_d: &Device, _t: DType, _c: &Siglip2Config, _x: &Tensor) {}

pub fn score_patches_bank_neutral(
    grid: &PatchGrid,
    bank: &AnchorBank,
    legible: Option<&crate::models::siglip2::legibility::LegibilityMap>,
) -> (Vec<String>, Vec<Vec<f32>>) {
    use std::collections::HashMap;

    let n = grid.len();
    if n == 0 || bank.bias.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // ── ① (category, key) 그룹 색인 ──
    let mut order: Vec<String> = Vec::new();
    let mut bias_idx: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, (_, k, _)) in bank.bias.iter().enumerate() {
        if !bias_idx.contains_key(k) {
            order.push(k.clone());
        }
        bias_idx.entry(k.clone()).or_default().push(i);
    }
    let mut prej_idx: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, (_, k, _)) in bank.prejudice.iter().enumerate() {
        prej_idx.entry(k.clone()).or_default().push(i);
    }

    let mut key_cat: HashMap<String, String> = HashMap::new();
    for (c, k, _) in bank.bias.iter() {
        key_cat.entry(k.clone()).or_insert_with(|| c.clone());
    }

    let dot = |a: &[f32], b: &[f32]| -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    };

    let mut prej_pool: HashMap<String, Vec<f32>> = HashMap::new();
    for (gname, list) in prej_idx.iter() {
        let mut v = vec![f32::MIN; n];
        for i in 0..n {
            let p = &grid.patches[i];
            if p.iter().all(|&x| x == 0.0) { continue; }
            let mut mp = f32::MIN;
            for &j in list {
                let s = dot(p, &bank.prejudice[j].2);
                if s > mp { mp = s; }
            }
            v[i] = mp;
        }
        prej_pool.insert(gname.clone(), v);
    }

    // ── ② 원시 Max-Pool 행렬 ──
    let m = order.len();
    let mut raw_b = vec![vec![f32::MIN; n]; m];
    let mut raw_p = vec![vec![f32::MIN; n]; m];
    let mut resolved = 0usize;

    for (ki, key) in order.iter().enumerate() {
        let bi = &bias_idx[key];
        // 필드명 → 실패 시 카테고리명 순으로 편견 그룹을 찾습니다.
        let pv: Option<&Vec<f32>> = prej_pool
            .get(key)
            .or_else(|| key_cat.get(key).and_then(|c| prej_pool.get(c)));
        if pv.is_some() { resolved += 1; }
        for i in 0..n {
            let p = &grid.patches[i];
            if p.iter().all(|&v| v == 0.0) { continue; }
            let mut mb = f32::MIN;
            for &j in bi {
                let s = dot(p, &bank.bias[j].2);
                if s > mb { mb = s; }
            }
            raw_b[ki][i] = mb;
            if let Some(pv) = pv {
                raw_p[ki][i] = pv[i];
            }
        }
    }

    // ── ③ 판독 가능 패치만으로 기준선 산출 ──
    let usable: Vec<usize> = (0..n)
        .filter(|&i| legible.map_or(true, |l| l.is_legible(i)))
        .filter(|&i| !grid.patches[i].iter().all(|&v| v == 0.0))
        .collect();
    let base: &[usize] = if usable.len() >= 8 {
        &usable
    } else {
        // 판독 가능 패치가 너무 적으면 전 패치로 폴백합니다.
        // (스캔 품질이 나쁠 때 통계가 통째로 무너지는 것을 막습니다)
        &[]
    };
    let idx_all: Vec<usize> = (0..n).collect();
    let base: &[usize] = if base.is_empty() { &idx_all } else { base };

    // 전역 pooled σ : 뱅크마다 σ 를 쓰면 1구 뱅크의 z 가 폭발합니다.
    let mut pooled_var = 0.0f32;
    let mut pooled_cnt = 0usize;
    let mut mu_b = vec![0.0f32; m];
    let mut mu_p = vec![0.0f32; m];
    for ki in 0..m {
        let mut sb = 0.0f32; let mut sp = 0.0f32; let mut c = 0usize;
        for &i in base {
            if raw_b[ki][i] == f32::MIN { continue; }
            sb += raw_b[ki][i];
            if raw_p[ki][i] != f32::MIN { sp += raw_p[ki][i]; }
            c += 1;
        }
        if c == 0 { continue; }
        mu_b[ki] = sb / c as f32;
        mu_p[ki] = sp / c as f32;
        for &i in base {
            if raw_b[ki][i] == f32::MIN { continue; }
            let d = raw_b[ki][i] - mu_b[ki];
            pooled_var += d * d;
            pooled_cnt += 1;
        }
    }
    let sd = if pooled_cnt > 1 {
        (pooled_var / pooled_cnt as f32).sqrt().max(1e-6)
    } else {
        1.0
    };

    // ── ④ 행 센터링 + 편견 상쇄 ──
    let mut net = vec![vec![f32::MIN; n]; m];
    for ki in 0..m {
        for i in 0..n {
            if raw_b[ki][i] == f32::MIN { continue; }
            let zb = (raw_b[ki][i] - mu_b[ki]) / sd;
            let zp = if raw_p[ki][i] == f32::MIN {
                0.0
            } else {
                ((raw_p[ki][i] - mu_p[ki]) / sd).max(0.0)
            };
            net[ki][i] = zb - zp;
        }
    }

    // ── ⑤ 열 센터링 : '전 개념에 반응하는 잉크 패치' 공통 성분 제거 ──
    for i in 0..n {
        let mut s = 0.0f32; let mut c = 0usize;
        for ki in 0..m {
            if net[ki][i] == f32::MIN { continue; }
            s += net[ki][i]; c += 1;
        }
        if c < 2 { continue; }
        let mean = s / c as f32;
        for ki in 0..m {
            if net[ki][i] == f32::MIN { continue; }
            net[ki][i] -= mean;
        }
    }

    println!(
        "    📐 [BANK-NEUTRAL] 키 {}개 | 편견 해석 성공 {}/{} | 편견 그룹 {}개 | 기준선 패치 {}개 | pooled σ {:.5} | √(2 ln N) 차감 폐기",
        m, resolved, m, prej_pool.len(), base.len(), sd
    );
    if resolved == 0 && !bank.prejudice.is_empty() {
        // 편견 뱅크가 있는데 한 건도 매칭되지 않으면 조용히 무력화된 상태입니다.
        // 구버전에서 실제로 이 상태였고, 로그가 없어 발견이 늦었습니다.
        println!(
            "    🚨 [PREJUDICE ORPHAN] 편견 구 {}개가 어떤 키와도 연결되지 않았습니다. key 축이 어긋났는지 확인하십시오.",
            bank.prejudice.len()
        );
    }

    (order, net)
}

pub fn pool_row_runs(
    grid: &PatchGrid,
    legible: &crate::models::siglip2::legibility::LegibilityMap,
) -> (Vec<Vec<f32>>, Vec<Vec<usize>>) {
    let mut embs: Vec<Vec<f32>> = Vec::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    let d = grid.patches.first().map(|p| p.len()).unwrap_or(0);
    if d == 0 { return (embs, members); }

    for r in 0..grid.grid_rows {
        let mut run: Vec<usize> = Vec::new();
        for c in 0..=grid.grid_cols {
            let idx = if c < grid.grid_cols { Some(r * grid.grid_cols + c) } else { None };
            let ok = idx.map_or(false, |i| legible.is_legible(i));
            if ok {
                run.push(idx.unwrap());
                continue;
            }
            if run.is_empty() { continue; }
            // 런 종료 → 평균 풀링 + L2
            let mut v = vec![0.0f32; d];
            for &i in &run {
                for k in 0..d { v[k] += grid.patches[i][k]; }
            }
            let inv = 1.0 / run.len() as f32;
            for k in 0..d { v[k] *= inv; }
            let nrm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if nrm > 1e-8 { for k in 0..d { v[k] /= nrm; } }
            embs.push(v);
            members.push(std::mem::take(&mut run));
        }
    }
    (embs, members)
}
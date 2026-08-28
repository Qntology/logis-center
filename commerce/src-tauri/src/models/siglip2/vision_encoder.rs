// =====================================================================
// 🌟 [SigLIP2 VISION ENCODER PIPELINE]
// ---------------------------------------------------------------------
//  이 파일은 기획서의 STEP 1 / STEP 2 를 담당합니다.
//
//   STEP 1 : Doc Type NMS Battle
//            원본 이미지 패치 임베딩  ×  서식 그룹/코드 앵커 뱅크
//            → SURPRISAL 채점 → 그룹 확정 → 코드 확정
//
//   STEP 2 : Column Cosine Matching
//            패치 임베딩  ×  스키마 컬럼 앵커 뱅크
//            → 카테고리별 2D 히트맵 생성
//
//  ── 왜 텍스트 NMS 와 같은 함수를 쓰는가 ──
//   scheduler.rs STEP A 는 27종 서식을 '그룹 → 코드' 2뎁스로 좁히면서
//   ai_utils::surprisal_dual_scores 로 채점합니다.
//   비전도 정확히 같은 함수를 씁니다. 판정 대상이
//   'PUG 라인 임베딩' 에서 '이미지 패치 임베딩' 으로 바뀔 뿐입니다.
//   그래야 새 서식이 추가돼도 logic.rs 의 사전 한 곳만 고치면 되고,
//   텍스트 트랙과 비전 트랙의 판정 근거가 갈라지지 않습니다.
//
//  ── 왜 parsing.rs 의 get_trade_doc_slice_config 를 대체하는가 ──
//   그 함수는 서식마다 (카테고리, top, bottom) 비율을 손으로 적어 둔 표입니다.
//   실제 문서는 레이아웃이 제각각이라 고정 비율 크롭은
//   ① 다른 카테고리 영역을 통째로 삼키고
//   ② 정작 필요한 필드는 절반이 잘려 나갑니다.
//   여기서는 '컬럼 이름 텍스트' 와 '이미지 패치' 의 코사인으로
//   그 카테고리가 실제로 인쇄된 위치를 찾아냅니다.
// =====================================================================

use candle_core::{DType, Device, Tensor};
use image::DynamicImage;

use super::preprocessor::{preprocess_image, PreprocessedImage};
use super::{Siglip2Config, Siglip2Model};
use crate::logic::TRADE_DOC_TITLES;
use crate::utils::ai_utils::{split_bias_phrases_full, surprisal_dual_scores};

// =====================================================================
// 텍스트 앵커 뱅크
// =====================================================================

/// (category, key, phrase) 트리플을 SigLIP2 텍스트 공간으로 인코딩한 결과.
///
/// 🌟 [Arc 공유] 같은 구가 여러 (category, key) 에 반복 등장하므로
///    벡터 실체는 한 벌만 두고 참조만 나눠 갖습니다.
///    (구버전은 13,943회 복사로 문서당 64MB 를 힙에 쏟아부었습니다)
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

/// SigLIP2 텍스트 인코더로 구 목록을 벡터화합니다.
///
/// 반환값은 L2 정규화된 f32 벡터입니다.
/// (`ai_utils::cosine_similarity` 가 내부에서 다시 정규화하지만,
///  히트맵 계산에서 직접 dot product 를 쓰므로 여기서 확정합니다)
pub fn encode_phrases(
    model: &Siglip2Model,
    phrases: &[String],
) -> anyhow::Result<Vec<Vec<f32>>> {
    if phrases.is_empty() {
        return Ok(Vec::new());
    }
    let text = model
        .text
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("SigLIP2 text encoder is not loaded"))?;
    let tok = model
        .tokenizer
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("SigLIP2 tokenizer is not loaded"))?;

    let mut out: Vec<Vec<f32>> = Vec::with_capacity(phrases.len());

    // 배치 크기는 VRAM 상한을 고려해 32 로 고정합니다.
    // (64토큰 × 1152차원 × 27층이므로 32가 안전선입니다)
    for chunk in phrases.chunks(32) {
        let owned: Vec<String> = chunk.to_vec();
        let batch = tok.encode_batch(&owned)?;
        let t = text.encode_batch(&batch, &model.device)?; // (b, D)
        let t = t.to_dtype(DType::F32)?;
        let rows: Vec<Vec<f32>> = t.to_vec2::<f32>()?;
        for mut r in rows {
            l2_normalize(&mut r);
            out.push(r);
        }
    }

    Ok(out)
}

/// (category, key, phrase) 정의 목록을 뱅크로 변환합니다.
/// 동일 문자열 구는 1회만 임베딩하고 재사용합니다.
pub fn build_anchor_bank(
    model: &Siglip2Model,
    bias_defs: &[(String, String, String)],
    prej_defs: &[(String, String, String)],
) -> anyhow::Result<AnchorBank> {
    use std::collections::HashMap;

    // 🌟 [O(N²) → O(N)] 구버전은 uniq 구축과 lookup 을 둘 다 선형 탐색으로 했습니다.
    //    실측 뱅크(bias 345 + prej 13,598 = 13,943구, uniq 345)에서
    //      uniq 구축 : 13,943 × 345/2 ≈ 240만 문자열 비교
    //      lookup    : 13,943 × 345/2 ≈ 240만 문자열 비교
    //    필드가 늘면 이 둘이 제곱으로 커져 STEP 3 이 CPU 에서 먼저 죽습니다.
    //    HashMap 색인으로 둘 다 O(1) 조회가 됩니다.
    let mut index: HashMap<&str, usize> = HashMap::new();
    let mut uniq: Vec<String> = Vec::new();
    for (_, _, p) in bias_defs.iter().chain(prej_defs.iter()) {
        if !index.contains_key(p.as_str()) {
            index.insert(p.as_str(), uniq.len());
            uniq.push(p.clone());
        }
    }
    let embs = encode_phrases(model, &uniq)?;

    // 🌟 [CLONE 제거] 구버전은 lookup 이 호출될 때마다 1152차원 f32 벡터를 복사했습니다.
    //    13,943회 × 1152 × 4B ≈ 64MB 힙 할당이 매 문서마다 발생했고,
    //    그 압박이 [VISION-ADAPTIVE] Free VRAM 669MB 직전 상황에 기여했습니다.
    //    Arc 로 공유하면 같은 구를 몇 번 참조해도 복사가 일어나지 않습니다.
    let shared: Vec<std::sync::Arc<Vec<f32>>> =
        embs.into_iter().map(std::sync::Arc::new).collect();
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

// =====================================================================
// 패치 임베딩 산출
// =====================================================================

/// 이미지 1장의 비전 산출물. STEP 1~4 가 모두 이 구조체를 소비합니다.
pub struct PatchGrid {
    /// 각 패치의 공유공간 임베딩 (L2 정규화 완료). len = rows * cols
    pub patches: Vec<Vec<f32>>,
    /// 이미지 전체 풀링 벡터 (L2 정규화 완료). LanceDB 비전 검색용.
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
    // 🌟 [OPTIONAL VISION] 텍스트 전용으로 로드된 인스턴스에서는 이미지 처리가 불가능합니다.
    //    조용히 0 벡터를 돌려주면 히트맵이 전부 무의미해지므로 명시적으로 실패시킵니다.
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

/// 이미지 전체 풀링 벡터만 필요할 때 (LanceDB 비전 인덱싱 / 검색 쿼리).
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
    let out = vision.forward(&px, pre.grid_rows, pre.grid_cols)?;
    let pooled_t = out.pooled.to_dtype(DType::F32)?;
    let mut pooled: Vec<f32> = pooled_t.squeeze(0)?.to_vec1::<f32>()?;
    l2_normalize(&mut pooled);
    Ok(pooled)
}

/// 텍스트 한 줄을 SigLIP2 공유공간 벡터로 (검색 쿼리용).
pub fn encode_query_text(model: &Siglip2Model, text: &str) -> anyhow::Result<Vec<f32>> {
    let v = encode_phrases(model, &[text.to_string()])?;
    v.into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("SigLIP2 text encoding returned nothing"))
}

// =====================================================================
// 🌟 [STEP 1] Doc Type NMS Battle
// =====================================================================

#[derive(Debug, Clone)]
pub struct DocTypeVerdict {
    pub group: String,
    pub group_score: f32,
    pub group_margin: f32,
    pub code: String,
    pub code_score: f32,
    pub code_margin: f32,
    /// 그룹 판정에서 편견 우세로 탈락한 패치 비율. 진단용.
    pub prejudice_dropped: usize,
    /// 코드 후보와 각 점수. LLM 재판정 프롬프트에 그대로 실립니다.
    pub code_candidates: Vec<(String, f32)>,
    /// 🌟 [MODE GATE] 상단 밴드에서 서식 전문이 인쇄되어 확정되었는가.
    ///    mode 리라우트(commerce→trading)는 이 플래그가 true 일 때만 허용됩니다.
    ///    본문 코사인만으로는 상품 사진과 무역 문서를 안전히 가를 수 없습니다.
    pub title_confirmed: bool,
    /// 🌟 확정된 서식 전문 문자열. 컬럼 히트맵의 TITLE PREJUDICE 로 재사용됩니다.
    pub title_text: String,
}

/// 🌟 [BANK-NEUTRAL KEY SCORES] score_patches_bank_neutral 출력을 (key → 패치 최댓값) 으로 축소합니다.
///    √(2 ln N) 차감이 없으므로 앵커 구 수에 무관한 공정한 경쟁이 됩니다.
///    반환: (정렬된 (key, score) 목록, 활성 패치 수)
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

/// 패치 임베딩 전체를 뱅크에 채점하여 (key → 최고 surprisal) 맵을 만듭니다.
///
/// 🌟 [PREJUDICE DROP] scheduler.rs 의 [FRONT-CLEAN] / [NAV PRE-FILTER] 와 같은 역할입니다.
///    어떤 패치의 최고 편견 점수가 최고 bias 점수보다 높으면
///    그 패치는 UI 껍데기(로고 / 워터마크 / 여백)로 보고 판정에서 제외합니다.
///    좌표를 손으로 적지 않고, 편견 사전만으로 노이즈를 걷어냅니다.
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

        // 🌟 [LOG] 패치별 코사인 상세 — 상위 3개 키의 원시 코사인 + surprisal
        //    전체 252패치를 출력하면 과다하므로, 채점 카운터로 조절합니다.
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

// =====================================================================
// 🌟 [DOC TITLE GATE] 무역 서식 '전문(풀 네임)' 에 의한 결정론 게이트
// ---------------------------------------------------------------------
//   ── 왜 접두어(code prefix)가 아니라 전문인가 ──
//    'CI' 같은 짧은 코드는 이 서식_own 번호 접두어일 뿐 아니라
//    다른 서식에도 인쇄됩니다. ED 는 reference_invoice 셀에 'CI-43726' 을
//    그대로 찍고, BL / PL 도 인보이스 번호를 인쇄합니다.
//    접두어 매칭은 그 서식들에서 결정론적으로 오발화합니다.
//    반면 전문 'COMMERCIAL INVOICE' 는 이 서식의 헤더(제목)이며,
//    다른 서식은 이 문구를 헤더로 인쇄하지 않습니다.
//    따라서 게이트의 키는 전문, 값은 코드입니다.
//
//   ── 왜 상단 밴드만 보는가 ──
//    헤더는 문서 상단에 인쇄됩니다. PL 이 본문에 "as per commercial invoice"
//    라고 적는 경우가 있으므로 상단 30% 행만 제목 밴드로 봅니다.
//    이것은 레이아웃의 구조적 사실이며 매직 상수가 아닙니다.
//
//   ── 왜 surprisal 0 게이트인가 ──
//    score_patches 와 동일한 극값 기준선을 재사용합니다.
//    0 = N개를 무작위로 뽑은 기대 최댓값. 전문이 그 기대치를 넘을 때만
//    '인쇄되어 있다' 고 인정합니다. LLM 을 부르지 않으므로 동일 입력은
//    항상 동일 출력을 냅니다.
//    동명 서식(BC/BK, PC/PHYTO, INS/IP)은 마진이 0 이 되어 게이트가
//    스스로 거부하고 벡터 판정에 위임합니다.

/// 🌟 [TITLE GATE] 산출물. code/group 은 전문에서 역引き한 확정값입니다.
struct TitleGateVerdict {
    code: String,
    group: String,
    title: String,
    score: f32,
    margin: f32,
}

/// 🌟 [TITLE GATE] 상단 밴드 패치만을 서식 전문 뱅크에 채점합니다.
///    일부 패치라도 전문의 surprisal 이 0 을 넘고, 1·2위 마진이 양수일 때만
///    확정값을 돌려줍니다. 아니면 None(벡터 판정 유지) 입니다.
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
    let mut patch_contributions: Vec<(usize, usize, usize, String, f32)> = Vec::new();

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
                // 🌟 [LOG] 타이틀 게이트에서 각 전문 키의 최고점을 갱신한 패치 기록
                if patch_contributions.len() < 20 {
                    patch_contributions.push((idx, r, c, s.key.clone(), s.surprisal));
                }
            }
        }
    }

    // 🌟 [LOG] 타이틀 게이트 스캔 요약
    emit(&format!(
        "     🔍 [TITLE GATE SCAN RESULT] 스캔 {} | 활성 {} | 양수 {} | 전문 키 {}개 발견",
        scanned_patches, active_patches, positive_patches, best.len()
    ));

    if !patch_contributions.is_empty() {
        emit("     🔍 [TITLE GATE CONTRIBUTORS] 전문별 최고점 갱신 패치:");
        for (idx, r, c, key, sur) in patch_contributions.iter() {
            emit(&format!(
                "       ↳ patch[{}] r{}c{} → '{}' {:+.4}",
                idx, r, c, key, sur
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
                .filter(|(c, _, _)| c == gn)
                .map(|(_, _, p)| p.as_str())
                .collect();
            let sample: Vec<&str> = phrases.iter().take(4).copied().collect();
            let prej_cnt = g_prej.iter().filter(|(c, _, _)| c == *gn).count();
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

    // 🌟 [SPLIT 1회 캐시 + 편견 축약]
    //
    //  ── 구버전의 두 가지 낭비 ──
    //   ① split_bias_phrases_full 이 후보수² 회 호출됩니다.
    //      27개 후보면 729회, Part 20 확장 후 56개면 3,136회입니다.
    //   ② c_prej 크기가 후보수 × (후보수-1) × 평균구수 로 폭발합니다.
    //      56개 후보 × 55 × 평균 12구 ≈ 37,000구. dedup 도 없습니다.
    //
    //  ── 왜 축약해도 결과가 같은가 ──
    //   surprisal_dual_scores 는 (category, key) 그룹의 편견 최댓값 하나만 감산합니다.
    //   'CI' 의 편견은 "CI 를 제외한 나머지 전부" 인데,
    //   그 최댓값은 곧 "전체 코드 앵커 중 CI 것을 뺀 최댓값" 입니다.
    //   따라서 전체 코드 앵커를 한 벌만 두고, 채점 시 자기 코드 구를 제외하면
    //   같은 값을 얻으면서 저장량이 후보수 배 줄어듭니다.
    //   여기서는 구조 변경 범위를 좁히기 위해 '중복 구 제거' 만 적용합니다.
    //   (코드 앵커는 서식마다 문구가 거의 겹치지 않아 실질 절감은 split 호출 쪽이 큽니다)
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
        for (c, _, _) in c_bias.iter() {
            if !code_names.contains(&c.as_str()) { code_names.push(c.as_str()); }
        }
        let sample_limit = code_names.len().min(8);
        for cn in code_names.iter().take(sample_limit) {
            let phrases: Vec<&str> = c_bias.iter()
                .filter(|(c, _, _)| c == cn)
                .map(|(_, _, p)| p.as_str())
                .collect();
            let sample: Vec<&str> = phrases.iter().take(3).copied().collect();
            let prej_cnt = c_prej.iter().filter(|(c, _, _)| c == *cn).count();
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
    emit: &dyn Fn(&str),
) -> anyhow::Result<Vec<CategoryHeatmap>> {
    use std::collections::HashMap;

    // ── 1) 카테고리 × 필드 앵커 구 수집 ──
    //    필드 → 카테고리 매핑은 logic.rs 가 소유합니다.
    let schema_fields = crate::parsing::get_detail_schema_fields(doc_type, "", doc_lang);

    let mut bias_defs: Vec<(String, String, String)> = Vec::new();
    let mut field_to_cat: HashMap<String, String> = HashMap::new();

    for (fname, _, bias_target, _) in schema_fields.iter() {
        let cat = crate::logic::trade_field_category(fname);
        if cat.is_empty() {
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
        for p in split_bias_phrases_full(&sem) {
            if crate::utils::ai_utils::is_value_example_phrase(&p) {
                value_cnt += 1;
            } else {
                label_cnt += 1;
            }
            bias_defs.push((cat.to_string(), fname.clone(), p));
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

    // ── 2) 편견 : 다른 카테고리의 bias 구 + 시각 노이즈 ──
    //
    // 🌟 [PREJUDICE COLLAPSE — 카테고리 단위]
    //
    //  ── 무엇이 문제였나 ──
    //   구버전은 편견을 '필드 단위' 로 만들었습니다.
    //     for (cat, field, _) in bias_defs   ← 구(phrase)마다 한 번씩 도는 루프
    //         for (other_cat, _, p) in bias_defs
    //   dedup 덕분에 (cat, field, phrase) 단위로 접히긴 했지만,
    //   같은 카테고리의 모든 필드가 '완전히 동일한 편견 집합' 을 중복 보유했습니다.
    //   header 의 11개 필드가 각각 "header 아닌 229구" 를 따로 들고 있었던 셈입니다.
    //   그 결과가 실측 로그의 편견 구 13,598개입니다.
    //
    //  ── 왜 카테고리 단위로 접어도 결과가 같은가 ──
    //   surprisal_dual_scores 는 편견을 이렇게 씁니다.
    //     if let Some(pi) = p_order.iter().position(|(a,b)| a==c && b==k) {
    //         if ps > 0.0 { sc -= ps; }
    //     }
    //   (category, key) 그룹의 '최댓값 하나' 만 감산에 쓰입니다.
    //   그런데 같은 카테고리의 모든 필드가 동일한 구 집합을 갖고 있었으므로
    //   그 최댓값도 필드와 무관하게 항상 같은 값이었습니다.
    //   따라서 key 를 필드명에서 카테고리 공용 키로 바꿔도 감산량이 변하지 않습니다.
    //
    //  ── 절감 ──
    //   편견 구 = Σ_cat (전체구수 - 구수_cat) + 카테고리수 × 21
    //   44필드 기준 13,598 → 약 2,100 (6.5배 감소)
    //   dedup 비교는 O(N²) 이므로 9,200만 → 약 220만 (42배 감소)
    //
    //  ⚠️ [CONTRACT] 아래 3-단계 채점 루프가 편견 키를 '카테고리명' 으로 조회해야 합니다.
    //     bias 쪽 key 는 필드명 그대로 두고, 편견만 카테고리명을 씁니다.
    //     surprisal_dual_scores 가 (category, key) 쌍으로 매칭하므로
    //     bias 의 (cat, field) 와 편견의 (cat, cat) 은 서로 만나지 않습니다.
    //     → 이 계약을 지키기 위해 아래 apply_category_prejudice() 로 감산을 직접 수행합니다.
    let cats: Vec<String> = {
        let mut v: Vec<String> = Vec::new();
        for (c, _, _) in bias_defs.iter() {
            if !v.iter().any(|x| x == c) {
                v.push(c.clone());
            }
        }
        v
    };

    // 카테고리별 구 집합을 미리 색인해 둡니다. (O(N) 1회 순회)
    let mut cat_phrases: HashMap<String, Vec<String>> = HashMap::new();
    for (c, _, p) in bias_defs.iter() {
        let e = cat_phrases.entry(c.clone()).or_insert_with(Vec::new);
        if !e.iter().any(|x| x == p) {
            e.push(p.clone());
        }
    }

    // 🌟 편견은 카테고리당 1세트. key 에도 카테고리명을 넣어 그룹을 만듭니다.
    let mut prej_defs: Vec<(String, String, String)> = Vec::new();
    for cat in cats.iter() {
        let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for (other_cat, phrases) in cat_phrases.iter() {
            if other_cat == cat {
                continue;
            }
            for p in phrases.iter() {
                if seen.insert(p.as_str()) {
                    prej_defs.push((cat.clone(), cat.clone(), p.clone()));
                }
            }
        }
        for p in split_bias_phrases_full(crate::logic::VISION_CHROME_ANCHOR) {
            if seen.contains(p.as_str()) {
                continue;
            }
            prej_defs.push((cat.clone(), cat.clone(), p));
        }
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

    // ── 3) 패치별 채점 → 카테고리 히트맵 ──
    //    surprisal_dual_scores 의 key 는 '필드명' 이므로,
    //    같은 카테고리에 속한 필드들의 최댓값을 그 카테고리 점수로 씁니다.
    let empty_names: Vec<String> = Vec::new();
    let empty_banks: Vec<Vec<Vec<f32>>> = Vec::new();
    let empty_skip: Vec<bool> = Vec::new();
    // 🌟 편견 뱅크를 bias 자리에 넣어 채점할 때 쓰는 빈 편견 축입니다.
    //    (편견의 편견은 없으므로 비어 있어야 합니다)
    let empty_prej: Vec<(String, String, std::sync::Arc<Vec<f32>>)> = Vec::new();

    let n = grid.len();
    let mut cat_scores: HashMap<String, Vec<f32>> = HashMap::new();
    let mut cat_top: HashMap<String, (String, f32)> = HashMap::new();
    for c in cats.iter() {
        cat_scores.insert(c.clone(), vec![f32::MIN; n]);
        cat_top.insert(c.clone(), (String::new(), f32::MIN));
    }

    // 🌟 [LOG] 패치 채점 추적 카운터
    let mut patch_scored_count = 0usize;
    let mut patch_positive_field_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut patch_prej_override_count = 0usize;

    // 🌟 [BANK-NEUTRAL] 필드별 √(2 ln N) 차감을 폐기하고
    //    행/열 이중 센터링으로 뱅크 크기·응집도 편향을 제거합니다.
    //    (실측: reference_sr 1구가 status 19구보다 2.4점 공짜 우위)
    let (keys, matrix) = score_patches_bank_neutral(grid, &bank, legibility);
    for (ki, fname) in keys.iter().enumerate() {
        let cat = match field_to_cat.get(fname) {
            Some(c) => c.clone(),
            None => continue,
        };
        for i in 0..n {
            let v = matrix[ki][i];
            if v == f32::MIN { continue; }
            if let Some(slot) = cat_scores.get_mut(&cat) {
                if v > slot[i] { slot[i] = v; }
            }
            if let Some(t) = cat_top.get_mut(&cat) {
                if v > t.1 { *t = (fname.clone(), v); }
            }
        }
    }

    // 🌟 [LOG] 히트맵 채점 요약
    emit(&format!(
        "    📊 [HEATMAP SCORING SUMMARY] 채점 패치 {} | 편견 상쇄 발생 {}회",
        patch_scored_count, patch_prej_override_count
    ));
    {
        let mut pf_detail: Vec<String> = Vec::new();
        for (cat, cnt) in patch_positive_field_counts.iter() {
            pf_detail.push(format!("{}({})", cat, cnt));
        }
        pf_detail.sort();
        emit(&format!(
            "    📊 [HEATMAP POSITIVE FIELDS] 카테고리별 양수 패치 수: {}",
            pf_detail.join(" | ")
        ));
    }

    let mut out: Vec<CategoryHeatmap> = Vec::with_capacity(cats.len());
    for c in cats.iter() {
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

/// 🌟 [BANK-NEUTRAL SCORING] 뱅크 크기 편향을 구조적으로 제거한 패치 채점.
///
///  ── 왜 √(2 ln N) 을 버리는가 ──
///   극값이론의 E[max] ≈ √(2 ln N) 은 '뱅크 구가 서로 독립' 일 때만 성립합니다.
///   그런데 한 필드의 앵커 구는 "Shipper / exporter / consignor" 처럼 동의어 나열이라
///   실효 표본 수가 N 보다 훨씬 작습니다. 큰 뱅크를 과잉 처벌합니다.
///
///   실측(로그 대조):
///     reference_sr(1구) 차감 ≈ 0     vs  status(19구) 차감 2.427
///     → 'reference sr' 이라는 무의미 구가 header 를 대표하고
///       header 크롭이 문서 하단으로 착지 → doc_number 소실 → task_id 폴백
///
///  ── 대체 원리 ──
///   각 뱅크의 '자기 기준선' 을 패치 축으로 실측해 뺍니다.
///     centered[k][i] = (raw[k][i] - μ_k) / σ_pooled
///   μ_k 는 그 뱅크가 이 문서에서 보이는 평균 반응이므로,
///   뱅크가 크든 작든 응집되든 자동으로 상쇄됩니다.
///   σ 는 뱅크마다 쓰지 않고 전역 pooled 를 씁니다.
///   (1구 뱅크는 σ_k 가 0 에 가까워 z 가 무한히 부풀기 때문입니다)
///
///   그 다음 패치 축으로 한 번 더 센터링합니다.
///     final[k][i] = centered[k][i] - mean_k(centered[·][i])
///   '잉크가 빽빽해서 전 개념에 반응하는 패치' 의 공통 성분이 제거됩니다.
///   이것은 scheduler.rs 의 double_center_matrix 와 동일한 도구입니다.
///
///  ── 판독성 ──
///   μ_k / σ 는 '판독 가능 패치' 로만 계산합니다.
///   여백 141개가 통계를 끌어내리면 모든 뱅크가 동시에 부풀어 변별력이 사라집니다.
///
///  반환: (keys, matrix[key][patch])
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

    let dot = |a: &[f32], b: &[f32]| -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    };

    // ── ② 원시 Max-Pool 행렬 ──
    let m = order.len();
    let mut raw_b = vec![vec![f32::MIN; n]; m];
    let mut raw_p = vec![vec![f32::MIN; n]; m];

    for (ki, key) in order.iter().enumerate() {
        let bi = &bias_idx[key];
        let pi = prej_idx.get(key);
        for i in 0..n {
            let p = &grid.patches[i];
            if p.iter().all(|&v| v == 0.0) { continue; }
            let mut mb = f32::MIN;
            for &j in bi {
                let s = dot(p, &bank.bias[j].2);
                if s > mb { mb = s; }
            }
            raw_b[ki][i] = mb;
            if let Some(pl) = pi {
                let mut mp = f32::MIN;
                for &j in pl {
                    let s = dot(p, &bank.prejudice[j].2);
                    if s > mp { mp = s; }
                }
                raw_p[ki][i] = mp;
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
        "    📐 [BANK-NEUTRAL] 키 {}개 | 기준선 패치 {}개 | pooled σ {:.5} | √(2 ln N) 차감 폐기",
        m, base.len(), sd
    );

    (order, net)
}

/// 🌟 [ROW RUN POOLING] 판독 가능 패치의 가로 연속 구간을 하나의 '텍스트 라인' 으로 봅니다.
///
///  ── 왜 필요한가 ──
///   16×16 패치는 원본 57×57px 입니다. 글자 2~3개밖에 안 들어갑니다.
///   "customs export filing" 같은 문서 단위 개념을 그 조각과 코사인 비교하는 것은
///   성립하지 않습니다. scheduler.rs STEP A 가 PUG '라인' 을 단위로 삼는 이유와 같습니다.
///
///  ── 반환 ──
///   (pooled_embeddings, member_patch_indices)
///   런 점수를 소속 패치에 되돌려 칠하면 위치 해상도를 잃지 않습니다.
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
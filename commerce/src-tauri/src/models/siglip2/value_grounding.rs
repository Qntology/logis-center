// =====================================================================
// 🌟 [VALUE GROUNDING] 추출된 값이 '그 크롭 안에 실제로 인쇄되어 있는가'
// ---------------------------------------------------------------------
//  ── 실측 사고 ──
//   ① reference_invoice = "CI-2026-08001"
//      → 문서 어디에도 없습니다. bias.json 설명문의 (e.g. CI-2026-08001) 복사입니다.
//   ② voyage_number = "26"
//      → logistics 크롭에 항차가 없는데, 같은 크롭의 CI-43726 뒤 두 자리를 떼어냈습니다.
//   ③ recipient_name = "BUYER (IF NOT CONSIGNEE)"
//      → 빈 박스의 헤더 라벨을 값으로 읽었습니다.
//
//  ── 판정 원리 ──
//   SigLIP2 텍스트 인코더와 비전 패치는 같은 1152차원 공간에 있습니다.
//   값 텍스트를 인코딩해 252개 패치 전체와 코사인을 잰 뒤,
//   그 값이 '자기 크롭 안' 에서 무작위 기대치를 넘는지 봅니다.
//
//     surprisal_in  = (max_in  - μ)/σ - √(2 ln N_in)
//     surprisal_out = (max_out - μ)/σ - √(2 ln N_out)
//
//   μ/σ 는 그 값 하나에 대한 전체 패치 코사인 분포이므로,
//   값마다 스케일이 달라도 자동으로 정규화됩니다.
//   √(2 ln N) 차감이 크롭 크기 차이를 상쇄합니다.
//   0 은 극값이론에서 유도된 값이라 매직 상수가 아닙니다.
//
//  ── 3중 게이트 ──
//   G-A 접지  : surprisal_in <= 0            → 폐기 (크롭 안에 근거 없음)
//   G-B 판독성: in-crop argmax 패치가 판독불가/여백 → 폐기 (블러·빈 칸에서 읽어낸 값)
//   G-C 유출  : surprisal_out > surprisal_in → 경고만 (다른 크롭 소유일 가능성)
//
//  ── 왜 G-C 는 폐기하지 않는가 ──
//   문서에 같은 문자열이 두 번 인쇄되는 경우(예: DATE 가 상단과 서명란에 모두 있음)가
//   정상적으로 존재합니다. 유출은 크롭 배정 문제이지 값 자체의 오류가 아니므로
//   기록만 남기고 STEP 4.5 크롭 감사에 피드백합니다.
// =====================================================================

use crate::models::siglip2::legibility::LegibilityMap;
use crate::utils::ai_utils::{cosine_similarity, gumbel_expected_z};

#[derive(Debug, Clone)]
pub struct GroundingClaim {
    pub category: String,
    pub field: String,
    pub value: String,
    /// 이 값이 나온 크롭의 픽셀 bbox
    pub bbox: (u32, u32, u32, u32),
}

#[derive(Debug, Clone)]
pub struct GroundingVerdict {
    pub category: String,
    pub field: String,
    pub value: String,
    pub surprisal_in: f32,
    pub surprisal_out: f32,
    pub top_patch: usize,
    pub top_legible: bool,
    pub accepted: bool,
    pub reason: String,
}

/// 픽셀 bbox 에 속하는 패치 인덱스 목록 (패치 중심 기준)
fn patches_in_bbox(
    bbox: (u32, u32, u32, u32),
    rows: usize,
    cols: usize,
    orig_w: u32,
    orig_h: u32,
) -> Vec<usize> {
    let mut out = Vec::new();
    if rows == 0 || cols == 0 || orig_w == 0 || orig_h == 0 {
        return out;
    }
    let cw = orig_w as f32 / cols as f32;
    let ch = orig_h as f32 / rows as f32;
    for r in 0..rows {
        for c in 0..cols {
            let cx = (c as f32 + 0.5) * cw;
            let cy = (r as f32 + 0.5) * ch;
            if cx >= bbox.0 as f32 && cx <= bbox.2 as f32
                && cy >= bbox.1 as f32 && cy <= bbox.3 as f32
            {
                out.push(r * cols + c);
            }
        }
    }
    out
}

/// 🌟 [VERIFY] 값 목록 전체를 한 번에 검증합니다.
///
///  ── 왜 배치인가 ──
///   SigLIP2 텍스트 인코더는 1.4GB 입니다. 값마다 로드/해제하면 파이프라인이 죽습니다.
///   호출부가 STEP 5 종료 후 Qwen 을 내리고 SigLIP2 를 한 번만 올린 상태에서
///   이 함수를 1회 호출하도록 설계했습니다.
///
///  ── encode_fn ──
///   텍스트 → 1152차원 L2 정규화 벡터. 실패 시 빈 벡터.
pub fn verify_claims<F>(
    claims: &[GroundingClaim],
    patch_embs: &[Vec<f32>],
    grid_rows: usize,
    grid_cols: usize,
    orig_w: u32,
    orig_h: u32,
    legibility: &crate::models::siglip2::legibility::LegibilityMap,
    encode_fn: F,
    emit: &dyn Fn(&str),
) -> Vec<GroundingVerdict>
where
    F: Fn(&str) -> Vec<f32>,
{
    use crate::models::siglip2::legibility::PatchLegibility;

    let n = patch_embs.len();
    let mut out: Vec<GroundingVerdict> = Vec::with_capacity(claims.len());
    if n == 0 || claims.is_empty() {
        return out;
    }

    for claim in claims.iter() {
        let v = claim.value.trim();
        // 값 자체가 없으면 검증 대상이 아닙니다.
        if v.is_empty() {
            continue;
        }

        let emb = encode_fn(v);
        if emb.len() != patch_embs[0].len() || emb.iter().all(|&x| x == 0.0) {
            out.push(GroundingVerdict {
                category: claim.category.clone(),
                field: claim.field.clone(),
                value: v.to_string(),
                surprisal_in: 0.0,
                surprisal_out: 0.0,
                top_patch: 0,
                top_legible: true,
                accepted: true,
                reason: "임베딩 생성 실패 — 검증 보류(값 유지)".to_string(),
            });
            continue;
        }

        // 전 패치 코사인
        let sims: Vec<f32> = patch_embs.iter().map(|p| cosine_similarity(&emb, p)).collect();
        let mean: f32 = sims.iter().sum::<f32>() / n as f32;
        let var: f32 = sims.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n as f32;
        let std = var.sqrt().max(1e-6);

        let inside = patches_in_bbox(claim.bbox, grid_rows, grid_cols, orig_w, orig_h);
        if inside.is_empty() {
            out.push(GroundingVerdict {
                category: claim.category.clone(),
                field: claim.field.clone(),
                value: v.to_string(),
                surprisal_in: 0.0,
                surprisal_out: 0.0,
                top_patch: 0,
                top_legible: true,
                accepted: true,
                reason: "크롭에 대응하는 패치 없음 — 검증 보류(값 유지)".to_string(),
            });
            continue;
        }

        let mut max_in = f32::MIN;
        let mut top_patch = inside[0];
        for &i in inside.iter() {
            if sims[i] > max_in {
                max_in = sims[i];
                top_patch = i;
            }
        }
        let mut max_out = f32::MIN;
        let mut n_out = 0usize;
        for i in 0..n {
            if inside.contains(&i) {
                continue;
            }
            n_out += 1;
            if sims[i] > max_out {
                max_out = sims[i];
            }
        }

        let s_in = (max_in - mean) / std - gumbel_expected_z(inside.len());
        let s_out = if n_out == 0 {
            f32::MIN
        } else {
            (max_out - mean) / std - gumbel_expected_z(n_out)
        };

        let top_state = legibility.verdict.get(top_patch).copied()
            .unwrap_or(PatchLegibility::Legible);
        let top_legible = top_state == PatchLegibility::Legible;

        // ── G-A : 접지 ──
        if s_in <= 0.0 {
            emit(&format!(
                "    🚫 [UNGROUNDED] [{}] '{}' = \"{}\" | in {:+.4} ≤ 0 (크롭 안에 근거 없음) → 폐기",
                claim.category, claim.field, v, s_in
            ));
            out.push(GroundingVerdict {
                category: claim.category.clone(),
                field: claim.field.clone(),
                value: v.to_string(),
                surprisal_in: s_in,
                surprisal_out: s_out,
                top_patch,
                top_legible,
                accepted: false,
                reason: "크롭 내부 접지 실패".to_string(),
            });
            continue;
        }

        // ── G-B : 판독성 ──
        if !top_legible {
            let label = match top_state {
                PatchLegibility::Blank => "여백",
                PatchLegibility::Illegible => "블러/마스킹",
                _ => "-",
            };
            emit(&format!(
                "    🚫 [ILLEGIBLE SOURCE] [{}] '{}' = \"{}\" | 최고 일치 패치 r{} c{} 가 {} → 폐기",
                claim.category, claim.field, v,
                top_patch / grid_cols, top_patch % grid_cols, label
            ));
            out.push(GroundingVerdict {
                category: claim.category.clone(),
                field: claim.field.clone(),
                value: v.to_string(),
                surprisal_in: s_in,
                surprisal_out: s_out,
                top_patch,
                top_legible,
                accepted: false,
                reason: format!("근거 패치가 {}", label),
            });
            continue;
        }

        // ── G-C : 유출 경고 (폐기하지 않음) ──
        if s_out > s_in {
            emit(&format!(
                "    ⚠️ [CROSS-CROP] [{}] '{}' = \"{}\" | in {:+.4} < out {:+.4} — 다른 영역 소유 가능",
                claim.category, claim.field, v, s_in, s_out
            ));
        }

        out.push(GroundingVerdict {
            category: claim.category.clone(),
            field: claim.field.clone(),
            value: v.to_string(),
            surprisal_in: s_in,
            surprisal_out: s_out,
            top_patch,
            top_legible,
            accepted: true,
            reason: "접지 확인".to_string(),
        });
    }

    let dropped = out.iter().filter(|v| !v.accepted).count();
    emit(&format!(
        "  ✅ [VALUE GROUNDING] 검증 {}건 | 유지 {} | 폐기 {}",
        out.len(),
        out.len() - dropped,
        dropped
    ));
    out
}

/// 🌟 [GROUNDING v2] SigLIP2 로 '값이 인쇄되어 있는가' 를 묻지 않습니다.
///
///  ── v1 이 왜 실패했나 (실측 26건 중 25건 폐기) ──
///   ① √(2 ln N_in) 이 bbox 크기에 비례해 커집니다.
///      items 는 전체 페이지 크롭이라 N_in=77 → 문턱 z>2.95 (상위 0.2%).
///      "T-Shirt" 같은 정답이 통과할 수 없습니다.
///   ② SigLIP2 는 이미지-캡션 모델입니다. "583948392" 가 이 패치에 인쇄되어
///      있는지는 원리적으로 판정할 수 없습니다. 문자 단위 접지 능력이 없습니다.
///   ③ 코사인이 노이즈라 argmax 가 여백(r13c0)에 착지해 정답을 죽였습니다.
///
///  ── v2 가 실제로 판정할 수 있는 것 ──
///   "그 크롭 안에 읽을 것이 있었는가" 뿐입니다. 이건 픽셀 사실이고 확실합니다.
///     · 판독 가능 패치 0개 → 그 크롭의 전 주장 폐기 (빈 영역에서 창작한 값)
///     · 그 외 → 유지
///   의미 판정은 폐기하고, [SCHEMA ECHO] / [ALREADY CLAIMED] 게이트에 맡깁니다.
///   '틀린 값을 지우는 이득' 보다 '정답 25건을 지우는 손실' 이 압도적으로 큽니다.
pub fn verify_claims_v2(
    claims: &[GroundingClaim],
    grid_rows: usize,
    grid_cols: usize,
    orig_w: u32,
    orig_h: u32,
    legibility: &LegibilityMap,
    emit: &dyn Fn(&str),
) -> Vec<GroundingVerdict> {
    let mut out = Vec::with_capacity(claims.len());
    let mut rejected = 0usize;
    for c in claims {
        let (lg, il, bl) = legibility.count_in_bbox(c.bbox, orig_w, orig_h);
        let accepted = lg > 0;
        if !accepted {
            rejected += 1;
            emit(&format!(
                "    🚫 [EMPTY SOURCE] [{}] '{}' = \"{}\" | 출처 영역 판독가능 {} / 불가 {} / 여백 {} → 폐기",
                c.category, c.field, c.value, lg, il, bl
            ));
        }
        out.push(GroundingVerdict {
            category: c.category.clone(),
            field: c.field.clone(),
            value: c.value.clone(),
            // 🌟 [V2 NEUTRAL FIELDS] v2 는 코사인 접지를 수행하지 않습니다(설계 의도).
            //    이 4개 필드는 v1(G-A/G-B 게이트) 전용이라 v2 에서 계산하지 않으므로
            //    중립 기본값으로 채웁니다. 소비처(apply_grounding_verdicts)는
            //    category/field/value/accepted/reason 만 읽으므로 동작 변화가 없습니다.
            //    top_legible 은 accepted 와 동일하게 둡니다.
            //    (v2 의 수용 조건 자체가 '출처 영역 판독 가능' 이므로)
            surprisal_in: 0.0,
            surprisal_out: 0.0,
            top_patch: 0,
            top_legible: accepted,
            accepted,
            reason: if accepted { String::new() } else { "출처 영역에 읽을 내용이 없음".to_string() },
        });
    }
    emit(&format!(
        "  ✅ [VALUE GROUNDING v2] 검증 {}건 | 유지 {} | 폐기 {}",
        claims.len(), claims.len() - rejected, rejected
    ));
    out
}
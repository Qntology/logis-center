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

pub fn verify_claims_v2(
    claims: &[GroundingClaim],
    grid_rows: usize,
    grid_cols: usize,
    orig_w: u32,
    orig_h: u32,
    legibility: &LegibilityMap,
    doc_lang: &str,
    emit: &dyn Fn(&str),
) -> Vec<GroundingVerdict> {
    // 🌟 [ROLE FIELD EXEMPT] 값 자체가 '역할 이름' 인 축은 라벨 어휘와 정당하게 겹칩니다.
    //    party_role 의 정답이 "Shipper" / "Consignee" 인데 그 둘은 인쇄 라벨이기도 합니다.
    let role_field = |f: &str| -> bool { matches!(f.trim(), "party_role" | "doc_type") };
    let mut out = Vec::with_capacity(claims.len());
    let mut rejected = 0usize;
    for c in claims {
        // 🌟 [LABEL ECHO GATE] 값 자리에 서식의 '박스 라벨' 이 그대로 들어온 경우를 폐기합니다.
        //    "SIGNATORY COMPANY" 는 이미지에 실제로 인쇄되어 있어 판독성 검사는 반드시 통과합니다.
        //    라벨인지 값인지는 픽셀이 아니라 어휘로만 판정할 수 있습니다.
        //    사전은 parsing.rs 의 TRADE_PRINTED_LABELS + TRADE_COLUMN_ALIASES 를 그대로 씁니다.
        if !role_field(&c.field) && crate::parsing::is_printed_label_echo(&c.value, doc_lang) {
            rejected += 1;
            emit(&format!(
                "    🚫 [LABEL ECHO] [{}] '{}' = \"{}\" | 이 문자열은 서식의 인쇄 라벨입니다. 값이 아니므로 폐기합니다.",
                c.category, c.field, c.value
            ));
            out.push(GroundingVerdict {
                category: c.category.clone(),
                field: c.field.clone(),
                value: c.value.clone(),
                surprisal_in: 0.0,
                surprisal_out: 0.0,
                top_patch: 0,
                top_legible: true,
                accepted: false,
                reason: "인쇄 라벨을 값으로 읽음".to_string(),
            });
            continue;
        }
        let (lg, il, bl) = legibility.count_in_bbox(c.bbox, orig_w, orig_h);
        let accepted = lg > 0;
        // 🌟 [SDS / V-4 입력] 출처 영역의 판독성 구성입니다.
        //
        //  ── 왜 남기는가 ──
        //   기획 V-4 는 '흐린 패치의 낮은 코사인' 과 '선명한 패치의 낮은 코사인' 을
        //   같은 값으로 취급하는 문제를 지적합니다. 그 판정을 하려면
        //   '이 값의 출처가 얼마나 읽을 만했는가' 라는 분포가 있어야 하는데,
        //   지금은 이 삼분값이 로그로만 흘러가고 통계에 남지 않았습니다.
        //
        //  ── 폐기 여부와 무관하게 남기는 이유 ──
        //   accepted 인 값의 출처 품질까지 있어야 '판독성이 낮을수록
        //   접지가 실패하는가' 를 상관으로 확인할 수 있습니다.
        //   폐기 건만 모으면 분포가 한쪽 꼬리만 남습니다.
        {
            let total = (lg + il + bl).max(1) as f32;
            crate::utils::score_dynamics::record_baseline(
                "vision.source_legible_ratio",
                lg as f32 / total,
            );
            crate::utils::score_dynamics::record_baseline(
                "vision.source_blank_ratio",
                bl as f32 / total,
            );
        }
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
            surprisal_in: 0.0,
            surprisal_out: 0.0,
            top_patch: 0,
            top_legible: accepted,
            accepted,
            reason: if accepted { String::new() } else { "출처 영역에 읽을 내용이 없음".to_string() },
        });
    }
    {
        let mut ratios: Vec<f32> = Vec::new();
        for c in claims {
            let (lg, il, bl) = legibility.count_in_bbox(c.bbox, orig_w, orig_h);
            let total = (lg + il + bl).max(1) as f32;
            ratios.push(lg as f32 / total);
        }
        if ratios.len() >= 2 {
            crate::utils::score_dynamics::record_decay("vision.grounding", &ratios);
        }
        if !ratios.is_empty() {
            crate::utils::score_dynamics::record_baseline(
                "vision.grounding_reject_ratio",
                rejected as f32 / claims.len().max(1) as f32,
            );
        }
    }
    emit(&format!(
        "  ✅ [VALUE GROUNDING v2] 검증 {}건 | 유지 {} | 폐기 {}",
        claims.len(), claims.len() - rejected, rejected
    ));
    out
}
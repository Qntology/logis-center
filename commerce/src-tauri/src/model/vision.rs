use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use serde_json::{json, Value};
use anyhow::anyhow;
use image::DynamicImage;
use std::io::Cursor;
use base64::prelude::BASE64_STANDARD;
use base64::Engine;
use tauri::Emitter;
use crate::openai_types::*;
use crate::model::merge::{record_grounding_claims, collect_claimed, merge_extracted, apply_grounding_verdicts, record_claim_violations};
static VISION_KV_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl crate::model::LogisModel {

    /// 🌟 [VISION PIPELINE] 이미지 1장 → 구조화 JSON → DB 저장.
    ///
    ///  ── 5단계 ──
    ///   STEP 1  SigLIP2 패치 임베딩 격자 생성 (NaFlex, 종횡비 보존)
    ///   STEP 2  Doc Type NMS Battle       (그룹 → 코드, 동률일 때만 LLM 1회)
    ///   STEP 3  Column Cosine Matching    (필드 앵커 히트맵)
    ///   STEP 4  Vision NMS & Cropping     (연결성분 → 배타 배정 → 픽셀 박스)
    ///   STEP 5  Qwen 3.5 2B 정제 추출      (카테고리별 정밀 크롭 입력)
    pub async fn extract_from_image(
        &self,
        task_id: String,
        image_path: String,
        language: String,
        search_mode: String,
        app_handle: &tauri::AppHandle,
        cancel_token: Option<Arc<AtomicBool>>,
        store_mutex: &Arc<tokio::sync::Mutex<Option<crate::store::VectorStore>>>,
    ) -> anyhow::Result<()> {
        let app_handle_clone = app_handle.clone();
        let task_id_clone = task_id.clone();
        let emit_term = move |msg: &str| {
            println!("{}", msg);
            use tauri::Emitter;
            let _ = app_handle_clone.emit("task-console-log", serde_json::json!({"task_id": task_id_clone, "text": format!("{}\n", msg)}));
        };

        emit_term("\n=======================================");
        emit_term(&format!("[ENGINE] 🚀 Starting Image Extraction Pipeline for Task: {}", task_id));
        crate::utils::score_dynamics::enter_scope(
            "",
            crate::utils::score_dynamics::Track::Vision,
            "unknown",
            "",
        );
        emit_term("[STAGE-1] Preparing SigLIP2 Vision Encoder + Qwen3.5 (2B)...");

        let payload_load = json!({ "task_id": task_id.clone(), "category": "Loading Model", "summary": "Initializing Vision Core...", "spinner": "⠋" });
        let _ = app_handle.emit("extraction-progress", &payload_load);
        crate::utils::logger::log_task_progress(app_handle, &task_id, &payload_load);

        // 🌟 SigLIP2 로드: 비전만 먼저 로드하여 메모리 피크 최소화
        //    텍스트 인코더는 코드 분류 단계에서 필요하므로 나중에 로드합니다.
        self.check_siglip2_downloaded().await?;
        self.ensure_siglip2_ext(true, false).await?;

        // 🌟 [VRAM SETTLE BEFORE QWEN3.5] SigLIP2 로드 후 실제 여유 메모리 확인
        if !self.is_cpu_mode {
            self.wait_for_vram_settle(1200, 10, cancel_token.clone()).await.ok();
        }

        if let Ok(img) = image::open(&image_path) {
            let dynamic_image = image::DynamicImage::ImageRgb8(img.to_rgb8());

            let mut is_trade_doc = search_mode == "shipping";
            let mut extracted_data = json!({});
            let grid = {
                let mut siglip_guard = self.siglip2_model.lock().await;
                let siglip = siglip_guard.as_mut()
                    .ok_or_else(|| anyhow::anyhow!("SigLIP2 model not loaded"))?;
                crate::models::siglip2::vision_encoder::encode_image_and_release(
                    siglip, &dynamic_image
                ).map_err(|e| anyhow::anyhow!("SigLIP2 encode failed: {}", e))?
            };
            emit_term(&format!(
                "  🧬 [PATCH GRID READY] {}x{} = {} patches (host {:.2}MB) | 비전 반납 완료, 텍스트는 캐시 미스 시에만 로드",
                grid.grid_rows, grid.grid_cols, grid.len(),
                (grid.len() * 1152 * 4) as f64 / 1e6
            ));
            let legibility = crate::models::siglip2::legibility::build_legibility_map(
                &dynamic_image,
                grid.grid_rows,
                grid.grid_cols,
                &emit_term,
            );
            let mut grounding_claims:
                Vec<crate::models::siglip2::value_grounding::GroundingClaim> = Vec::new();
            let mut relay_plan: Vec<(&'static str, crate::parsing::TradeRelayKey)> = Vec::new();
            let mut cached_verdict:
                Option<crate::models::siglip2::vision_encoder::DocTypeVerdict> = None;
            let layout_hit = crate::models::siglip2::layout_cache::LAYOUT_CACHE.probe(
                &grid,
                &legibility,
                !is_trade_doc,
                &emit_term,
            );
            let layout_reuse = layout_hit.as_ref().map_or(false, |h| h.reuse());
            let mut layout_fresh: Option<Vec<crate::models::siglip2::layout_cache::CachedHeatmap>> = None;
            if let Some(h) = layout_hit.as_ref().filter(|h| h.reuse()) {
                cached_verdict = h.verdict();
                if h.route_trade && !is_trade_doc {
                    emit_term(&format!(
                        "  🔀 [MODE REROUTE / LAYOUT CACHE] 신뢰 템플릿 '{}' 이 trading 서식('{}')입니다. 사전 분류 없이 trading 파이프라인으로 전환합니다.",
                        h.template_id, h.detected_type
                    ));
                    is_trade_doc = true;
                }
            }

            if !is_trade_doc && !layout_reuse {
                // 🌟 [LAZY TEXT] 앵커가 전부 캐시에 있으면 텍스트 인코더 없이 판정됩니다.
                match self
                    .with_siglip_text("doc type classification (reroute probe)", |m| {
                        crate::models::siglip2::vision_encoder::classify_doc_type(m, &grid, &emit_term)
                    })
                    .await
                {
                    Ok(v) => {
                        if v.title_confirmed && v.code != "TRACKING" && v.code != "Unknown" {
                            emit_term(&format!(
                                "  🔀 [MODE REROUTE] mode='commerce' 이지만 서식 전문 '{}' 이 인쇄 확인되었습니다. trading 파이프라인으로 전환합니다. (code='{}', margin {:+.4})",
                                v.title_text, v.code, v.code_margin
                            ));
                            is_trade_doc = true;
                        } else {
                            emit_term(&format!(
                                "  🛒 [MODE KEEP] mode='commerce' 유지 (title_confirmed={}, code='{}')",
                                v.title_confirmed, v.code
                            ));
                        }
                        cached_verdict = Some(v);
                    }
                    Err(e) => {
                        emit_term(&format!("  ⚠️ [MODE REROUTE SKIP] 사전 분류 실패로 커머스 경로를 유지합니다: {}", e));
                    }
                }
            }

            if is_trade_doc {
                emit_term(&format!(
                    "  🧬 [PATCH GRID] {}x{} = {} patches | scale({:.3}, {:.3})",
                    grid.grid_rows, grid.grid_cols, grid.len(), grid.scale_x, grid.scale_y
                ));

                emit_term("[STAGE-2] 🚢 Trade Document Mode: SigLIP2 Cosine Classification...");
                let verdict = match cached_verdict.take() {
                    Some(v) => {
                        emit_term(&format!(
                            "  ♻️ [VERDICT REUSE] 리라우트 프로브의 판정을 재사용합니다. (code='{}', group='{}', margin {:+.4}) — 앵커 뱅크 3종 재구축을 생략합니다.",
                            v.code, v.group, v.code_margin
                        ));
                        v
                    }
                    None => self
                        .with_siglip_text("doc type classification (step 2)", |m| {
                            crate::models::siglip2::vision_encoder::classify_doc_type(m, &grid, &emit_term)
                        })
                        .await
                        .map_err(|e| anyhow::anyhow!("SigLIP2 classify failed: {}", e))?,
                };

                let mut detected_type = verdict.code.clone();

                let conflict = !verdict.conflict_candidates.is_empty();
                let mut tie_candidates: Vec<(String, f32)> = if conflict {
                    verdict.conflict_candidates.clone()
                } else if verdict.title_confirmed && !verdict.title_band.is_empty() {
                    verdict.title_band.clone()
                } else {
                    verdict.code_candidates.clone()
                };
                let leads = tie_candidates
                    .first()
                    .map_or(true, |(c, _)| *c == verdict.code);
                if !tie_candidates.iter().any(|(c, _)| *c == verdict.code) {
                    tie_candidates.push((verdict.code.clone(), verdict.code_score));
                }
                let tie_margin: f32 = if verdict.title_confirmed && tie_candidates.len() >= 2 {
                    (tie_candidates[0].1 - tie_candidates[1].1).abs()
                } else {
                    verdict.code_margin
                };
                if !conflict && leads {
                    crate::utils::score_dynamics::record_baseline("vision.tie_margin", tie_margin);
                }
                if let Some(t) = layout_hit.as_ref().filter(|h| h.reuse() && !h.detected_type.is_empty()) {
                    detected_type = t.detected_type.clone();
                    emit_term(&format!(
                        "  🗂️ [LAYOUT CACHE / VERDICT] 신뢰 템플릿 '{}' 의 확정 서식 '{}' 을 그대로 씁니다. 이 서식에서 LLM 재판정이 필요했던 경우에도 그 결과가 템플릿에 들어 있으므로 다시 묻지 않습니다.",
                        t.template_id, t.detected_type
                    ));
                } else if verdict.title_confirmed && !conflict && leads && (tie_candidates.len() <= 1 || tie_margin >= 0.15) {
                    emit_term(&format!(
                        "  🪪 [TITLE VERDICT TRUSTED] 판정 '{}' 은 상단 제목 행에 인쇄된 서식 전문의 선두이고 바디 축과 충돌하지 않습니다. 제목 행 밴드 안 1·2위 마진 {:+.4}. 밴드 밖 코드는 이미 후보가 아니므로 재판정 여부는 밴드 안 마진으로 정합니다. LLM 재판정을 열지 않습니다.",
                        verdict.code, tie_margin
                    ));
                } else if tie_candidates.len() > 1 && (conflict || !leads || tie_margin < 0.15) {
                    if conflict || !leads {
                        emit_term(&format!(
                            "  ⚔️ [TIE BREAK / TITLE-BODY CONFLICT] 판정 '{}' 이 제목 행 밴드의 선두가 아니거나, 바디 축이 결정 기준을 넘어 다른 서식을 가리킵니다. 제목 행 밴드와 두 축의 1위를 합친 후보 {}개로 LLM 재판정 1회 수행: {:?}",
                            verdict.code,
                            tie_candidates.len(),
                            tie_candidates.iter().map(|(c, _)| c.as_str()).collect::<Vec<_>>()
                        ));
                    } else {
                        emit_term(&format!(
                            "  🤝 [TIE BREAK] 마진 {:+.4} 가 임계 미만. LLM 재판정 1회 수행 (후보 {}개{}).",
                            tie_margin,
                            tie_candidates.len(),
                            if verdict.title_confirmed { " — 제목 행에서 동점인 전문만, 밴드 안 마진 기준" } else { "" }
                        ));
                    }
                    let prompt = crate::parsing::get_trade_doc_classification_prompt_with_evidence(
                        &verdict.group,
                        &tie_candidates,
                    );
                    let type_res = self.chat_with_qwen3_5_image_spinner(
                        "You are a document classifier.", &prompt, Some(dynamic_image.clone()), app_handle, "extraction-progress",
                        json!({ "category": "Vision (Step 2)", "summary": "Verifying document type..." }), 64, cancel_token.clone(), Some(task_id.clone()), None
                    ).await?;
                    if let Some(v) = crate::parsing::parse_json_from_llm(&type_res).get("doc_type").and_then(|d| d.as_str()) {
                        if tie_candidates.iter().any(|(c, _)| c == v) {
                            emit_term(&format!("  ✅ [TIE BREAK] LLM 판정 '{}' 채택.", v));
                            detected_type = v.to_string();
                        } else {
                            emit_term(&format!(
                                "  🚫 [TIE BREAK] LLM 이 후보 밖 '{}' 반환. 비전 판정 '{}' 유지.",
                                v, detected_type
                            ));
                        }
                    }
                    crate::utils::score_dynamics::record_baseline(
                        "vision.tie_break_changed",
                        if detected_type == verdict.code { 0.0 } else { 1.0 },
                    );
                }

                emit_term(&format!("✅ Document identified as: **{}** (group: {})", detected_type, verdict.group));
                {
                    let t_orphans = crate::utils::ai_utils::trade_title_ml_orphans();
                    let g_orphans = crate::utils::ai_utils::trade_group_ml_orphans();
                    if !t_orphans.is_empty() || !g_orphans.is_empty() {
                        emit_term(&format!(
                            "  ⚠️ [ML TABLE ORPHANS] 12개 언어 전문표에서 TRADE_DOC_TITLES 의 영문 전문과 맞지 않는 키 {}개 {:?} · TRADE_GROUP_CODES 에 없는 그룹 {}개 {:?} — 이 항목의 다국어 구는 어느 코드·그룹에도 붙지 않았습니다. 표의 키를 TRADE_DOC_TITLES / TRADE_GROUP_CODES 의 표기와 같게 고치면 사라집니다.",
                            t_orphans.len(), t_orphans, g_orphans.len(), g_orphans
                        ));
                    }
                }
                crate::utils::score_dynamics::refine_primary(&detected_type);
                if detected_type == "TRACKING" {
                    emit_term("[STAGE-2] 📦 Fast-Tracking Parcel Label...");
                    self.release_siglip2("TRACKING fast-track, before Qwen3.5 load").await;
                    let prompt = crate::parsing::get_image_extraction_prompt("kr", &language, "tracking", "");
                    let (_track_bias, track_prej) = crate::parsing::get_vision_tracking_bias(&language);
                    let result_str = self.chat_with_qwen3_5_image_spinner(
                        "You are a highly precise logistics data extraction assistant.", &prompt, Some(dynamic_image.clone()), app_handle, "extraction-progress",
                        json!({ "category": "Vision Analysis", "summary": "Extracting Tracking Label data..." }), 512, cancel_token.clone(), Some(task_id.clone()), Some(&track_prej)
                    ).await?;

                    extracted_data = crate::parsing::parse_json_from_llm(&result_str);
                    record_grounding_claims(
                        &mut grounding_claims,
                        "tracking",
                        &extracted_data,
                        (0, 0, grid.orig_width, grid.orig_height),
                    );

                    if let Some(obj) = extracted_data.as_object_mut() {
                        obj.insert("doc_type".to_string(), json!("TRACKING"));
                    }
                } else {
                    // 🌟 [STEP 3~5] 히트맵 → 크롭 → Qwen3.5 추출 파이프라인
                    emit_term("[STAGE-3] 🔥 Column Cosine Matching (Heatmap)...");

                    let title_prej: Vec<String> = if verdict.title_text.is_empty() {
                        Vec::new()
                    } else {
                        vec![verdict.title_text.clone()]
                    };
                    let reused_heatmaps = layout_hit
                        .as_ref()
                        .filter(|h| h.reuse())
                        .and_then(|h| h.heatmaps_for(&detected_type));
                    let mut heatmaps = match reused_heatmaps {
                        Some(hm) => {
                            emit_term(&format!(
                                "  🗂️ [LAYOUT CACHE / HEATMAP REUSE] 신뢰 템플릿의 열 히트맵 {}개를 그대로 씁니다. 제목 행·여백 억제, 아레나, 크롭 계획은 이 문서의 판독 지도로 다시 계산하므로 표 길이나 빈 칸이 달라도 이 문서 기준으로 잘립니다.",
                                hm.len()
                            ));
                            hm
                        }
                        None => {
                            let hm = self
                                .with_siglip_text("column heatmaps (trade)", |m| {
                                    crate::models::siglip2::vision_encoder::build_column_heatmaps(
                                        m, &grid, &detected_type, &language, Some(&legibility), &title_prej, &emit_term
                                    )
                                })
                                .await
                                .map_err(|e| anyhow::anyhow!("Heatmap build failed: {}", e))?;
                            let snap = crate::models::siglip2::layout_cache::snapshot(&hm);
                            if let Some(h) = layout_hit.as_ref() {
                                crate::models::siglip2::layout_cache::LAYOUT_CACHE.observe(h, &detected_type, &snap, &emit_term);
                            }
                            layout_fresh = Some(snap);
                            hm
                        }
                    };
                    {
                        let title_row_max = (grid.grid_rows / 18).max(1).min(grid.grid_rows.saturating_sub(1));
                        let mut suppressed = 0usize;
                        for hm in heatmaps.iter_mut() {
                            for r in 0..=title_row_max.min(grid.grid_rows.saturating_sub(1)) {
                                for c in 0..grid.grid_cols {
                                    let i = r * grid.grid_cols + c;
                                    if i < hm.scores.len() && hm.scores[i] > f32::MIN {
                                        hm.scores[i] = f32::MIN;
                                        suppressed += 1;
                                    }
                                }
                            }
                        }
                                                emit_term(&format!(
                            "  🚫 [TITLE ROW SUPPRESSION] 상단 {}행(제목 인쇄 행만) 점수 {}개 억제 → r2 라벨 행 생존, header 봉우리가 값 행(r2~r4)에서 결정됩니다.",
                            title_row_max + 1, suppressed
                        ));
                    }

                    {
                        let cols = grid.grid_cols.max(1);
                        let rows = grid.grid_rows.max(1);
                        let cw = grid.orig_width as f32 / cols as f32;
                        let ch = grid.orig_height as f32 / rows as f32;
                        let blank: Vec<bool> = (0..rows * cols)
                            .map(|i| {
                                let r = i / cols;
                                let c = i % cols;
                                let bx = (
                                    (c as f32 * cw).floor() as u32,
                                    (r as f32 * ch).floor() as u32,
                                    ((((c + 1) as f32) * cw).ceil() as u32).min(grid.orig_width),
                                    ((((r + 1) as f32) * ch).ceil() as u32).min(grid.orig_height),
                                );
                                let (lg, il, bl) =
                                    legibility.count_in_bbox(bx, grid.orig_width, grid.orig_height);
                                lg == 0 && il == 0 && bl > 0
                            })
                            .collect();
                        let blank_cnt = blank.iter().filter(|b| **b).count();
                        let mut cut = 0usize;
                        let mut shrunk: Vec<String> = Vec::new();
                        let mut protected: Vec<String> = Vec::new();
                        for hm in heatmaps.iter_mut() {
                            let before = hm.scores.iter().filter(|s| **s > 0.0).count();
                            if before == 0 { continue; }
                            let after = hm
                                .scores
                                .iter()
                                .enumerate()
                                .filter(|(i, s)| {
                                    **s > 0.0 && !blank.get(*i).copied().unwrap_or(false)
                                })
                                .count();
                            if after == 0 {
                                protected.push(hm.category.clone());
                                continue;
                            }
                            let m = hm.scores.len().min(blank.len());
                            for i in 0..m {
                                if blank[i] && hm.scores[i] > f32::MIN {
                                    hm.scores[i] = f32::MIN;
                                    cut += 1;
                                }
                            }
                            shrunk.push(format!("{}({}→{})", hm.category, before, after));
                        }
                        emit_term(&format!(
                            "  🫥 [BLANK CELL SUPPRESSION] 여백 칸 {}/{} 에서 점수 {}개를 내려놓았습니다. 활성 패치 변화: {} — 여백에서 카테고리끼리 상대 비교를 하면 전부 낮은 점수 중 잡음이 큰 쪽이 그 칸을 가져가고, 그 영토가 밴드 확장과 구제 지분을 왜곡합니다.",
                            blank_cnt, rows * cols, cut,
                            if shrunk.is_empty() { "-".to_string() } else { shrunk.join(" | ") }
                        ));
                        if !protected.is_empty() {
                            emit_term(&format!(
                                "  🛡️ [BLANK SUPPRESSION PROTECT] 여백을 걷어내면 활성 패치가 0개가 되는 카테고리 {:?} 는 원본을 유지합니다. 그 축의 봉우리가 전부 여백에 찍혔다는 뜻이며, 여기서 히트맵을 없애면 크롭 자체가 불가능해집니다.",
                                protected
                            ));
                        }
                        crate::utils::score_dynamics::record_baseline(
                            "vision.blank_suppressed",
                            cut as f32 / (rows * cols).max(1) as f32,
                        );
                    }

                    // ── STEP 3.5 : NMS Arena ──
                    {
                        let mut protect: Vec<&str> =
                            crate::logic::TRADE_ARRAY_CATEGORIES.to_vec();
                        protect.push(crate::logic::TRADE_IDENTITY_CATEGORY);
                        let arena = crate::models::siglip2::nms_arena::run_arena(
                            &heatmaps, &grid, &legibility, &protect, &emit_term,
                        );
                        crate::utils::score_dynamics::record_baseline(
                            "vision.arena_rounds",
                            arena.rounds as f32,
                        );
                        crate::utils::score_dynamics::record_baseline(
                            "vision.arena_margin_gate",
                            arena.margin_gate,
                        );
                        for t in arena.territories.iter() {
                            crate::utils::score_dynamics::record_baseline(
                                &format!("vision.territory.{}", t.category),
                                t.patches.len() as f32
                                    / (grid.grid_rows * grid.grid_cols).max(1) as f32,
                            );
                        }
                        crate::models::siglip2::nms_arena::apply_arena(
                            &mut heatmaps, &arena, &emit_term,
                        );
                    }

                    // ── STEP 4 : Vision NMS & Cropping ──
                    emit_term("[STAGE-4] ✂️ Vision NMS & Cropping...");
                    let height_baseline =
                        crate::models::siglip2::vision_crop::measure_doc_text_height(
                            &dynamic_image, &emit_term,
                        );
                    let mut plans = crate::models::siglip2::vision_crop::plan_crops(
                        &heatmaps,
                        &grid,
                        &legibility,
                        crate::logic::TRADE_ARRAY_CATEGORIES,
                        crate::logic::TRADE_IDENTITY_CATEGORY,
                        crate::logic::TRADE_IDENTITY_FIELD,
                        &emit_term,
                    );
                    emit_term(&format!("  🧾 [PLAN DONE] 크롭 계획 {}건 확정. release_siglip2 진입 전...", plans.len()));
                    if plans.is_empty() {
                        let cats = crate::parsing::get_trade_doc_categories(&detected_type);
                        emit_term(&format!(
                            "  🛟 [FALLBACK] 크롭 영역 미확정. 전체 페이지를 {}개 카테고리에 넘깁니다.",
                            cats.len()
                        ));
                        plans = crate::models::siglip2::vision_crop::whole_page_fallback(&cats, &grid);
                    }

                    // ── STEP 5 : Qwen 3.5 2B 정제 추출 ──
                    // 🌟 [VRAM STAGE] STEP 1~4 완료. SigLIP2(비전 820MB + 텍스트 1.4GB) 전량 반환.
                    //    이 해제가 없으면 ensure_qwen3_5 의 `SigLIP2 is resident` 가드가 발동해
                    //    deep purge 가 통째로 생략되고, 첫 크롭 시 free VRAM 이 147MB 까지 떨어집니다.
                    //    pooled 벡터는 STEP 1 의 grid.pooled 를 재사용하므로 여기서 내려도 안전합니다.
                    self.release_siglip2("STEP 1~4 complete, before Qwen3.5 crop OCR").await;

                    emit_term(&format!("[STAGE-5] 🤖 크롭 {}개 정제 추출", plans.len()));

                    let mut final_data_map = serde_json::Map::new();
                    for c in crate::logic::TRADE_EXTRACTION_CATEGORIES.iter() {
                        if crate::logic::is_trade_array_category(c) {
                            final_data_map.insert(c.to_string(), json!([]));
                        } else {
                            final_data_map.insert(c.to_string(), json!({}));
                        }
                    }
                    emit_term(&format!(
                        "  🗂️ [CATEGORY SLOTS] logic::TRADE_EXTRACTION_CATEGORIES 기준 {}개 슬롯을 만듭니다 (배열 {}개). 배열 카테고리를 객체로 미리 만들어 두면 병합이 그 자리에 배열을 넣지 못해, 크롭마다 원소 하나씩 쌓여야 할 값이 서로를 덮습니다.",
                        crate::logic::TRADE_EXTRACTION_CATEGORIES.len(),
                        crate::logic::TRADE_EXTRACTION_CATEGORIES.iter()
                            .filter(|c| crate::logic::is_trade_array_category(c)).count()
                    ));
                    final_data_map.insert("header".to_string(), json!({"doc_type": detected_type}));
                    // 🌟 [ARRAY KEY UNIFY] 초기화 키를 카테고리명과 일치시킵니다.
                    //
                    //  ── 실측 사고 ──
                    //   merge_extracted 는 `merged.entry(category)` 로 배열을 넣으므로
                    //   items 카테고리의 결과는 "items" 키에 쌓입니다.
                    //   그런데 여기서 "line_items" 를 만들어 두어 두 키가 공존했고,
                    //   저장 결과가 items 3행 / line_items 빈 배열로 갈렸습니다.
                    //   STEP C 의 FLATTEN 도 "line_items" 를 훑기 때문에
                    //   hs_code 루트 승격이 한 번도 성립하지 않았습니다.
                    //   containers 는 카테고리명과 키가 우연히 같아 정상 동작했습니다.
                    //
                    //  ── 하위 호환 ──
                    //   generate_rich_summary 등 기존 소비처가 line_items 를 읽으므로
                    //   저장 직전 STEP C 에서 items → line_items 로 미러합니다.
                    final_data_map.insert("items".to_string(), json!([]));
                    final_data_map.insert("containers".to_string(), json!([]));

                    let self_ref_field: String = crate::logic::trade_reference_field_of(&detected_type)
                        .unwrap_or("")
                        .to_string();
                    let schema_fields: Vec<String> = crate::parsing::get_detail_schema_fields(&detected_type, "", &language)
                        .into_iter()
                        .map(|(f, _, _, _)| f)
                        .filter(|f| !crate::utils::bias_schema::is_system_axis(f))
                        .filter(|f| self_ref_field.is_empty() || *f != self_ref_field)
                        .collect();
                    if !self_ref_field.is_empty() {
                        emit_term(&format!(
                            "  🧹 [SELF-REFERENCE FIELD DROP] '{}' 는 '{}' 서식이 자기 자신을 가리키는 참조 축이라 이 문서에 존재할 수 없습니다. 라벨 뱅크와 식별 패스에서 제외해, 자기 문서번호가 doc_number 대신 이 축으로 라우팅되는 경로를 막습니다.",
                            self_ref_field, detected_type
                        ));
                    }
                    let gate_banks: Vec<(String, Vec<Vec<f32>>, Vec<f32>)> = {
                        let mut phr_all: Vec<String> = Vec::new();
                        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                        let mut per_field: Vec<(String, Vec<String>, Vec<f32>)> = Vec::new();
                        let sup_fields = crate::logic::trade_label_supplement_fields();
                        let mut sup_hit = 0usize;
                        let mut thin = 0usize;
                        for f in schema_fields.iter() {
                            let (ph, wt) = crate::model::merge::owner_label_bank(&language, "shipping_doc", f);
                            if sup_fields.iter().any(|s| *s == f.as_str()) { sup_hit += 1; }
                            if ph.len() <= 2 { thin += 1; }
                            for p in ph.iter() {
                                if seen.insert(p.clone()) { phr_all.push(p.clone()); }
                            }
                            per_field.push((f.clone(), ph, wt));
                        }
                        let mut embs: Vec<Vec<f32>> = Vec::with_capacity(phr_all.len());
                        for part in phr_all.chunks(200) {
                            let e = self
                                .get_embedding_batch(part.to_vec())
                                .await
                                .unwrap_or_else(|_| vec![Vec::new(); part.len()]);
                            embs.extend(e);
                        }
                        let table: std::collections::HashMap<String, Vec<f32>> =
                            phr_all.into_iter().zip(embs.into_iter()).collect();
                        let mut banks: Vec<(String, Vec<Vec<f32>>, Vec<f32>)> = Vec::new();
                        for (f, ph, wt) in per_field.into_iter() {
                            let mut b: Vec<Vec<f32>> = Vec::new();
                            let mut w: Vec<f32> = Vec::new();
                            for (p, x) in ph.iter().zip(wt.iter()) {
                                if let Some(e) = table.get(p) {
                                    if e.is_empty() { continue; }
                                    b.push(e.clone());
                                    w.push(*x);
                                }
                            }
                            banks.push((f, b, w));
                        }
                        emit_term(&format!(
                            "  📖 [LABEL BANK] 스키마 필드 {}개의 라벨 뱅크를 크롭 루프 진입 전에 한 번만 세웁니다 (고유 구 {}개 · 보강표 적용 {}축 · 구 2개 이하 {}축). 스칼라 크롭은 스키마 프롬프트 대신 인쇄된 라벨↔값 쌍을 옮겨 적고, 읽힌 라벨을 이 뱅크로 스키마 전체에 라우팅합니다. 뱅크 정의는 merge.rs 의 owner_label_bank 하나뿐이므로 이 뱅크와 소유권 경쟁·복구 게이트의 뱅크가 어긋날 수 없습니다. '구 2개 이하' 축이 남아 있으면 그 축은 다국어가 아니라 구 자체가 없는 것이므로 bias.json 또는 보강표에 넣어야 합니다.",
                            banks.len(), table.len(), sup_hit, thin
                        ));
                        banks
                    };
                    let label_exact: crate::model::merge::LabelExactIndex = {
                        let per_field: Vec<(String, Vec<String>)> = schema_fields
                            .iter()
                            .map(|f| {
                                (
                                    f.clone(),
                                    crate::model::merge::owner_label_bank(&language, "shipping_doc", f).0,
                                )
                            })
                            .collect();
                        let idx = crate::model::merge::build_label_exact_index(&per_field);
                        let unique = idx.values().filter(|v| v.len() == 1).count();
                        crate::utils::score_dynamics::record_baseline(
                            "vision.label_exact_shared_ratio",
                            if idx.is_empty() { 0.0 } else { (idx.len() - unique) as f32 / idx.len() as f32 },
                        );
                        emit_term(&format!(
                            "  📖 [LABEL EXACT INDEX] 라벨 뱅크의 구를 대소문자·기호를 뺀 키로 묶었습니다. 한 축에만 속하는 키 {}개는 인쇄 라벨과 글자가 같으면 완전일치 근거가 되고, 여러 축이 함께 쓰는 키 {}개는 그 키 하나만으로는 축을 정하지 않습니다('CONSIGNEE VAT/EORI' 처럼 '/' 로 묶인 라벨은 뱅크 구와 같은 분리 규칙으로 나눠, 모든 부분이 가리키는 축이 하나로 좁혀질 때만 씁니다). 라벨 뱅크와 같은 owner_label_bank 에서 뽑으므로 두 근거가 서로 다른 어휘를 보지 않습니다.",
                            unique,
                            idx.len() - unique
                        ));
                        idx
                    };
                    let mut pair_evidence: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
                    let mut pair_label: std::collections::HashMap<String, String> = std::collections::HashMap::new();
                    let mut identity_locked: std::collections::HashSet<String> = std::collections::HashSet::new();
                    let mut read_legible: Vec<usize> = Vec::new();
                    let mut deferred_pairs: Vec<(String, String, Vec<f32>, (u32, u32, u32, u32))> = Vec::new();
                    let mut array_deferred: Vec<(String, String, String, String, f32, (u32, u32, u32, u32))> = Vec::new();
                    let peak_cols = grid.grid_cols.max(1);
                    let peak_cw = grid.orig_width as f32 / peak_cols as f32;
                    let peak_ch = grid.orig_height as f32 / grid.grid_rows.max(1) as f32;
                    let field_peak_inside = |field: &str, bbox: (u32, u32, u32, u32)| -> bool {
                        heatmaps.iter().any(|hm| {
                            hm.field_peaks.iter().any(|(f, patch, _)| {
                                if f.as_str() != field { return false; }
                                let px = ((patch % peak_cols) as f32 + 0.5) * peak_cw;
                                let py = ((patch / peak_cols) as f32 + 0.5) * peak_ch;
                                px >= bbox.0 as f32 && px <= bbox.2 as f32 && py >= bbox.1 as f32 && py <= bbox.3 as f32
                            })
                        })
                    };
                    let legible_set_in = |bbox: (u32, u32, u32, u32)| -> Vec<usize> {
                        (0..grid.grid_rows * grid.grid_cols)
                            .filter(|&i| {
                                if !legibility.is_legible(i) { return false; }
                                let px = ((i % peak_cols) as f32 + 0.5) * peak_cw;
                                let py = ((i / peak_cols) as f32 + 0.5) * peak_ch;
                                px >= bbox.0 as f32 && px <= bbox.2 as f32 && py >= bbox.1 as f32 && py <= bbox.3 as f32
                            })
                            .collect()
                    };

                    // 🌟 grounding_claims 는 바깥 스코프에 선언되어 있습니다. (STEP 6 이 소비)

                    for (idx, plan) in plans.iter().enumerate() {
                        if cancel_token
                            .as_ref()
                            .map_or(false, |t| t.load(std::sync::atomic::Ordering::Relaxed))
                        {
                            emit_term("🛑 Task cancelled by user. Terminating safely.");
                            return Ok(());
                        }

                        let (lg_cnt, _il_cnt, _bl_cnt) =
                            legibility.count_in_bbox(plan.bbox, grid.orig_width, grid.orig_height);
                        crate::utils::score_dynamics::record_baseline(
                            "vision.crop_legible_patches",
                            lg_cnt as f32,
                        );
                        if lg_cnt == 0 {
                            emit_term(&format!(
                                "    🚫 [EMPTY CROP SKIP] '{}' 는 판독 가능 패치가 0개입니다. Qwen 호출을 생략합니다.",
                                plan.category
                            ));
                            crate::utils::score_dynamics::record_baseline("vision.empty_crop_skip", 1.0);
                            continue;
                        }
                        crate::utils::score_dynamics::record_baseline("vision.empty_crop_skip", 0.0);

                        if !plan.twin_of.is_empty()
                            && crate::logic::TRADE_ARRAY_CATEGORIES
                                .iter()
                                .any(|c| *c == plan.category.as_str())
                        {
                            emit_term(&format!(
                                "    👯 [TWIN ARRAY SKIP] '{}' 는 '{}' 와 좌표가 같은 쌍둥이 크롭입니다. 같은 표에서 배열을 두 번 만들면 행이 그대로 복제되므로 이 크롭은 건너뜁니다.",
                                plan.category, plan.twin_of
                            ));
                            continue;
                        }

                        if plan.owned_patches == 0 {
                            emit_term(&format!(
                                "    🧭 [TERRITORY TAG] '{}' 크롭 안에 자기 영토 패치가 한 칸도 없습니다. 이 크롭에서는 명시된 라벨↔값만 읽고 줄 전체를 값으로 승격하지 않아야 합니다.",
                                plan.category
                            ));
                        }

                        let array_plan = crate::logic::TRADE_ARRAY_CATEGORIES
                            .iter()
                            .any(|c| *c == plan.category.as_str());
                        let array_as_pairs = array_plan && {
                            let ident = crate::model::merge::row_identity_fields(&plan.category);
                            !ident.is_empty()
                                && ident.iter().all(|k| {
                                    let k: &str = k;
                                    let anywhere = heatmaps.iter().any(|hm| {
                                        hm.field_peaks.iter().any(|(f, _, _)| f.as_str() == k)
                                    });
                                    anywhere && !field_peak_inside(k, plan.bbox)
                                })
                        };
                        if array_plan {
                            crate::utils::score_dynamics::record_baseline(
                                "vision.array_crop_as_pairs",
                                if array_as_pairs { 1.0 } else { 0.0 },
                            );
                        }
                        if array_as_pairs {
                            emit_term(&format!(
                                "    🧾 [ARRAY CROP → PAIRS] '{}' 크롭 px({},{})-({},{}) 안에 행 정체 축 {:?} 의 라벨 봉우리가 하나도 없습니다 (봉우리는 모두 크롭 밖 패치에 있습니다). 이 크롭에서 표 행을 만들면 병합 단계가 '정체 없는 행' 으로 전부 폐기하므로, 표 스키마 대신 라벨↔값 쌍으로 읽어 스키마 전체로 라우팅합니다. 표 아래에 붙은 총계 박스의 스칼라 라벨(총중량·인코텀즈 등)이 여기서 회수됩니다.",
                                plan.category, plan.bbox.0, plan.bbox.1, plan.bbox.2, plan.bbox.3,
                                crate::model::merge::row_identity_fields(&plan.category)
                            ));
                        }
                        let row_tile_cat = array_plan && !array_as_pairs;
                        let mut pending_read: Vec<usize> = Vec::new();
                        if !row_tile_cat && plan.category != crate::logic::TRADE_IDENTITY_CATEGORY {
                            let mine = legible_set_in(plan.bbox);
                            let region_read = !array_plan && !mine.is_empty() && mine.iter().all(|i| read_legible.contains(i));
                            let unread_axes: Vec<String> = if region_read {
                                schema_fields
                                    .iter()
                                    .filter(|f| crate::logic::trade_field_category(f) == plan.category.as_str())
                                    .filter(|f| !pair_evidence.contains_key(f.as_str()))
                                    .filter(|f| {
                                        final_data_map
                                            .get(f.as_str())
                                            .map_or(true, |v| v.is_null() || v.as_str().map_or(false, |s| s.trim().is_empty()))
                                    })
                                    .filter(|f| field_peak_inside(f, plan.bbox))
                                    .cloned()
                                    .collect()
                            } else {
                                Vec::new()
                            };
                            let centre_of = |patch: usize| -> (f32, f32) {
                                (((patch % peak_cols) as f32 + 0.5) * peak_cw, ((patch / peak_cols) as f32 + 0.5) * peak_ch)
                            };
                            let in_box = |pt: (f32, f32), b: (u32, u32, u32, u32)| -> bool {
                                pt.0 >= b.0 as f32 && pt.0 <= b.2 as f32 && pt.1 >= b.1 as f32 && pt.1 <= b.3 as f32
                            };
                            let has_value = |f: &str| -> bool {
                                let cell = |v: &Value| !(v.is_null() || v.as_str().map_or(false, |s| s.trim().is_empty()));
                                if final_data_map.get(f).map_or(false, |v| !v.is_array() && cell(v)) {
                                    return true;
                                }
                                let cat = crate::logic::trade_field_category(f);
                                crate::logic::is_trade_array_category(cat)
                                    && final_data_map
                                        .get(cat)
                                        .and_then(|v| v.as_array())
                                        .map_or(false, |rows| rows.iter().any(|r| r.get(f).map_or(false, |v| cell(v))))
                            };
                            let (reread_axes, dead_axes): (Vec<String>, Vec<String>) = unread_axes.iter().cloned().partition(|f| {
                                heatmaps.iter().any(|hm| {
                                    hm.field_peaks.iter().any(|(pf, patch, _)| {
                                        pf == f
                                            && in_box(centre_of(*patch), plan.bbox)
                                            && legibility.verdict.get(*patch).copied()
                                                == Some(crate::models::siglip2::legibility::PatchLegibility::Legible)
                                            && !heatmaps.iter().any(|h2| {
                                                h2.field_peaks.iter().any(|(of, op, _)| {
                                                    op == patch
                                                        && of != f
                                                        && has_value(of)
                                                        && grounding_claims.iter().any(|g| g.field == *of && in_box(centre_of(*op), g.bbox))
                                                })
                                            })
                                    })
                                })
                            });
                            if region_read && reread_axes.is_empty() {
                                if dead_axes.is_empty() {
                                    emit_term(&format!(
                                        "    ♻️ [REGION ALREADY READ] '{}' 크롭 px({},{})-({},{}) 의 판독 가능 패치 {}칸이 앞선 스칼라 크롭들이 이미 쌍으로 읽은 지면 안에 전부 들어 있고, 이 범주의 축 가운데 라벨 봉우리가 이 크롭 안에 있으면서 비어 있는 축도 없습니다. 이 크롭의 호출을 건너뜁니다.",
                                        plan.category, plan.bbox.0, plan.bbox.1, plan.bbox.2, plan.bbox.3, mine.len()
                                    ));
                                } else {
                                    emit_term(&format!(
                                        "    ♻️ [REGION ALREADY READ / DEAD PEAKS] '{}' 크롭 px({},{})-({},{}) 의 지면은 앞선 크롭이 읽었고, 비어 있는 축 {:?} 는 라벨 봉우리가 이 크롭 안에 있지만 그 칸이 판독 불가이거나 이미 채워진 다른 축의 출처 칸입니다. 같은 지면을 이 범주 정의로 다시 읽어도 이미 확정된 쌍만 되돌아오므로 호출을 건너뜁니다. 이 축들은 FIELD RECOVERY 가 같은 규칙(판독 가능한 봉우리 · 출처 칸 가지치기)으로 다시 판정합니다.",
                                        plan.category, plan.bbox.0, plan.bbox.1, plan.bbox.2, plan.bbox.3, dead_axes
                                    ));
                                    crate::utils::score_dynamics::record_baseline("vision.region_reread_dead", 1.0);
                                }
                                crate::utils::score_dynamics::record_baseline("vision.region_already_read", 1.0);
                                continue;
                            }
                            if region_read {
                                crate::utils::score_dynamics::record_baseline("vision.region_reread", 1.0);
                                emit_term(&format!(
                                    "    🔁 [REGION RE-READ] '{}' 크롭 px({},{})-({},{}) 의 지면은 앞선 크롭이 읽었지만, 이 범주의 축 {:?} 는 판독 가능한 라벨 봉우리가 이 크롭 안에 있고 그 칸이 다른 축의 출처도 아닌데 아직 어떤 쌍도 이 축으로 라우팅되지 않았습니다. 앞선 크롭은 다른 범주의 정의로 물었으므로 이름·주소가 여러 줄로 쌓인 당사자 상자 같은 블록은 쌍으로 옮겨지지 않았을 수 있습니다. 같은 지면이라도 이 범주 정의로 다시 읽습니다.{}",
                                    plan.category, plan.bbox.0, plan.bbox.1, plan.bbox.2, plan.bbox.3, reread_axes,
                                    if dead_axes.is_empty() { String::new() } else { format!(" (봉우리가 판독 불가이거나 출처 칸이라 이번 재판독에서 뺀 축: {:?})", dead_axes) }
                                ));
                            } else {
                                crate::utils::score_dynamics::record_baseline("vision.region_already_read", 0.0);
                            }
                            pending_read = mine;
                        }
                        let tile_cats: &[&str] = if array_as_pairs {
                            &[]
                        } else {
                            crate::logic::TRADE_ARRAY_CATEGORIES
                        };
                        let (tile_count, _why) = crate::models::siglip2::vision_crop::decide_tile_count(
                            plan,
                            &heatmaps,
                            &grid,
                            &legibility,
                            tile_cats,
                            &emit_term,
                        );
                        let table_evidence = if row_tile_cat {
                            Some(crate::models::siglip2::vision_crop::table_row_evidence(&dynamic_image, plan.bbox))
                        } else {
                            None
                        };
                        let tiles = match table_evidence {
                            Some((table_rows, bands, min_cols, is_table)) if tile_count > 1 && !is_table => {
                                emit_term(&format!(
                                    "    🧾 [NON-TABLE ARRAY REGION] '{}' 크롭 px({},{})-({},{}) 안의 잉크 행 밴드 {}개 중 열 뭉치 {}개 이상인 행이 {}개뿐입니다. 표는 헤더 행과 데이터 행이 같은 열 구조를 공유해야 성립하므로 이 영역은 표가 아닙니다. 행 타일로 나누지 않고 한 번만 읽습니다 — 총계 박스나 서명 행처럼 격자만 있는 영역을 행 타일로 쪼개면 타일마다 기대 어휘가 복사되어 정체성 없는 행이 생성됩니다.",
                                    plan.category, plan.bbox.0, plan.bbox.1, plan.bbox.2, plan.bbox.3,
                                    bands, min_cols, table_rows
                                ));
                                crate::utils::score_dynamics::record_baseline("vision.non_table_array", 1.0);
                                crate::models::siglip2::vision_crop::plan_overlap_tiles(plan.bbox, 1, 0.25)
                            }
                            Some(_) if tile_count > 1 => {
                                crate::utils::score_dynamics::record_baseline("vision.non_table_array", 0.0);
                                crate::models::siglip2::vision_crop::plan_row_tiles(
                                    &dynamic_image, plan.bbox, &emit_term
                                )
                                .unwrap_or_else(|| {
                                    crate::models::siglip2::vision_crop::plan_overlap_tiles(
                                        plan.bbox, tile_count, 0.25
                                    )
                                })
                            }
                            _ => crate::models::siglip2::vision_crop::plan_overlap_tiles(
                                plan.bbox, tile_count, 0.25
                            ),
                        };
                        crate::utils::score_dynamics::record_baseline(
                            "vision.tile_count", tiles.len() as f32
                        );
                        for tile in tiles.iter() {
                            // 타일 bbox 로 임시 CropPlan 을 만들어 기존 crop_region 을 재사용합니다.
                            let crop = crate::models::siglip2::vision_crop::crop_tile(
                                &dynamic_image, plan, tile, 512
                            );

                            let tile_tag = if tile.total > 1 {
                                format!(" | 타일 {}/{}", tile.index + 1, tile.total)
                            } else {
                                String::new()
                            };
                            emit_term(&format!(
                                "    📤 [{}] {}x{} 크롭 전송 ({}/{}){}",
                                plan.category, crop.width(), crop.height(),
                                idx + 1, plans.len(), tile_tag
                            ));

                            // 🌟 [ALREADY CLAIMED] 앞선 크롭·타일이 확정한 값을 금지 목록으로 전달합니다.
                            //    겹침 타일에서 같은 값이 두 번 나오는 것은 정상이므로
                            //    배열 카테고리는 이 목록을 넘기지 않습니다.
                            //    (넘기면 두 번째 타일이 정당한 반복 행을 스스로 버립니다)
                            let is_array_cat = !array_as_pairs
                                && crate::logic::TRADE_ARRAY_CATEGORIES
                                    .iter()
                                    .any(|c| *c == plan.category.as_str());
                            let claimed = if is_array_cat {
                                Vec::new()
                            } else {
                                collect_claimed(&final_data_map)
                            };
                            if !claimed.is_empty() {
                                emit_term(&format!(
                                    "    🔒 [ALREADY CLAIMED] 확정값 {}건을 금지 목록으로 전달합니다.",
                                    claimed.len()
                                ));
                            }

                            let identity_pass: Option<(std::collections::HashSet<String>, std::collections::HashSet<String>)> =
                                if plan.category == crate::logic::TRADE_IDENTITY_CATEGORY {
                                    let band_bottom = crate::models::siglip2::vision_crop::identity_band_bottom_px(&grid);
                                    if plan.bbox.1 >= band_bottom {
                                        emit_term(&format!(
                                            "    🪪 [IDENTITY PASS EXEMPT] header 크롭 px({},{})-({},{}) 는 식별 밴드 하한 y{} 아래에 있습니다. 문서번호·참조 축은 식별 밴드에서만 인쇄되므로 식별 축을 묻는 스키마 패스를 열지 않고 라벨↔값 쌍 읽기만 수행합니다.",
                                            plan.bbox.0, plan.bbox.1, plan.bbox.2, plan.bbox.3, band_bottom
                                        ));
                                        crate::utils::score_dynamics::record_baseline("vision.identity_pass_exempt", 1.0);
                                        None
                                    } else {
                                        crate::utils::score_dynamics::record_baseline("vision.identity_pass_exempt", 0.0);
                                        let all: Vec<String> =
                                            crate::parsing::get_detail_schema_fields(&detected_type, "", &language)
                                                .into_iter()
                                                .map(|(f, _, _, _)| f)
                                                .filter(|f| {
                                                    crate::logic::trade_field_category(f) == plan.category.as_str()
                                                })
                                                .collect();
                                        let mut ident: std::collections::HashSet<String> =
                                            std::collections::HashSet::new();
                                        ident.insert(crate::logic::TRADE_IDENTITY_FIELD.to_string());
                                        for f in all.iter() {
                                            if f.starts_with("reference_") && *f != self_ref_field {
                                                ident.insert(f.clone());
                                            }
                                        }
                                        let rest: std::collections::HashSet<String> = all
                                            .iter()
                                            .filter(|f| !ident.contains(*f))
                                            .cloned()
                                            .collect();
                                        if ident.is_empty() || rest.is_empty() {
                                            None
                                        } else {
                                            emit_term(&format!(
                                                "    🪪 [HEADER IDENTITY + PAIR] 식별 축 {}개는 문서번호 규칙이 담긴 스키마 패스로 묻고, 비식별 축 {}개는 별도 스키마 패스 대신 라벨↔값 쌍 읽기로 회수합니다. 발행일처럼 정의가 한 줄뿐인 축은 스키마로 물으면 식별 규칙에 밀려 비어 돌아오지만, 인쇄된 라벨을 옮겨 적게 하면 라벨 코사인이 축을 정합니다.",
                                                ident.len(), rest.len()
                                            ));
                                            Some((ident, rest))
                                        }
                                    }
                                } else {
                                    None
                                };

                            let pair_mode_crop = !is_array_cat;
                            let passes: Vec<(String, std::collections::HashSet<String>)> =
                                match identity_pass {
                                    None => {
                                        if pair_mode_crop {
                                            Vec::new()
                                        } else {
                                            vec![(String::new(), std::collections::HashSet::new())]
                                        }
                                    }
                                    Some((_ident, rest)) => vec![("IDENTITY".to_string(), rest)],
                                };
                            let mut schema_pass_ran = !passes.is_empty();
                            let identity_pass_ran = passes.iter().any(|(t, _)| t == "IDENTITY");
                            let mut pair_routed_fields: std::collections::HashSet<String> =
                                std::collections::HashSet::new();

                            let verify_crop = crop.clone();
                            let mut tile_json = Value::Object(serde_json::Map::new());

                            for (pass_tag, absent_in_pass) in passes.into_iter() {
                                let prompt = if pass_tag.is_empty() {
                                    crate::parsing::get_trade_crop_prompt(
                                        &plan.category,
                                        &detected_type,
                                        &plan.top_field,
                                        plan.score,
                                        &claimed,
                                    )
                                } else {
                                    crate::parsing::get_trade_crop_prompt_scoped(
                                        &plan.category,
                                        &detected_type,
                                        &plan.top_field,
                                        plan.score,
                                        &claimed,
                                        &std::collections::HashSet::new(),
                                        &absent_in_pass,
                                    )
                                };

                                let pass_res = self.chat_with_qwen3_5_image_spinner(
                                    "You are a highly precise document data extraction assistant.",
                                    &prompt,
                                    Some(verify_crop.clone()),
                                    app_handle,
                                    "extraction-progress",
                                    json!({
                                        "category": format!(
                                            "Vision (Crop {}/{}{}{})",
                                            idx + 1, plans.len(), tile_tag,
                                            if pass_tag.is_empty() { String::new() } else { format!(" / {}", pass_tag) }
                                        ),
                                        "summary": format!("Extracting {}...", plan.category)
                                    }),
                                    1024,
                                    cancel_token.clone(),
                                    Some(task_id.clone()),
                                    None
                                ).await?;

                                let parsed_pass = crate::parsing::parse_json_from_llm(&pass_res);
                                if !pass_tag.is_empty() {
                                    let filled = parsed_pass
                                        .as_object()
                                        .map(|o| {
                                            o.values()
                                                .filter(|v| {
                                                    !(v.is_null()
                                                        || v.as_str()
                                                            .map(|s| s.trim().is_empty())
                                                            .unwrap_or(false))
                                                })
                                                .count()
                                        })
                                        .unwrap_or(0);
                                    let asked = parsed_pass.as_object().map(|o| o.len()).unwrap_or(0);
                                    emit_term(&format!(
                                        "    📊 [HEADER PASS / {}] 질문 {}축 중 {}축이 채워졌습니다 (이 패스에서 뺀 축 {}개). 두 패스의 질문 축 합이 헤더 전체 축과 같아야 하며, 한쪽이 부풀어 있으면 패스 분할이 성립하지 않은 것입니다.",
                                        pass_tag, asked, filled, absent_in_pass.len()
                                    ));
                                    crate::utils::score_dynamics::record_baseline(
                                        &format!("vision.header_pass_yield.{}", pass_tag),
                                        if asked == 0 { 0.0 } else { filled as f32 / asked as f32 },
                                    );
                                }

                                if let (Some(dst), Some(src)) =
                                    (tile_json.as_object_mut(), parsed_pass.as_object())
                                {
                                    for (k, v) in src.iter() {
                                        let empty = v.is_null()
                                            || v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false);
                                        let have = dst
                                            .get(k)
                                            .map(|x| {
                                                !(x.is_null()
                                                    || x.as_str()
                                                        .map(|s| s.trim().is_empty())
                                                        .unwrap_or(false))
                                            })
                                            .unwrap_or(false);
                                        if have && empty {
                                            continue;
                                        }
                                        dst.insert(k.clone(), v.clone());
                                    }
                                } else if tile_json.as_object().map(|o| o.is_empty()).unwrap_or(false) {
                                    tile_json = parsed_pass;
                                }
                            }
                            if identity_pass_ran {
                                let off_shape: Vec<(String, String)> = tile_json
                                    .as_object()
                                    .map(|o| {
                                        o.iter()
                                            .filter(|(k, _)| {
                                                k.as_str() == crate::logic::TRADE_IDENTITY_FIELD
                                                    || k.starts_with("reference_")
                                            })
                                            .filter_map(|(k, v)| {
                                                let s = match v {
                                                    Value::String(s) => s.trim().to_string(),
                                                    Value::Number(n) => n.to_string(),
                                                    _ => return None,
                                                };
                                                if s.is_empty() || crate::model::merge::is_schema_echo(&s) {
                                                    return None;
                                                }
                                                if crate::utils::ai_utils::is_document_number_shaped(&s) {
                                                    return None;
                                                }
                                                Some((k.clone(), s))
                                            })
                                            .collect()
                                    })
                                    .unwrap_or_default();
                                if let Some(o) = tile_json.as_object_mut() {
                                    for (k, s) in off_shape.iter() {
                                        o.insert(k.clone(), Value::Null);
                                        crate::utils::score_dynamics::record_field_seen(k);
                                        crate::utils::score_dynamics::record_field_reject(
                                            k,
                                            crate::utils::score_dynamics::GateKind::Format,
                                        );
                                        emit_term(&format!(
                                            "    🚫 [IDENTITY SHAPE] 식별 패스가 '{}' 에 \"{}\" 를 돌려주었습니다. 문서번호·참조번호 축은 숫자를 품은 식별자만 담을 수 있는데 이 값에는 숫자가 없거나 날짜 모양입니다. 옆 칸의 지명·상호·문구를 식별 축으로 승격한 것이므로 잠그기 전에 비웁니다.",
                                            k, s
                                        ));
                                    }
                                }
                                let dn_now: Option<String> = tile_json
                                    .get(crate::logic::TRADE_IDENTITY_FIELD)
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.trim().to_string())
                                    .filter(|s| !s.is_empty() && !crate::model::merge::is_schema_echo(s))
                                    .or_else(|| {
                                        final_data_map
                                            .get(crate::logic::TRADE_IDENTITY_FIELD)
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.trim().to_string())
                                            .filter(|s| !s.is_empty() && !crate::model::merge::is_schema_echo(s))
                                    });
                                if let Some(dn) = dn_now {
                                    let echo: Vec<String> = tile_json
                                        .as_object()
                                        .map(|o| {
                                            o.iter()
                                                .filter(|(k, v)| {
                                                    k.starts_with("reference_")
                                                        && v.as_str()
                                                            .map(|s| crate::model::merge::same_printed_value(s.trim(), &dn))
                                                            .unwrap_or(false)
                                                })
                                                .map(|(k, _)| k.clone())
                                                .collect()
                                        })
                                        .unwrap_or_default();
                                    if let Some(o) = tile_json.as_object_mut() {
                                        for k in echo.iter() {
                                            o.insert(k.clone(), Value::Null);
                                            crate::utils::score_dynamics::record_baseline("vision.self_reference_echo", 1.0);
                                            emit_term(&format!(
                                                "    🚫 [SELF-REFERENCE ECHO] 식별 패스가 '{}' 에 자기 문서번호 \"{}\" 를 그대로 되돌려주었습니다. 참조 축은 다른 문서의 번호만 담을 수 있으므로 비웁니다. 그대로 두면 릴레이가 자기 자신을 가리키는 초안을 만듭니다.",
                                                k, dn
                                            ));
                                        }
                                    }
                                }
                                if let Some(o) = tile_json.as_object() {
                                    for (k, v) in o.iter() {
                                        let filled = !(v.is_null() || v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false));
                                        if filled
                                            && (k.as_str() == crate::logic::TRADE_IDENTITY_FIELD || k.starts_with("reference_"))
                                        {
                                            identity_locked.insert(k.clone());
                                        }
                                    }
                                }
                                if !identity_locked.is_empty() {
                                    emit_term(&format!(
                                        "    🪪 [IDENTITY LOCK] 식별 패스가 문서번호 규칙으로 확정한 축 {:?} 는 이후 어떤 크롭의 쌍 라우팅도 다른 값으로 바꾸지 못합니다. 라벨 코사인은 '어느 서식의 번호인가' 를 구별하지 못하므로 식별 축의 소유권은 규칙 쪽에 둡니다.",
                                        identity_locked
                                    ));
                                }
                            }
                            if pair_mode_crop {
                                let cat_fields: Vec<String> = schema_fields
                                    .iter()
                                    .filter(|f| crate::logic::trade_field_category(f) == plan.category.as_str())
                                    .cloned()
                                    .collect();
                                let defs: Vec<(String, String)> = cat_fields
                                    .iter()
                                    .map(|f| (f.clone(), crate::parsing::trade_field_definition(&language, f)))
                                    .collect();
                                let pair_prompt = crate::parsing::get_trade_pair_read_prompt(&detected_type, &defs);
                                let truncated_rate = crate::utils::score_dynamics::adaptive_baseline("vision.pair_read_truncated")
                                    .map(|(m, _)| m.clamp(0.0, 1.0))
                                    .unwrap_or(0.0);
                                let per_patch = (16.0 * (1.0 + truncated_rate)).ceil() as usize;
                                let pair_cap = ((lg_cnt as usize) * per_patch).clamp(384, 1024);
                                crate::utils::score_dynamics::record_baseline("vision.pair_read_cap", pair_cap as f32);
                                emit_term(&format!(
                                    "    🏷️ [PAIR READ / CROP] [{}] 축 {}개를 스키마로 묻는 대신 이 크롭에 인쇄된 라벨↔값 쌍을 전부 옮겨 적게 합니다. 라벨→축 배정은 라벨 코사인 게이트가 스키마 {}축 전체를 상대로 수행합니다. 생성 상한 {} 토큰 (판독 가능 패치 {} × {}, 384~1024, SDS 절단율 {:.3} 반영) — 쌍 하나가 약 30 토큰이고 판독 가능 패치 한 칸에 인쇄 쌍이 평균 0.4~0.5개 들어 있어, 고정 384 토큰은 표 머리글과 첫 행만으로 소진되어 둘째 행부터 잘립니다.",
                                    plan.category, cat_fields.len(), gate_banks.len(), pair_cap, lg_cnt, per_patch, truncated_rate
                                ));
                                let pair_res = self.chat_with_qwen3_5_image_spinner(
                                    "You are a highly precise document data extraction assistant.",
                                    &pair_prompt,
                                    Some(verify_crop.clone()),
                                    app_handle,
                                    "extraction-progress",
                                    json!({
                                        "category": format!("Vision (Pairs {}/{}{})", idx + 1, plans.len(), tile_tag),
                                        "summary": format!("Transcribing {} pairs...", plan.category)
                                    }),
                                    pair_cap,
                                    cancel_token.clone(),
                                    Some(task_id.clone()),
                                    None
                                ).await?;
                                let pair_open = pair_res.matches('{').count();
                                let pair_close = pair_res.matches('}').count();
                                let pair_truncated = pair_open > pair_close;
                                crate::utils::score_dynamics::record_baseline(
                                    "vision.pair_read_truncated",
                                    if pair_truncated { 1.0 } else { 0.0 },
                                );
                                if pair_truncated {
                                    emit_term(&format!(
                                        "    ✂️ [PAIR READ TRUNCATED] [{}] 쌍 읽기 응답이 생성 상한 {} 토큰에서 끊겼습니다 (여는 괄호 {} > 닫는 괄호 {}). 끊긴 뒤의 쌍은 JSON 복구가 버리므로 이 크롭의 뒷부분 라벨은 읽히지 않은 것입니다. SDS 의 vision.pair_read_truncated 평균이 0 보다 크면 패치당 토큰 계수를 올려야 합니다.",
                                        plan.category, pair_cap, pair_open, pair_close
                                    ));
                                }
                                let raw_pairs = crate::parsing::parse_json_from_llm(&pair_res);
                                let pair_complete = !pair_truncated
                                    && raw_pairs.get("pairs").map_or(false, |v| v.is_array());
                                if !pending_read.is_empty() {
                                    crate::utils::score_dynamics::record_baseline(
                                        "vision.region_read_complete",
                                        if pair_complete { 1.0 } else { 0.0 },
                                    );
                                    if pair_complete {
                                        let tile_read = legible_set_in(tile.bbox);
                                        for i in pending_read.iter().copied() {
                                            if tile_read.contains(&i) && !read_legible.contains(&i) {
                                                read_legible.push(i);
                                            }
                                        }
                                    } else {
                                        emit_term(&format!(
                                            "    📝 [REGION READ INCOMPLETE] [{}] 이 크롭의 쌍 읽기가 {} 끝나, 판독 가능 패치 {}칸을 '이미 읽은 지면' 으로 표시하지 않습니다. 뒤따르는 크롭이 같은 지면을 덮으면 REGION ALREADY READ 로 건너뛰지 않고 다시 읽습니다. 끊긴 응답 뒤의 라벨은 한 번도 읽힌 적이 없기 때문입니다.",
                                            plan.category,
                                            if pair_truncated { "생성 상한에서 끊겨" } else { "쌍 목록 없이" },
                                            pending_read.len()
                                        ));
                                    }
                                }
                                let pairs: Vec<(String, String)> = raw_pairs
                                    .get("pairs")
                                    .and_then(|v| v.as_array())
                                    .map(|arr| {
                                        arr.iter()
                                            .filter_map(|e| {
                                                let l = e.get("label").and_then(|x| x.as_str())?.trim().to_string();
                                                let v = e
                                                    .get("value")
                                                    .and_then(|x| match x {
                                                        Value::String(s) => Some(s.trim().to_string()),
                                                        Value::Number(n) => Some(n.to_string()),
                                                        _ => None,
                                                    })
                                                    .unwrap_or_default();
                                                if l.is_empty() || v.is_empty() { return None; }
                                                if crate::model::merge::is_schema_echo(&v) { return None; }
                                                Some((l, v))
                                            })
                                            .collect()
                                    })
                                    .unwrap_or_default();
                                if pairs.is_empty() {
                                    emit_term(&format!(
                                        "    ⚪ [PAIR READ / CROP EMPTY] [{}] 이 크롭에서 읽어낸 라벨↔값 쌍이 없습니다. 라벨 없는 문장은 쌍이 아니므로 어느 축에도 들어가지 않습니다.",
                                        plan.category
                                    ));
                                } else {
                                    emit_term(&format!(
                                        "    🏷️ [PAIR READ / CROP] [{}] 쌍 {}건: {:?}",
                                        plan.category,
                                        pairs.len(),
                                        pairs.iter().map(|(l, v)| format!("\"{}\"→\"{}\"", l, v)).take(10).collect::<Vec<_>>()
                                    ));
                                    crate::utils::score_dynamics::record_baseline("vision.pair_read_count", pairs.len() as f32);
                                    let labels: Vec<String> = pairs.iter().map(|(l, _)| l.clone()).collect();
                                    let pair_embs = self
                                        .get_embedding_batch(labels)
                                        .await
                                        .unwrap_or_else(|_| vec![Vec::new(); pairs.len()]);
                                    let title_table = crate::utils::ai_utils::all_trade_doc_titles();
                                    let doc_title_codes = |label: &str| -> Vec<String> {
                                        let key: String = label
                                            .to_lowercase()
                                            .chars()
                                            .filter(|c| c.is_alphanumeric())
                                            .collect();
                                        let mut hits: Vec<(String, String)> = Vec::new();
                                        for (code, title) in title_table.iter() {
                                            let t: String = title
                                                .to_lowercase()
                                                .chars()
                                                .filter(|c| c.is_alphanumeric())
                                                .collect();
                                            let n = t.chars().count();
                                            let min_n = if t.chars().any(|c| (c as u32) >= 0x2E80) { 2 } else { 5 };
                                            if n < min_n || !key.contains(&t) { continue; }
                                            match hits.iter_mut().find(|(c, _)| c == code) {
                                                Some((_, prev)) => {
                                                    if t.len() > prev.len() { *prev = t; }
                                                }
                                                None => hits.push((code.clone(), t)),
                                            }
                                        }
                                        let mut codes: Vec<(String, usize)> = Vec::new();
                                        for (code, t) in hits.iter() {
                                            let nested = hits.iter().any(|(oc, ot)| {
                                                oc != code && ot.len() > t.len() && ot.contains(t.as_str())
                                            });
                                            if nested { continue; }
                                            codes.push((code.clone(), t.chars().count()));
                                        }
                                        codes.sort_by(|a, b| b.1.cmp(&a.1));
                                        codes.into_iter().map(|(c, _)| c).collect()
                                    };
                                    let mut queue: Vec<(String, String, String, f32, bool)> = Vec::new();
                                    let mut rest_pairs: Vec<(String, String)> = Vec::new();
                                    let mut rest_embs: Vec<Vec<f32>> = Vec::new();
                                    let mut foreign_title_labels: std::collections::HashSet<String> = std::collections::HashSet::new();
                                    for (pi, (l, v)) in pairs.iter().enumerate() {
                                        let codes: Vec<String> = doc_title_codes(l)
                                            .into_iter()
                                            .filter(|c| c.as_str() != detected_type.as_str())
                                            .collect();
                                        if codes.is_empty() {
                                            rest_pairs.push((l.clone(), v.clone()));
                                            rest_embs.push(pair_embs.get(pi).cloned().unwrap_or_default());
                                            continue;
                                        }
                                        let id_shaped = crate::utils::ai_utils::is_document_number_shaped(v)
                                            && crate::utils::ai_utils::value_matches_format(
                                                crate::utils::ai_utils::FieldFormat::Identifier,
                                                v,
                                            );
                                        let target = if !id_shaped {
                                            None
                                        } else {
                                            codes.iter().find_map(|c| {
                                                crate::logic::trade_reference_field_of(c)
                                                    .filter(|rf| schema_fields.iter().any(|f| f.as_str() == *rf))
                                                    .map(|rf| (c.clone(), rf.to_string()))
                                            })
                                        };
                                        match target {
                                            Some((code, rf)) => {
                                                let emb = pair_embs.get(pi).cloned().unwrap_or_default();
                                                let (_, own, _, _) = crate::model::merge::recovery_label_gate(&emb, &rf, &gate_banks);
                                                let in_win = cat_fields.iter().any(|f| *f == rf);
                                                emit_term(&format!(
                                                    "      📑 [PAIR DOC-TITLE ROUTE] \"{}\" → \"{}\" | 라벨이 다른 서식 '{}' 의 전문을 품고 있습니다. 다른 서식의 번호는 이 문서의 doc_number 가 될 수 없고 그 서식을 가리키는 참조 축 '{}' 이므로, 라벨 코사인 경쟁에 넣지 않고 곧장 배정합니다 (자기 중립점수 {:+.4}).",
                                                    l, v, code, rf, own
                                                ));
                                                queue.push((l.clone(), v.clone(), rf, own, in_win));
                                            }
                                            None => {
                                                foreign_title_labels.insert(l.clone());
                                                rest_pairs.push((l.clone(), v.clone()));
                                                rest_embs.push(pair_embs.get(pi).cloned().unwrap_or_default());
                                                if id_shaped {
                                                    emit_term(&format!(
                                                        "      ⚪ [PAIR DOC-TITLE / NO AXIS] \"{}\" → \"{}\" | 라벨이 다른 서식 {:?} 의 전문을 품고 있지만 이 서식의 스키마에 그 서식을 가리키는 참조 축이 없습니다. 나머지 축 경쟁에는 맡기되 doc_number 로는 들어가지 못하게 막습니다.",
                                                        l, v, codes
                                                    ));
                                                } else {
                                                    emit_term(&format!(
                                                        "      ⚪ [PAIR DOC-TITLE / NOT AN ID] \"{}\" → \"{}\" | 라벨이 다른 서식 {:?} 의 전문을 품고 있지만 값이 문서번호 모양(숫자를 품은 4자 이상 코드 토큰)이 아니거나 날짜입니다. 서식 이름을 품은 수량·날짜 라벨이므로 참조 축으로 곧장 보내지 않고 나머지 축 경쟁에 맡깁니다. doc_number 로는 들어가지 못하게 막습니다.",
                                                        l, v, codes
                                                    ));
                                                }
                                            }
                                        }
                                    }
                                    let (routed, route_logs, row_axis_hits) = crate::model::merge::route_pairs_to_fields_exact(
                                        &rest_pairs, &rest_embs, &cat_fields, &gate_banks, &label_exact,
                                    );
                                    for line in route_logs.iter() { emit_term(line); }
                                    if array_as_pairs {
                                        let ident = crate::model::merge::row_identity_fields(&plan.category);
                                        let own_cols: Vec<(String, String)> = row_axis_hits
                                            .iter()
                                            .filter(|(_, f, _)| crate::logic::trade_field_category(f) == plan.category.as_str())
                                            .filter_map(|(pi, f, _)| rest_pairs.get(*pi).map(|(l, _)| (l.clone(), f.clone())))
                                            .collect();
                                        let ident_hit = own_cols
                                            .iter()
                                            .find(|(_, f)| {
                                                ident.iter().any(|k| *k == f.as_str())
                                                    && crate::utils::ai_utils::detect_field_format(f)
                                                        != crate::utils::ai_utils::FieldFormat::Enum
                                            })
                                            .cloned();
                                        crate::utils::score_dynamics::record_baseline(
                                            "vision.array_pair_table_rescue",
                                            if ident_hit.is_some() { 1.0 } else { 0.0 },
                                        );
                                        let pair_rows = match ident_hit.as_ref() {
                                            Some((il, ifld)) => crate::model::merge::rows_from_pair_sequence(
                                                &rest_pairs,
                                                &rest_embs,
                                                il,
                                                ifld,
                                                &plan.category,
                                                &gate_banks,
                                            ),
                                            None => None,
                                        };
                                        if ident_hit.is_some() {
                                            crate::utils::score_dynamics::record_baseline(
                                                "vision.array_pair_rows",
                                                pair_rows.as_ref().map_or(0.0, |r| r.rows.len() as f32),
                                            );
                                        }
                                        if let Some(rb) = pair_rows {
                                            emit_term(&format!(
                                                "    🧾 [ARRAY PAIRS → ROWS] '{}' 크롭의 쌍 읽기가 행 정체 머리글 \"{}\" 를 {}번 옮겨 적었습니다. 반복되는 행 정체 머리글을 행 경계로 삼아 (앞에 붙는 열 {}개) 쌍 순서에서 행 {}개를 직접 복원합니다. 열 {}개: {:?}{}. 같은 크롭을 표 스키마로 다시 묻지 않습니다 — 2B 모델은 표 재판독에서 첫 행만 객체 하나로 돌려줄 수 있고(행 1개), 이미 옮겨 적은 쌍의 순서가 행 구조의 더 직접적인 근거입니다.",
                                                plan.category,
                                                ident_hit.as_ref().map(|(l, _)| l.as_str()).unwrap_or(""),
                                                rb.segments,
                                                rb.lead,
                                                rb.rows.len(),
                                                rb.columns.len(),
                                                rb.columns.iter().map(|(l, f, z)| format!("{}→{}({:+.2})", l, f, z)).collect::<Vec<_>>(),
                                                if rb.format_drops.is_empty() {
                                                    String::new()
                                                } else {
                                                    format!(" | 형식 불일치로 비운 칸 {:?}", rb.format_drops)
                                                }
                                            ));
                                            crate::utils::score_dynamics::record_baseline(
                                                "vision.array_pair_row_columns",
                                                rb.columns.len() as f32,
                                            );
                                            let table_json = Value::Array(rb.rows.into_iter().map(Value::Object).collect());
                                            record_grounding_claims(&mut grounding_claims, &plan.category, &table_json, tile.bbox);
                                            merge_extracted(&mut final_data_map, &plan.category, &table_json, &emit_term);
                                        } else if let Some((ident_label, ident_field)) = ident_hit {
                                            emit_term(&format!(
                                                "    🧾 [ARRAY PAIRS → TABLE RESCUE] '{}' 크롭의 쌍 읽기가 행 정체 축 '{}' 의 인쇄 머리글 \"{}\" 를 직접 옮겨 적었습니다 (같은 표의 열 머리글 {}개: {:?}). ARRAY CROP → PAIRS 는 SigLIP2 라벨 봉우리가 크롭 밖에 있다는 좌표 근거로 표 읽기를 껐지만, 모델이 이 크롭에서 표 머리글을 읽었다는 것은 표가 이 크롭 안에 있다는 직접 근거입니다. 같은 크롭을 표 스키마로 한 번 더 읽어 행을 먼저 병합합니다. 행이 먼저 있어야 아래 쌍 라우팅의 TABLE CELL ECHO 가 'TOTAL VALUE' 같은 열 머리글의 값을 문서 총계로 올리지 않고, 미뤄진 쌍 재라우팅도 표 칸 에코를 걸러냅니다.",
                                                plan.category, ident_field, ident_label,
                                                own_cols.len(),
                                                own_cols.iter().map(|(l, _)| l.as_str()).take(10).collect::<Vec<_>>()
                                            ));
                                            let table_prompt = crate::parsing::get_trade_crop_prompt(
                                                &plan.category,
                                                &detected_type,
                                                &plan.top_field,
                                                plan.score,
                                                &[],
                                            );
                                            let table_res = self.chat_with_qwen3_5_image_spinner(
                                                "You are a highly precise document data extraction assistant.",
                                                &table_prompt,
                                                Some(verify_crop.clone()),
                                                app_handle,
                                                "extraction-progress",
                                                json!({
                                                    "category": format!("Vision (Table Rescue {}/{}{})", idx + 1, plans.len(), tile_tag),
                                                    "summary": format!("Extracting {} rows...", plan.category)
                                                }),
                                                1024,
                                                cancel_token.clone(),
                                                Some(task_id.clone()),
                                                None
                                            ).await?;
                                            let open_ident = |o: &serde_json::Map<String, Value>| -> bool {
                                                ident.iter().any(|k| {
                                                    crate::utils::ai_utils::detect_field_format(k)
                                                        != crate::utils::ai_utils::FieldFormat::Enum
                                                        && o.get(*k)
                                                            .and_then(|v| v.as_str())
                                                            .map(|s| s.trim())
                                                            .map_or(false, |s| !s.is_empty() && !crate::model::merge::is_schema_echo(s))
                                                })
                                            };
                                            let mut parsed_rows = crate::parsing::parse_json_from_llm(&table_res);
                                            let norm = |s: &str| -> String {
                                                s.chars().filter(|c| c.is_alphanumeric()).flat_map(|c| c.to_lowercase()).collect()
                                            };
                                            let wrapped = parsed_rows
                                                .as_object()
                                                .filter(|o| o.len() == 1)
                                                .and_then(|o| o.iter().next())
                                                .filter(|(k, v)| {
                                                    (v.is_array() || v.is_object())
                                                        && (norm(k) == norm(&plan.category) || norm(k) == norm(&detected_type))
                                                })
                                                .map(|(_, v)| v.clone());
                                            if let Some(inner) = wrapped {
                                                parsed_rows = inner;
                                            }
                                            let (rows_read, table_json) = match parsed_rows {
                                                Value::Array(a) => {
                                                    let n = a.len();
                                                    (n, Value::Array(a.into_iter().filter(|r| r.as_object().map_or(false, |o| open_ident(o))).collect()))
                                                }
                                                Value::Object(o) if open_ident(&o) => (1, Value::Array(vec![Value::Object(o)])),
                                                Value::Object(_) => (1, Value::Array(Vec::new())),
                                                _ => (0, Value::Array(Vec::new())),
                                            };
                                            let rows_kept = table_json.as_array().map_or(0, |a| a.len());
                                            crate::utils::score_dynamics::record_baseline("vision.array_pair_table_rows", rows_kept as f32);
                                            emit_term(&format!(
                                                "    🧾 [TABLE RESCUE ROWS] '{}' 표 재판독 행 {}개 중 닫힌 어휘가 아닌 행 정체 값을 가진 {}개를 병합합니다. 표 재판독은 ROW IDENTITY ECHO 의 목록 없는 재확인을 거치지 않으므로, 행 정체가 닫힌 어휘(규격·코드 목록) 하나에만 걸린 행은 기대 어휘 복사일 수 있어 받지 않습니다.",
                                                plan.category, rows_read, rows_kept
                                            ));
                                            if rows_kept > 0 {
                                                record_grounding_claims(&mut grounding_claims, &plan.category, &table_json, tile.bbox);
                                                merge_extracted(&mut final_data_map, &plan.category, &table_json, &emit_term);
                                            }
                                        }
                                    }
                                    for (pi, (l, v)) in rest_pairs.iter().enumerate() {
                                        if routed.iter().any(|r| r.label == *l && r.value == *v) { continue; }
                                        let emb = match rest_embs.get(pi) {
                                            Some(e) if !e.is_empty() => e.clone(),
                                            _ => continue,
                                        };
                                        if deferred_pairs.iter().any(|(dl, dv, _, _)| dl.eq_ignore_ascii_case(l) && dv.eq_ignore_ascii_case(v)) {
                                            continue;
                                        }
                                        deferred_pairs.push((l.clone(), v.clone(), emb, tile.bbox));
                                    }
                                    for r in routed.iter() {
                                        queue.push((r.label.clone(), r.value.clone(), r.field.clone(), r.own, r.in_window));
                                    }
                                    crate::utils::score_dynamics::record_baseline(
                                        "vision.pair_route_ratio",
                                        queue.len() as f32 / pairs.len().max(1) as f32,
                                    );
                                    let claimed_now = collect_claimed(&final_data_map);
                                    let scalar_text = |v: &Value| -> String {
                                        match v {
                                            Value::String(s) => s.trim().to_string(),
                                            Value::Number(n) => n.to_string(),
                                            _ => String::new(),
                                        }
                                    };
                                    let mut foreign: std::collections::HashMap<String, serde_json::Map<String, Value>> =
                                        std::collections::HashMap::new();
                                    for (label, value, field, own, in_window) in queue.into_iter() {
                                        let rcat = crate::logic::trade_field_category(&field).to_string();
                                        if rcat.is_empty() { continue; }
                                        if field.as_str() == crate::logic::TRADE_IDENTITY_FIELD && foreign_title_labels.contains(&label) {
                                            emit_term(&format!(
                                                "      🚫 [PAIR DOC-TITLE BLOCK] \"{}\" → doc_number = \"{}\" | 다른 서식의 전문을 품은 라벨이 문서 식별자로 라우팅되었습니다. 식별자가 다른 서식의 번호로 바뀌면 같은 문서가 다른 index 로 두 번 저장되고 릴레이가 엉뚱한 초안을 만듭니다. 배정하지 않습니다.",
                                                label, value
                                            ));
                                            continue;
                                        }
                                        if value.eq_ignore_ascii_case(&label)
                                            || crate::parsing::is_printed_label_echo(&value, &language)
                                            || crate::parsing::is_printed_label_fragment(&value, &language)
                                        {
                                            emit_term(&format!(
                                                "      🚫 [PAIR LABEL AS VALUE] \"{}\" → {} = \"{}\" | 값 자리에 라벨이 들어왔습니다.",
                                                label, field, value
                                            ));
                                            continue;
                                        }
                                        let multiline = value.lines().filter(|l| !l.trim().is_empty()).count() >= 2;
                                        let (field, value, rcat) = if multiline
                                            && crate::utils::ai_utils::detect_field_format(&field)
                                                != crate::utils::ai_utils::FieldFormat::Address
                                        {
                                            match crate::model::merge::party_block_axis(&label, &field, &label_exact, &gate_banks, &final_data_map) {
                                                Some((name_axis, addr_axis)) => {
                                                    let block = value
                                                        .lines()
                                                        .map(|l| l.trim())
                                                        .filter(|l| !l.is_empty())
                                                        .collect::<Vec<_>>()
                                                        .join(" / ");
                                                    crate::utils::score_dynamics::record_baseline("vision.pair_party_block", 1.0);
                                                    emit_term(&format!(
                                                        "      🧱 [PAIR PARTY BLOCK] \"{}\" → {} | 줄바꿈 블록이 당사자 상자(이름 줄 + 주소 줄)입니다. 라우팅 축 '{}' 대신 이름 축에 줄을 ' / ' 로 이어 넣고, 저장 전 정규화가 텍스트 경로와 같은 규칙으로 첫 줄은 '{}', 나머지 줄은 '{}' 로 나눕니다.",
                                                        label, name_axis, field, name_axis, addr_axis
                                                    ));
                                                    let cat = crate::logic::trade_field_category(&name_axis).to_string();
                                                    (name_axis, block, cat)
                                                }
                                                None => {
                                                    emit_term(&format!(
                                                        "      🚫 [PAIR MULTILINE] \"{}\" → {} | 줄바꿈으로 나뉜 블록은 주소 축에만 들어갈 수 있습니다. 이 축의 형식은 주소가 아니므로 배정하지 않고, 같은 크롭의 잔여 스키마 패스와 복구 창에 맡깁니다.",
                                                        label, field
                                                    ));
                                                    continue;
                                                }
                                            }
                                        } else {
                                            (field, value, rcat)
                                        };
                                        if crate::logic::is_trade_array_category(&rcat) && rcat != plan.category {
                                            let rows = final_data_map
                                                .get(&rcat)
                                                .and_then(|v| v.as_array())
                                                .map(|a| a.len())
                                                .unwrap_or(0);
                                            let row_current = final_data_map
                                                .get(&rcat)
                                                .and_then(|v| v.as_array())
                                                .and_then(|a| a.first())
                                                .and_then(|row| row.get(&field))
                                                .map(|v| scalar_text(v))
                                                .unwrap_or_default();
                                            if rows == 1
                                                && row_current.is_empty()
                                                && crate::model::merge::write_into_single_row(&mut final_data_map, &rcat, &field, &value)
                                            {
                                                let mut p = serde_json::Map::new();
                                                p.insert(field.clone(), json!(value.clone()));
                                                record_grounding_claims(&mut grounding_claims, &rcat, &Value::Object(p), tile.bbox);
                                                pair_evidence.insert(field.clone(), own);
                                                pair_label.insert(field.clone(), label.clone());
                                                crate::utils::score_dynamics::record_field_seen(&field);
                                                crate::utils::score_dynamics::record_field_assigned(&field, own);
                                                emit_term(&format!(
                                                    "      ✅ [PAIR ROUTE / ARRAY ROW WRITE] \"{}\" → {}.{} = \"{}\" | 배열 카테고리에 행이 하나뿐이라 그 행에 채웠습니다 (중립점수 {:+.4}). 새 행을 만들면 같은 당사자의 사실이 두 레코드로 갈립니다.",
                                                    label, rcat, field, value, own
                                                ));
                                                continue;
                                            }
                                            emit_term(&format!(
                                                "      ⚪ [PAIR ROUTE / ARRAY OWNER] \"{}\" → {}.{} = \"{}\" | 이 축은 다른 배열 카테고리의 것인데 행이 {}개이고 첫 행의 값은 \"{}\" 입니다. 어느 행인지 단정할 근거가 없으므로 복구 단계에 맡깁니다.",
                                                label, rcat, field, value, rows,
                                                if row_current.is_empty() { "비어 있음" } else { row_current.as_str() }
                                            ));
                                            if row_current.is_empty()
                                                && !array_deferred
                                                    .iter()
                                                    .any(|d| d.2 == field && d.1.eq_ignore_ascii_case(&value))
                                            {
                                                array_deferred.push((
                                                    label.clone(),
                                                    value.clone(),
                                                    field.clone(),
                                                    rcat.clone(),
                                                    own,
                                                    tile.bbox,
                                                ));
                                            }
                                            continue;
                                        }
                                        if !crate::logic::is_trade_array_category(&rcat) {
                                            let mut echo: Option<(String, String, usize, usize, bool)> = None;
                                            for acat in crate::logic::TRADE_ARRAY_CATEGORIES.iter() {
                                                let rows = match final_data_map.get(*acat).and_then(|v| v.as_array()) {
                                                    Some(a) => a,
                                                    None => continue,
                                                };
                                                for (ri, row) in rows.iter().enumerate() {
                                                    let o = match row.as_object() { Some(o) => o, None => continue };
                                                    let hit = o.iter().find(|(_, v)| {
                                                        let s = scalar_text(*v);
                                                        !s.is_empty() && crate::model::merge::same_printed_value(&s, &value)
                                                    });
                                                    let afield = match hit { Some((k, _)) => k.clone(), None => continue };
                                                    let label_in_row = o.iter().any(|(k, v)| {
                                                        if *k == afield { return false; }
                                                        let s = scalar_text(v);
                                                        !s.is_empty() && crate::model::merge::same_printed_value(&s, &label)
                                                    });
                                                    let better = match echo.as_ref() {
                                                        None => true,
                                                        Some(prev) => label_in_row && !prev.4,
                                                    };
                                                    if better {
                                                        echo = Some((acat.to_string(), afield, ri, rows.len(), label_in_row));
                                                    }
                                                    if label_in_row { break; }
                                                }
                                                if echo.as_ref().map_or(false, |e| e.4) { break; }
                                            }
                                            if let Some((acat, afield, ri, nrows, label_in_row)) = echo {
                                                let fmt = crate::utils::ai_utils::query_value_format(&field);
                                                let numeric_like = field.starts_with("reference_")
                                                    || matches!(
                                                        fmt,
                                                        crate::utils::ai_utils::FieldFormat::Numeric
                                                            | crate::utils::ai_utils::FieldFormat::Identifier
                                                            | crate::utils::ai_utils::FieldFormat::TrackingCode
                                                    );
                                                let mut column_own = f32::MIN;
                                                if !label_in_row && numeric_like && nrows >= 2 {
                                                    if let Some(emb) = pairs
                                                        .iter()
                                                        .position(|(l, _)| l == &label)
                                                        .and_then(|i| pair_embs.get(i))
                                                    {
                                                        let (_, o, _, _) = crate::model::merge::recovery_label_gate(emb, &afield, &gate_banks);
                                                        column_own = o;
                                                    }
                                                }
                                                let is_cell = label_in_row
                                                    || (numeric_like
                                                        && nrows >= 2
                                                        && column_own != f32::MIN
                                                        && column_own >= own - 1.0);
                                                if is_cell {
                                                    crate::utils::score_dynamics::record_baseline("vision.pair_table_cell_echo", 1.0);
                                                    crate::utils::score_dynamics::record_field_seen(&field);
                                                    crate::utils::score_dynamics::record_field_reject(
                                                        &field,
                                                        crate::utils::score_dynamics::GateKind::Prejudice,
                                                    );
                                                    emit_term(&format!(
                                                        "      🚫 [PAIR TABLE CELL ECHO] \"{}\" → {} = \"{}\" | 이 값은 이미 읽은 {} 표 {}번째 행의 '{}' 칸과 같습니다{}. 표 행의 칸을 라벨↔값 쌍으로 다시 읽은 것이므로 스칼라 축에 넣지 않습니다. 총계 축에 어느 한 행의 값이 들어가면 산술 정합이 그 총계를 지우고, 참조 축에 품목 코드가 들어가면 존재하지 않는 문서를 가리키게 됩니다.",
                                                        label, field, value, acat, ri + 1, afield,
                                                        if label_in_row {
                                                            " (라벨도 같은 행의 다른 칸입니다)".to_string()
                                                        } else {
                                                            format!(" (라벨의 '{}' 열 중립점수 {:+.4} 가 '{}' {:+.4} 와 pooled σ 한 칸 안)", afield, column_own, field, own)
                                                        }
                                                    ));
                                                    continue;
                                                }
                                            }
                                        }
                                        if let Some((owner, _)) = claimed_now
                                            .iter()
                                            .find(|(k, v)| *k != field && v.eq_ignore_ascii_case(&value))
                                        {
                                            let same_label = pair_label
                                                .get(owner)
                                                .map(|l| l.eq_ignore_ascii_case(&label))
                                                .unwrap_or(false);
                                            if same_label {
                                                emit_term(&format!(
                                                    "      🚫 [PAIR CLAIMED] \"{}\" → {} = \"{}\" | 같은 인쇄 라벨이 이미 '{}' 로 라우팅되어 있습니다. 한 라벨은 한 축만 가리킵니다.",
                                                    label, field, value, owner
                                                ));
                                                continue;
                                            }
                                            emit_term(&format!(
                                                "      ↔️ [PAIR SAME VALUE / DISTINCT LABEL] \"{}\" → {} = \"{}\" | '{}' 가 같은 값을 갖고 있지만 그 값은 다른 라벨({})에서 왔습니다. 서로 다른 두 인쇄 라벨이 같은 값을 갖는 것은 두 사실이므로 막지 않습니다. 발행일과 선적일이 같은 날짜인 서식이 그렇습니다.",
                                                label, field, value, owner,
                                                pair_label.get(owner).map(|s| s.as_str()).unwrap_or("스키마 패스")
                                            ));
                                        }
                                        let in_tile = tile_json
                                            .get(&field)
                                            .map(|v| !scalar_text(v).is_empty())
                                            .unwrap_or(false);
                                        let current = if in_tile {
                                            tile_json.get(&field).map(|v| scalar_text(v)).unwrap_or_default()
                                        } else {
                                            final_data_map.get(&field).map(|v| scalar_text(v)).unwrap_or_default()
                                        };
                                        let incumbent_ev = pair_evidence.get(&field).copied();
                                        if !current.is_empty() && !current.eq_ignore_ascii_case(&value) {
                                            if identity_locked.contains(&field) {
                                                crate::utils::score_dynamics::record_baseline("vision.pair_identity_keep", 1.0);
                                                emit_term(&format!(
                                                    "      🪪 [PAIR IDENTITY KEEP] {} = \"{}\" 는 식별 패스가 문서번호 규칙으로 확정한 값입니다. 쌍 \"{}\"→\"{}\" (중립점수 {:+.4}) 로 바꾸지 않습니다. 라벨만 따로 물어 확인해도 2B 모델은 크롭 안의 아무 번호나 읽어 확인해 주므로 재판독은 근거가 되지 못합니다.",
                                                    field, current, label, value, own
                                                ));
                                                continue;
                                            }
                                            let aggregate_keep = match incumbent_ev {
                                                Some(e) if own > e
                                                    && own - e < 1.0
                                                    && crate::model::merge::is_aggregate_axis(&detected_type, &field) =>
                                                {
                                                    match (
                                                        crate::model::merge::printed_number(&current),
                                                        crate::model::merge::printed_number(&value),
                                                    ) {
                                                        (Some(c), Some(v)) if v <= c => Some(e),
                                                        _ => None,
                                                    }
                                                }
                                                _ => None,
                                            };
                                            crate::utils::score_dynamics::record_baseline(
                                                "vision.pair_aggregate_keep",
                                                if aggregate_keep.is_some() { 1.0 } else { 0.0 },
                                            );
                                            if let Some(e) = aggregate_keep {
                                                emit_term(&format!(
                                                    "      ⚖️ [PAIR AGGREGATE KEEP] {} = \"{}\" (근거 {:+.4}) 를 쌍 \"{}\"→\"{}\" (근거 {:+.4}) 로 바꾸지 않습니다. 두 근거의 차 {:+.4} 가 pooled σ 한 칸 안이라 라벨로는 가를 수 없는데, 이 축은 스키마 정의가 총계인 축이고 총계는 자기 구성 항목보다 작을 수 없습니다. 작거나 같은 쪽은 표 행의 칸이거나 소계입니다. 이 규칙은 교체를 막기만 하고 새로 만들지는 않습니다.",
                                                    field, current, e, label, value, own, own - e
                                                ));
                                                continue;
                                            }
                                            if incumbent_ev.map_or(false, |e| e >= own) {
                                                emit_term(&format!(
                                                    "      ⚪ [PAIR OCCUPIED KEEP] {} 는 이미 \"{}\" 로 확정되어 있고 그 라벨 근거({:+.4})가 이번 쌍 \"{}\"→\"{}\" 의 근거({:+.4}) 이상입니다. 유지합니다.",
                                                    field, current, incumbent_ev.unwrap_or(f32::MIN), label, value, own
                                                ));
                                                continue;
                                            }
                                            if in_tile {
                                                if let Some(o) = tile_json.as_object_mut() {
                                                    o.insert(field.clone(), json!(value.clone()));
                                                }
                                                if rcat == plan.category {
                                                    pair_routed_fields.insert(field.clone());
                                                }
                                            } else {
                                                grounding_claims.retain(|g| {
                                                    !(g.field == field && g.value.eq_ignore_ascii_case(&current))
                                                });
                                                final_data_map.insert(field.clone(), json!(value.clone()));
                                                if let Some(o) = final_data_map.get_mut(&rcat).and_then(|v| v.as_object_mut()) {
                                                    o.insert(field.clone(), json!(value.clone()));
                                                }
                                                let mut p = serde_json::Map::new();
                                                p.insert(field.clone(), json!(value.clone()));
                                                record_grounding_claims(&mut grounding_claims, &rcat, &Value::Object(p), tile.bbox);
                                            }
                                            pair_evidence.insert(field.clone(), own);
                                            pair_label.insert(field.clone(), label.clone());
                                            crate::utils::score_dynamics::record_baseline("vision.pair_replace", 1.0);
                                            crate::utils::score_dynamics::record_confusion(&field, &field, own - incumbent_ev.unwrap_or(0.0));
                                            emit_term(&format!(
                                                "      🔁 [PAIR REPLACE] {} : \"{}\" (근거 {}, {}) → \"{}\" | 라벨 \"{}\" 중립점수 {:+.4} — 라벨 근거가 더 강한 주장이 자리를 가져갑니다.",
                                                field, current,
                                                match incumbent_ev { Some(e) => format!("{:+.4}", e), None => "없음".to_string() },
                                                if in_tile { "이 크롭의 스키마 패스" } else { "앞선 크롭" },
                                                value, label, own
                                            ));
                                            continue;
                                        }
                                        if let Some(e) = incumbent_ev {
                                            if e >= own && !current.is_empty() { continue; }
                                        }
                                        emit_term(&format!(
                                            "      🧭 [PAIR ROUTE{}] \"{}\" → {}.{} = \"{}\" | 스키마 {}축 전체와 경쟁시켜 중립점수 {:+.4}",
                                            if in_window { "" } else { " / OUT OF WINDOW" },
                                            label, rcat, field, value, gate_banks.len(), own
                                        ));
                                        pair_evidence.insert(field.clone(), own);
                                        pair_label.insert(field.clone(), label.clone());
                                        crate::utils::score_dynamics::record_field_seen(&field);
                                        crate::utils::score_dynamics::record_field_assigned(&field, own);
                                        if rcat == plan.category {
                                            if let Some(o) = tile_json.as_object_mut() {
                                                o.insert(field.clone(), json!(value.clone()));
                                            }
                                            pair_routed_fields.insert(field.clone());
                                        } else {
                                            foreign
                                                .entry(rcat.clone())
                                                .or_insert_with(serde_json::Map::new)
                                                .insert(field.clone(), json!(value.clone()));
                                        }
                                    }
                                    for (rcat, obj) in foreign.into_iter() {
                                        let v = Value::Object(obj);
                                        record_grounding_claims(&mut grounding_claims, &rcat, &v, tile.bbox);
                                        merge_extracted(&mut final_data_map, &rcat, &v, &emit_term);
                                    }
                                }
                                let residual: Vec<String> = cat_fields
                                    .iter()
                                    .filter(|f| {
                                        if identity_pass_ran
                                            && (f.as_str() == crate::logic::TRADE_IDENTITY_FIELD || f.starts_with("reference_"))
                                        {
                                            return false;
                                        }
                                        let in_tile = tile_json
                                            .get(f.as_str())
                                            .map(|v| !(v.is_null() || v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false)))
                                            .unwrap_or(false);
                                        if in_tile { return false; }
                                        let filled = final_data_map
                                            .get(f.as_str())
                                            .map(|v| !(v.is_null() || v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false)))
                                            .unwrap_or(false);
                                        if filled { return false; }
                                        if !crate::parsing::trade_expected_vocab(&plan.category, &detected_type, f).is_empty() {
                                            return false;
                                        }
                                        field_peak_inside(f, tile.bbox)
                                    })
                                    .cloned()
                                    .collect();
                                if plan.owned_patches == 0 {
                                    emit_term(&format!(
                                        "    ⚪ [RESIDUAL PASS SKIP / TERRITORY] [{}] 이 크롭 안에 자기 영토 패치가 한 칸도 없습니다 (봉우리만 있는 축 {:?}). 영토 없는 크롭에서 스키마로 물으면 모델은 옆 칸의 값을 그 축으로 승격합니다. 명시된 라벨↔값 쌍만 씁니다.",
                                        plan.category, residual
                                    ));
                                } else if array_as_pairs {
                                    emit_term(&format!(
                                        "    ⚪ [RESIDUAL PASS SKIP / ARRAY AS PAIRS] [{}] 이 크롭은 표 스키마 대신 라벨↔값 쌍으로만 읽습니다 (봉우리만 있는 축 {:?}). 표 카테고리의 축을 스키마로 다시 물으면 행 정체 축이 빈 행이 만들어져 병합 단계에서 그대로 폐기됩니다.",
                                        plan.category, residual
                                    ));
                                } else if residual.is_empty() {
                                    emit_term(&format!(
                                        "    ⚪ [RESIDUAL PASS SKIP] [{}] 쌍으로 채워지지 않았으면서 이 크롭 안에 자기 라벨 봉우리를 가진 열린 축이 없습니다. 스키마 프롬프트를 열지 않습니다. 닫힌 어휘 축은 인쇄되어 있으면 반드시 라벨↔값 쌍으로 읽히므로, 쌍에 없는 닫힌 어휘 축을 다시 물으면 어휘 복사만 돌아옵니다.",
                                        plan.category
                                    ));
                                } else {
                                    let absent: std::collections::HashSet<String> = cat_fields
                                        .iter()
                                        .filter(|f| !residual.iter().any(|r| r == *f))
                                        .cloned()
                                        .collect();
                                    emit_term(&format!(
                                        "    🎯 [RESIDUAL PASS] [{}] 쌍 읽기 뒤에도 비어 있고 이 크롭 안에 SigLIP2 라벨 봉우리를 가진 열린 축 {}개만 스키마로 묻습니다: {:?} — 봉우리가 없거나 닫힌 어휘인 축 {}개는 묻지 않습니다.",
                                        plan.category, residual.len(), residual, absent.len()
                                    ));
                                    let res_prompt = crate::parsing::get_trade_crop_prompt_scoped(
                                        &plan.category,
                                        &detected_type,
                                        &plan.top_field,
                                        plan.score,
                                        &claimed,
                                        &std::collections::HashSet::new(),
                                        &absent,
                                    );
                                    let res_out = self.chat_with_qwen3_5_image_spinner(
                                        "You are a highly precise document data extraction assistant.",
                                        &res_prompt,
                                        Some(verify_crop.clone()),
                                        app_handle,
                                        "extraction-progress",
                                        json!({
                                            "category": format!("Vision (Residual {}/{}{})", idx + 1, plans.len(), tile_tag),
                                            "summary": format!("Extracting {} residual axes...", plan.category)
                                        }),
                                        1024,
                                        cancel_token.clone(),
                                        Some(task_id.clone()),
                                        None
                                    ).await?;
                                    let parsed_res = crate::parsing::parse_json_from_llm(&res_out);
                                    let mut filled_n = 0usize;
                                    let mut pair_echo: Vec<String> = Vec::new();
                                    if let (Some(dst), Some(src)) = (tile_json.as_object_mut(), parsed_res.as_object()) {
                                        for (k, v) in src.iter() {
                                            let empty = v.is_null()
                                                || v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false);
                                            if empty { continue; }
                                            if !residual.iter().any(|r| r == k) { continue; }
                                            let have = dst
                                                .get(k)
                                                .map(|x| !(x.is_null() || x.as_str().map(|s| s.trim().is_empty()).unwrap_or(false)))
                                                .unwrap_or(false);
                                            if have { continue; }
                                            let shown = match v {
                                                Value::String(s) => s.trim().to_string(),
                                                Value::Number(n) => n.to_string(),
                                                _ => String::new(),
                                            };
                                            let echo_of: Option<(String, String)> = if shown.is_empty() {
                                                None
                                            } else {
                                                pairs.iter().find_map(|(pl, pv)| {
                                                    if !crate::model::merge::same_printed_value(pv, &shown) { return None; }
                                                    let owners: Vec<String> = pair_label
                                                        .iter()
                                                        .filter(|(_, lab)| lab.eq_ignore_ascii_case(pl))
                                                        .map(|(f, _)| f.clone())
                                                        .chain(
                                                            array_deferred
                                                                .iter()
                                                                .filter(|d| d.0.eq_ignore_ascii_case(pl) && crate::model::merge::same_printed_value(&d.1, pv))
                                                                .map(|d| d.2.clone()),
                                                        )
                                                        .collect();
                                                    if owners.is_empty() || owners.iter().any(|f| f == k) { return None; }
                                                    Some((pl.clone(), owners.join(", ")))
                                                })
                                            };
                                            if let Some((pl, owners)) = echo_of {
                                                crate::utils::score_dynamics::record_field_seen(k);
                                                crate::utils::score_dynamics::record_field_reject(
                                                    k,
                                                    crate::utils::score_dynamics::GateKind::Prejudice,
                                                );
                                                pair_echo.push(format!("{}=\"{}\" ← \"{}\"→{}", k, shown, pl, owners));
                                                continue;
                                            }
                                            dst.insert(k.clone(), v.clone());
                                            filled_n += 1;
                                        }
                                    }
                                    crate::utils::score_dynamics::record_baseline(
                                        "vision.residual_pair_echo",
                                        pair_echo.len() as f32,
                                    );
                                    if !pair_echo.is_empty() {
                                        emit_term(&format!(
                                            "    🚫 [RESIDUAL PAIR ECHO] [{}] 잔여 스키마 패스가 돌려준 값 {}건이 같은 크롭에서 다른 인쇄 라벨로 읽힌 값과 같아 비웁니다: {:?} — 잔여 패스는 쌍으로 채워지지 않은 축만 묻는데, 그 축의 라벨이 인쇄되어 있지 않으면 모델은 옆 라벨의 값을 그 축으로 옮겨 적습니다. 그 라벨이 이미 다른 축(배열 소유가 미뤄진 쌍 포함)으로 라우팅되어 있으면 이 값은 그 축의 사실입니다. 미뤄진 쌍까지 보지 않으면 서명자 이름이 송하인으로 들어갑니다.",
                                            plan.category, pair_echo.len(), pair_echo
                                        ));
                                    }
                                    crate::utils::score_dynamics::record_baseline(
                                        "vision.residual_pass_yield",
                                        filled_n as f32 / residual.len().max(1) as f32,
                                    );
                                    schema_pass_ran = true;
                                }
                            }
                            if !is_array_cat {
                                let echo_fields: Vec<(String, String)> = if !schema_pass_ran {
                                    Vec::new()
                                } else {
                                    tile_json
                                        .as_object()
                                        .map(|o| {
                                            o.iter()
                                                .filter_map(|(k, v)| {
                                                    if pair_routed_fields.contains(k.as_str()) { return None; }
                                                    let s = v.as_str()?.trim().to_string();
                                                    if s.is_empty() { return None; }
                                                    let vocab = crate::parsing::trade_expected_vocab(&plan.category, &detected_type, k);
                                                    if vocab.iter().any(|t| t.eq_ignore_ascii_case(&s)) {
                                                        Some((k.clone(), s))
                                                    } else {
                                                        None
                                                    }
                                                })
                                                .collect()
                                        })
                                        .unwrap_or_default()
                                };
                                for (field, value) in echo_fields.into_iter() {
                                    let definition = crate::parsing::trade_field_definition(&language, &field);
                                    let blind_prompt = crate::parsing::get_trade_blind_read_prompt(&detected_type, &field, &definition);
                                    let blind_res = self.chat_with_qwen3_5_image_spinner(
                                        "You are a highly precise document data extraction assistant.",
                                        &blind_prompt,
                                        Some(verify_crop.clone()),
                                        app_handle,
                                        "extraction-progress",
                                        json!({
                                            "category": format!("Vision (Verify {}/{})", idx + 1, plans.len()),
                                            "summary": format!("Verifying {}...", field)
                                        }),
                                        96,
                                        cancel_token.clone(),
                                        Some(task_id.clone()),
                                        None
                                    ).await?;
                                    let blind_value = crate::parsing::parse_json_from_llm(&blind_res)
                                        .get("value")
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.trim().to_string())
                                        .unwrap_or_default();
                                    if crate::model::merge::same_printed_token(&value, &blind_value) {
                                        emit_term(&format!(
                                            "    ✅ [VOCAB ECHO VERIFIED] [{}] '{}' = \"{}\" | 기대 어휘 목록 없이 다시 읽어도 같은 토큰이 인쇄되어 있습니다.",
                                            plan.category, field, value
                                        ));
                                    } else {
                                        emit_term(&format!(
                                            "    🚫 [VOCAB ECHO DROP] [{}] '{}' = \"{}\" | 프롬프트 기대 어휘와 같은 토큰인데, 목록 없이 다시 읽으면 \"{}\" 입니다. 인쇄되지 않은 기대 어휘 복사로 보고 폐기합니다.",
                                            plan.category, field, value,
                                            if blind_value.is_empty() { "null" } else { blind_value.as_str() }
                                        ));
                                        if let Some(o) = tile_json.as_object_mut() {
                                            o.insert(field.clone(), Value::Null);
                                        }
                                    }
                                }
                                let dup_fields: Vec<(String, String)> = {
                                    let mut seen: Vec<(String, String)> = Vec::new();
                                    let mut dup: Vec<(String, String)> = Vec::new();
                                    if let Some(o) = tile_json.as_object() {
                                        for (k, v) in o.iter() {
                                            let s = match v {
                                                Value::String(s) => s.trim().to_string(),
                                                Value::Number(n) => n.to_string(),
                                                _ => continue,
                                            };
                                            if s.is_empty() || crate::model::merge::is_schema_echo(&s) { continue; }
                                            if s.chars().filter(|c| c.is_alphanumeric()).count() < 2 { continue; }
                                            if let Some((pk, pv)) = seen
                                                .iter()
                                                .find(|(_, x)| crate::model::merge::same_printed_value(x, &s))
                                            {
                                                if !dup.iter().any(|(dk, _)| dk == pk) {
                                                    dup.push((pk.clone(), pv.clone()));
                                                }
                                                dup.push((k.clone(), s));
                                                continue;
                                            }
                                            seen.push((k.clone(), s));
                                        }
                                    }
                                    dup
                                };
                                if !dup_fields.is_empty() {
                                    emit_term(&format!(
                                        "    ♊ [INTRA-CROP DUPLICATE] [{}] 한 크롭 응답 안에서 같은 값이 서로 다른 축 {}개에 배정되었습니다: {:?} — 인쇄된 한 자리는 라벨을 하나만 가지므로 이 중 최대 하나만 참입니다. 각 축의 라벨만 따로 물어 확인합니다.",
                                        plan.category,
                                        dup_fields.len(),
                                        dup_fields.iter().map(|(f, v)| format!("{}=\"{}\"", f, v)).collect::<Vec<_>>()
                                    ));
                                }
                                for (field, value) in dup_fields.into_iter() {
                                    if pair_routed_fields.contains(field.as_str()) {
                                        emit_term(&format!(
                                            "    ✅ [PAIR-BACKED DUPLICATE KEEP] [{}] '{}' = \"{}\" | 인쇄된 라벨을 옮겨 적어 라우팅된 값이라 라벨 근거를 이미 갖고 있습니다. 다시 묻지 않습니다.",
                                            plan.category, field, value
                                        ));
                                        continue;
                                    }
                                    let backed_by_pair = tile_json
                                        .as_object()
                                        .map(|o| {
                                            o.iter().any(|(k, v)| {
                                                if k == &field || !pair_routed_fields.contains(k.as_str()) { return false; }
                                                let s = match v {
                                                    Value::String(s) => s.trim().to_string(),
                                                    Value::Number(n) => n.to_string(),
                                                    _ => return false,
                                                };
                                                crate::model::merge::same_printed_value(&s, &value)
                                            })
                                        })
                                        .unwrap_or(false);
                                    if backed_by_pair {
                                        crate::utils::score_dynamics::record_baseline("vision.dup_confirm", 0.0);
                                        crate::utils::score_dynamics::record_field_seen(&field);
                                        crate::utils::score_dynamics::record_field_reject(
                                            &field,
                                            crate::utils::score_dynamics::GateKind::Prejudice,
                                        );
                                        emit_term(&format!(
                                            "    🚫 [PAIR-BACKED DUPLICATE DROP] [{}] '{}' = \"{}\" | 같은 값이 라벨 근거를 가진 쌍 라우팅 축에 이미 배정되어 있습니다. 스키마 패스가 그 인쇄 칸을 옆 축으로 복사한 것이므로 재판독 없이 비웁니다. 라벨 없이 물어 확인하면 모델은 같은 칸을 또 읽어 복사를 확인해 주므로, 근거 없는 쪽이 근거 있는 쪽을 밀어내는 역전이 생깁니다.",
                                            plan.category, field, value
                                        ));
                                        if let Some(o) = tile_json.as_object_mut() {
                                            o.insert(field.clone(), Value::Null);
                                        }
                                        continue;
                                    }
                                    let definition = crate::parsing::trade_field_definition(&language, &field);
                                    let blind_prompt = crate::parsing::get_trade_blind_read_prompt(&detected_type, &field, &definition);
                                    let blind_res = self.chat_with_qwen3_5_image_spinner(
                                        "You are a highly precise document data extraction assistant.",
                                        &blind_prompt,
                                        Some(verify_crop.clone()),
                                        app_handle,
                                        "extraction-progress",
                                        json!({
                                            "category": format!("Vision (Duplicate {}/{})", idx + 1, plans.len()),
                                            "summary": format!("Confirming {}...", field)
                                        }),
                                        96,
                                        cancel_token.clone(),
                                        Some(task_id.clone()),
                                        None
                                    ).await?;
                                    let blind_value = crate::parsing::parse_json_from_llm(&blind_res)
                                        .get("value")
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.trim().to_string())
                                        .unwrap_or_default();
                                    if crate::model::merge::same_printed_value(&value, &blind_value) {
                                        crate::utils::score_dynamics::record_baseline("vision.dup_confirm", 1.0);
                                        emit_term(&format!(
                                            "    ✅ [DUPLICATE CONFIRMED] [{}] '{}' = \"{}\" | 이 축의 라벨만 물었을 때도 같은 값이 돌아옵니다. 이 자리에 이 축의 라벨이 실제로 인쇄되어 있습니다.",
                                            plan.category, field, value
                                        ));
                                    } else {
                                        crate::utils::score_dynamics::record_baseline("vision.dup_confirm", 0.0);
                                        crate::utils::score_dynamics::record_field_seen(&field);
                                        crate::utils::score_dynamics::record_field_reject(
                                            &field,
                                            crate::utils::score_dynamics::GateKind::Prejudice,
                                        );
                                        emit_term(&format!(
                                            "    🚫 [DUPLICATE DROP] [{}] '{}' = \"{}\" | 이 축의 라벨만 물으면 \"{}\" 입니다. 같은 값을 나눠 가진 다른 축의 라벨을 보고 이 축까지 채운 것이므로 폐기합니다. 합이 총계와 맞아떨어지는 배분은 사후 검출이 불가능하므로 주장 시점에 끊어야 합니다.",
                                            plan.category, field, value,
                                            if blind_value.is_empty() { "null" } else { blind_value.as_str() }
                                        ));
                                        if let Some(o) = tile_json.as_object_mut() {
                                            o.insert(field.clone(), Value::Null);
                                        }
                                    }
                                }
                            }
                            if is_array_cat {
                                let ident = crate::model::merge::row_identity_fields(&plan.category);
                                let mut suspects: Vec<(usize, String, String)> = Vec::new();
                                if !ident.is_empty() {
                                    let rows: Vec<&Value> = match &tile_json {
                                        Value::Array(a) => a.iter().collect(),
                                        v @ Value::Object(_) => vec![v],
                                        _ => Vec::new(),
                                    };
                                    for (ri, row) in rows.iter().enumerate() {
                                        let o = match row.as_object() {
                                            Some(o) => o,
                                            None => continue,
                                        };
                                        let filled: Vec<(String, String)> = ident
                                            .iter()
                                            .filter_map(|k| {
                                                let s = o.get(*k)?.as_str()?.trim().to_string();
                                                if s.is_empty() || crate::model::merge::is_schema_echo(&s) {
                                                    None
                                                } else {
                                                    Some((k.to_string(), s))
                                                }
                                            })
                                            .collect();
                                        if filled.is_empty() {
                                            continue;
                                        }
                                        let all_vocab = filled.iter().all(|(k, v)| {
                                            let vocab = crate::parsing::trade_expected_vocab(&plan.category, &detected_type, k);
                                            !vocab.is_empty() && crate::model::merge::closed_vocab_echo(v, &vocab)
                                        });
                                        if !all_vocab {
                                            continue;
                                        }
                                        for (k, v) in filled.into_iter() {
                                            suspects.push((ri, k, v));
                                        }
                                    }
                                }
                                if !suspects.is_empty() {
                                    emit_term(&format!(
                                        "    🧾 [ROW IDENTITY ECHO] [{}] 행 정체성이 닫힌 어휘 토큰만으로 성립한 축 {}건: {:?} — 기대 어휘 목록 없이 같은 타일을 다시 읽어 실제 인쇄 여부를 확인합니다. 배열 카테고리는 스칼라의 VOCAB ECHO 검증을 거치지 않았고, 행 정체성이 닫힌 어휘 하나에만 걸려 있으면 그 행 전체가 기대 어휘 복사로 생성된 것일 수 있습니다.",
                                        plan.category,
                                        suspects.len(),
                                        suspects.iter().map(|(ri, k, v)| format!("행{} {}=\"{}\"", ri, k, v)).collect::<Vec<_>>()
                                    ));
                                }
                                for (ri, field, value) in suspects.into_iter() {
                                    let definition = crate::parsing::trade_field_definition(&language, &field);
                                    let blind_prompt = crate::parsing::get_trade_blind_read_prompt(&detected_type, &field, &definition);
                                    let blind_res = self.chat_with_qwen3_5_image_spinner(
                                        "You are a highly precise document data extraction assistant.",
                                        &blind_prompt,
                                        Some(verify_crop.clone()),
                                        app_handle,
                                        "extraction-progress",
                                        json!({
                                            "category": format!("Vision (Row Verify {}/{})", idx + 1, plans.len()),
                                            "summary": format!("Verifying {}...", field)
                                        }),
                                        96,
                                        cancel_token.clone(),
                                        Some(task_id.clone()),
                                        None
                                    ).await?;
                                    let blind_value = crate::parsing::parse_json_from_llm(&blind_res)
                                        .get("value")
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.trim().to_string())
                                        .unwrap_or_default();
                                    if crate::model::merge::same_printed_value(&value, &blind_value) {
                                        crate::utils::score_dynamics::record_baseline("vision.row_identity_echo_confirm", 1.0);
                                        emit_term(&format!(
                                            "    ✅ [ROW IDENTITY CONFIRMED] [{}] 행{} '{}' = \"{}\" | 기대 어휘 목록 없이 다시 읽어도 같은 토큰이 인쇄되어 있습니다.",
                                            plan.category, ri, field, value
                                        ));
                                        continue;
                                    }
                                    crate::utils::score_dynamics::record_baseline("vision.row_identity_echo_confirm", 0.0);
                                    crate::utils::score_dynamics::record_field_seen(&field);
                                    crate::utils::score_dynamics::record_field_reject(
                                        &field,
                                        crate::utils::score_dynamics::GateKind::Prejudice,
                                    );
                                    emit_term(&format!(
                                        "    🚫 [ROW IDENTITY ECHO DROP] [{}] 행{} '{}' = \"{}\" | 목록 없이 다시 읽으면 \"{}\" 입니다. 인쇄되지 않은 기대 어휘 복사이므로 비웁니다. 이 축이 비면 행 정체성이 사라져 병합 단계에서 행 자체가 폐기됩니다.",
                                        plan.category, ri, field, value,
                                        if blind_value.is_empty() { "null" } else { blind_value.as_str() }
                                    ));
                                    match &mut tile_json {
                                        Value::Array(a) => {
                                            if let Some(o) = a.get_mut(ri).and_then(|v| v.as_object_mut()) {
                                                o.insert(field.clone(), Value::Null);
                                            }
                                        }
                                        Value::Object(o) => {
                                            o.insert(field.clone(), Value::Null);
                                        }
                                        _ => {}
                                    }
                                }
                            }
                            let violation_scan = match &tile_json {
                                Value::Object(o) if !pair_routed_fields.is_empty() => Value::Object(
                                    o.iter()
                                        .filter(|(k, _)| !pair_routed_fields.contains(*k))
                                        .map(|(k, v)| (k.clone(), v.clone()))
                                        .collect(),
                                ),
                                _ => tile_json.clone(),
                            };
                            record_claim_violations(
                                &claimed,
                                &violation_scan,
                                &plan.category,
                                &emit_term,
                            );
                            // 🌟 병합 '전' 에 이 타일이 주장한 값을 출처 bbox 와 함께 기록합니다.
                            //    STEP 6 이 이 목록으로 접지 검증을 수행합니다.
                            {
                                let mut filled = 0usize;
                                let mut total = 0usize;
                                let mut count_obj = |o: &serde_json::Map<String, Value>,
                                                     filled: &mut usize,
                                                     total: &mut usize| {
                                    for (_, v) in o.iter() {
                                        *total += 1;
                                        let empty = v.is_null()
                                            || v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false);
                                        if !empty { *filled += 1; }
                                    }
                                };
                                if let Some(o) = tile_json.as_object() {
                                    count_obj(o, &mut filled, &mut total);
                                } else if let Some(a) = tile_json.as_array() {
                                    for e in a.iter() {
                                        if let Some(o) = e.as_object() {
                                            count_obj(o, &mut filled, &mut total);
                                        }
                                    }
                                }
                                if total > 0 {
                                    crate::utils::score_dynamics::record_baseline(
                                        "vision.crop_yield",
                                        filled as f32 / total as f32,
                                    );
                                }
                            }
                            record_grounding_claims(
                                &mut grounding_claims,
                                &plan.category,
                                &tile_json,
                                tile.bbox,
                            );
                            merge_extracted(&mut final_data_map, &plan.category, &tile_json, &emit_term);
                        }
                    }
                    {
                        let dropped = crate::models::siglip2::value_grounding::retain_merged_claims(
                            &mut grounding_claims,
                            &final_data_map,
                        );
                        crate::utils::score_dynamics::record_baseline(
                            "vision.unmerged_claims",
                            dropped.len() as f32,
                        );
                        if !dropped.is_empty() {
                            emit_term(&format!(
                                "  🧹 [CLAIM PRUNE / CROP LOOP] 크롭이 주장했지만 병합이 받아들이지 않은 값 {}건을 접지 목록에서 뺍니다: {:?} — 주장은 병합 '전' 에 기록되므로, 식별 잠금에 막힌 값이나 정체 없는 행으로 폐기된 값이 저장되지 않았는데도 복구 가지치기·접지 검증·중복 소유 경쟁의 근거로 남아 있었습니다.",
                                dropped.len(),
                                dropped
                                    .iter()
                                    .map(|c| format!("{}.{}=\"{}\"", c.category, c.field, c.value))
                                    .take(8)
                                    .collect::<Vec<_>>()
                            ));
                        }
                    }
                    if !deferred_pairs.is_empty() {
                        let scalar_banks: Vec<(String, Vec<Vec<f32>>, Vec<f32>)> = gate_banks
                            .iter()
                            .filter(|(f, _, _)| {
                                let c = crate::logic::trade_field_category(f);
                                !c.is_empty()
                                    && !crate::logic::is_trade_array_category(c)
                                    && f.as_str() != crate::logic::TRADE_IDENTITY_FIELD
                            })
                            .cloned()
                            .collect();
                        let scalar_fields: Vec<String> = scalar_banks.iter().map(|(f, _, _)| f.clone()).collect();
                        let mut keep: Vec<(String, String)> = Vec::new();
                        let mut keep_embs: Vec<Vec<f32>> = Vec::new();
                        let mut keep_bbox: Vec<(u32, u32, u32, u32)> = Vec::new();
                        let mut cell_echo = 0usize;
                        let mut row_label = 0usize;
                        {
                            let is_table_cell = |value: &str| -> Option<(String, String)> {
                                for acat in crate::logic::TRADE_ARRAY_CATEGORIES.iter() {
                                    let rows = match final_data_map.get(*acat).and_then(|v| v.as_array()) {
                                        Some(a) => a,
                                        None => continue,
                                    };
                                    for row in rows.iter() {
                                        let o = match row.as_object() { Some(o) => o, None => continue };
                                        for (k, v) in o.iter() {
                                            let s = match v {
                                                Value::String(s) => s.trim().to_string(),
                                                Value::Number(n) => n.to_string(),
                                                _ => continue,
                                            };
                                            if s.is_empty() { continue; }
                                            if crate::model::merge::same_printed_value(&s, value) {
                                                return Some((acat.to_string(), k.clone()));
                                            }
                                        }
                                    }
                                }
                                None
                            };
                            for (label, value, emb, bbox) in deferred_pairs.iter() {
                                if let Some((acat, afield)) = is_table_cell(value) {
                                    cell_echo += 1;
                                    emit_term(&format!(
                                        "      🧾 [DEFERRED PAIR / TABLE CELL] \"{}\" → \"{}\" | 값이 {} 표의 '{}' 칸과 같습니다. 표 행을 라벨↔값 쌍으로 읽은 것이므로 스칼라 재라우팅에서 제외합니다.",
                                        label, value, acat, afield
                                    ));
                                    continue;
                                }
                                let ranking = crate::model::merge::label_axis_scores(emb, &gate_banks);
                                if let Some((top_f, top_z)) = ranking.first().cloned() {
                                    let top_cat = crate::logic::trade_field_category(&top_f);
                                    if !top_cat.is_empty() && crate::logic::is_trade_array_category(top_cat) {
                                        let scalar_best = ranking
                                            .iter()
                                            .find(|(f, _)| scalar_fields.iter().any(|s| s == f))
                                            .cloned();
                                        let marker = crate::model::merge::label_has_aggregate_marker(label);
                                        let why = match scalar_best.as_ref() {
                                            None => Some("스칼라 축 후보가 없습니다.".to_string()),
                                            Some((sf, sz)) => {
                                                if !crate::model::merge::row_axis_aggregate_related(&top_f, sf) {
                                                    Some(format!(
                                                        "스칼라 1순위 '{}'({:+.4}) 와 표 열 '{}' 의 이름이 같은 양을 가리키지 않습니다.",
                                                        sf, sz, top_f
                                                    ))
                                                } else if !marker && *sz < top_z - 1.0 {
                                                    Some(format!(
                                                        "라벨에 총계 표지가 없고 스칼라 1순위 '{}'({:+.4}) 가 표 열보다 pooled σ 한 칸 넘게 뒤집니다 (격차 {:+.4}).",
                                                        sf, sz, top_z - sz
                                                    ))
                                                } else {
                                                    None
                                                }
                                            }
                                        };
                                        if let Some(why) = why {
                                            row_label += 1;
                                            emit_term(&format!(
                                                "      🚫 [DEFERRED PAIR / ROW LABEL] \"{}\" → \"{}\" | 스키마 전체 라벨 경쟁의 1위가 표 열 '{}'({:+.4}) 입니다. {} 미뤄진 쌍 재라우팅은 표 열이 argmax 를 가져가 막힌 문서 총계를 되살리는 경로이지, 표 머리글을 스칼라 축에 다시 꽂는 경로가 아닙니다. 표 열을 뺀 풀의 1위는 원래 1위가 빠진 자리의 잡음이라, 그대로 두면 품목 코드가 계약번호로, 단위중량이 순중량 총계로 들어가고 같은 축을 노리던 진짜 총계 라벨이 경합에서 밀립니다.",
                                                label, value, top_f, top_z, why
                                            ));
                                            continue;
                                        }
                                    }
                                }
                                keep.push((label.clone(), value.clone()));
                                keep_embs.push(emb.clone());
                                keep_bbox.push(*bbox);
                            }
                        }
                        crate::utils::score_dynamics::record_baseline(
                            "vision.deferred_pair_row_label",
                            row_label as f32 / deferred_pairs.len().max(1) as f32,
                        );
                        emit_term(&format!(
                            "  🔁 [DEFERRED PAIR PASS] 크롭 루프에서 라우팅되지 못한 쌍 {}건 중 표 칸 에코 {}건과 표 머리글 라벨 {}건을 제외한 {}건을 스칼라 축 {}개만으로 다시 라우팅합니다. 표 열이 라벨 argmax 를 가져가 막힌 총계·중량 축은 표를 다 읽은 뒤에야 '어느 행의 값도 아니다' 를 확인할 수 있습니다. 표 머리글 라벨은 재라우팅 전에 빼야, 같은 축을 노리던 총계 라벨이 경합에서 머리글에 밀리지 않습니다.",
                            deferred_pairs.len(), cell_echo, row_label, keep.len(), scalar_fields.len()
                        ));
                        if !keep.is_empty() && !scalar_banks.is_empty() {
                            let (routed, logs, _) = crate::model::merge::route_pairs_to_fields_exact(
                                &keep, &keep_embs, &scalar_fields, &scalar_banks, &label_exact,
                            );
                            for line in logs.iter() { emit_term(line); }
                            let mut adopted = 0usize;
                            for r in routed.iter() {
                                let bbox = keep
                                    .iter()
                                    .position(|(l, v)| *l == r.label && *v == r.value)
                                    .and_then(|i| keep_bbox.get(i).copied())
                                    .unwrap_or((0, 0, grid.orig_width, grid.orig_height));
                                let rcat = crate::logic::trade_field_category(&r.field).to_string();
                                if rcat.is_empty() || crate::logic::is_trade_array_category(&rcat) { continue; }
                                if identity_locked.contains(&r.field) { continue; }
                                if r.value.eq_ignore_ascii_case(&r.label)
                                    || crate::model::merge::is_schema_echo(&r.value)
                                    || crate::parsing::is_printed_label_echo(&r.value, &language)
                                    || crate::parsing::is_printed_label_fragment(&r.value, &language)
                                {
                                    continue;
                                }
                                let current = final_data_map
                                    .get(&r.field)
                                    .map(|v| match v {
                                        Value::String(s) => s.trim().to_string(),
                                        Value::Number(n) => n.to_string(),
                                        _ => String::new(),
                                    })
                                    .unwrap_or_default();
                                if !current.is_empty() {
                                    emit_term(&format!(
                                        "      ⚪ [DEFERRED PAIR / OCCUPIED] \"{}\" → {} = \"{}\" | 이미 \"{}\" 가 있습니다. 미뤄진 쌍은 빈 축만 채웁니다.",
                                        r.label, r.field, r.value, current
                                    ));
                                    continue;
                                }
                                let claimed = collect_claimed(&final_data_map);
                                if let Some((owner, _)) = claimed.iter().find(|(k, v)| *k != r.field && v.eq_ignore_ascii_case(&r.value)) {
                                    emit_term(&format!(
                                        "      🚫 [DEFERRED PAIR / CLAIMED] \"{}\" → {} = \"{}\" | 이미 '{}' 가 확정한 값입니다.",
                                        r.label, r.field, r.value, owner
                                    ));
                                    continue;
                                }
                                final_data_map.insert(r.field.clone(), json!(r.value.clone()));
                                if let Some(o) = final_data_map.get_mut(&rcat).and_then(|v| v.as_object_mut()) {
                                    o.insert(r.field.clone(), json!(r.value.clone()));
                                }
                                let mut p = serde_json::Map::new();
                                p.insert(r.field.clone(), json!(r.value.clone()));
                                record_grounding_claims(&mut grounding_claims, &rcat, &Value::Object(p), bbox);
                                pair_evidence.insert(r.field.clone(), r.own);
                                pair_label.insert(r.field.clone(), r.label.clone());
                                crate::utils::score_dynamics::record_field_seen(&r.field);
                                crate::utils::score_dynamics::record_field_assigned(&r.field, r.own);
                                adopted += 1;
                                emit_term(&format!(
                                    "      ✅ [DEFERRED PAIR ROUTE] \"{}\" → {}.{} = \"{}\" | 스칼라 축 {}개와 경쟁시켜 중립점수 {:+.4}",
                                    r.label, rcat, r.field, r.value, scalar_fields.len(), r.own
                                ));
                            }
                            crate::utils::score_dynamics::record_baseline(
                                "vision.deferred_pair_adopt",
                                adopted as f32 / keep.len().max(1) as f32,
                            );
                        }
                    }
                    {
                        let recovery_ceiling = plans.len().max(4);
                        let mut cands: Vec<(String, String, usize, f32)> = Vec::new();
                        let is_filled = |f: &str| -> bool {
                            let non_empty = |v: &Value| -> bool {
                                !(v.is_null() || v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false))
                            };
                            if final_data_map.get(f).map(|v| non_empty(v)).unwrap_or(false) {
                                return true;
                            }
                            let cat = crate::logic::trade_field_category(f);
                            if cat.is_empty() || !crate::logic::is_trade_array_category(cat) {
                                return false;
                            }
                            final_data_map
                                .get(cat)
                                .and_then(|v| v.as_array())
                                .map(|rows| rows.iter().any(|r| r.get(f).map(|v| non_empty(v)).unwrap_or(false)))
                                .unwrap_or(false)
                        };
                        let cols = grid.grid_cols.max(1);
                        let cw = grid.orig_width as f32 / cols as f32;
                        let ch = grid.orig_height as f32 / grid.grid_rows.max(1) as f32;
                        let mut filled_peaks: Vec<usize> = Vec::new();
                        let mut unsourced: Vec<String> = Vec::new();
                        for hm in heatmaps.iter() {
                            for (field, patch, _) in hm.field_peaks.iter() {
                                if !is_filled(field) { continue; }
                                let px = ((patch % cols) as f32 + 0.5) * cw;
                                let py = ((patch / cols) as f32 + 0.5) * ch;
                                let sourced = grounding_claims.iter().any(|g| {
                                    g.field == *field
                                        && px >= g.bbox.0 as f32
                                        && px <= g.bbox.2 as f32
                                        && py >= g.bbox.1 as f32
                                        && py <= g.bbox.3 as f32
                                });
                                if sourced {
                                    if !filled_peaks.contains(patch) { filled_peaks.push(*patch); }
                                } else if !unsourced.iter().any(|u| u == field) {
                                    unsourced.push(field.clone());
                                }
                            }
                        }
                        if !unsourced.is_empty() {
                            emit_term(&format!(
                                "  🧭 [RECOVERY PEAK SOURCE] 값은 채워졌지만 그 값을 읽은 크롭이 자기 라벨 봉우리 칸을 품지 않은 필드 {}개: {:?} — 이 필드들의 봉우리 칸은 '이미 다른 값의 출처' 라는 근거가 없으므로 그 칸을 공유한 빈 필드를 가지치기하지 않습니다. 표 열의 값은 표 행에서 읽히는데 봉우리는 표 밖 총계 라벨에 찍힐 수 있고, 패치 한 칸이 글자 행 여러 줄을 덮으면 봉우리를 공유해도 라벨은 서로 다릅니다.",
                                unsourced.len(), unsourced
                            ));
                        }
                        let mut verify_fields: Vec<String> = Vec::new();
                        let mut pruned: Vec<String> = Vec::new();
                        let mut history_skip: Vec<String> = Vec::new();
                        for hm in heatmaps.iter() {
                            if crate::logic::TRADE_ARRAY_CATEGORIES.iter().any(|c| *c == hm.category.as_str()) { continue; }
                            for (field, patch, z) in hm.field_peaks.iter() {
                                if field.starts_with("__") || field == "doc_type" { continue; }
                                let legible = legibility.verdict.get(*patch).copied()
                                    == Some(crate::models::siglip2::legibility::PatchLegibility::Legible);
                                if !legible { continue; }
                                if is_filled(field) {
                                    if crate::utils::ai_utils::detect_field_format(field) != crate::utils::ai_utils::FieldFormat::Text { continue; }
                                    let px = ((patch % cols) as f32 + 0.5) * cw;
                                    let py = ((patch / cols) as f32 + 0.5) * ch;
                                    let sources: Vec<(u32, u32, u32, u32)> = grounding_claims
                                        .iter()
                                        .filter(|g| g.field == *field)
                                        .map(|g| g.bbox)
                                        .collect();
                                    if sources.is_empty() { continue; }
                                    let mut covered = false;
                                    let mut at_edge = false;
                                    for b in sources.iter() {
                                        let inside = px >= b.0 as f32 && px <= b.2 as f32
                                            && py >= b.1 as f32 && py <= b.3 as f32;
                                        if !inside { continue; }
                                        covered = true;
                                        let dx = (px - b.0 as f32).min(b.2 as f32 - px);
                                        let dy = (py - b.1 as f32).min(b.3 as f32 - py);
                                        if dx <= cw || dy <= ch { at_edge = true; }
                                    }
                                    if covered && !at_edge { continue; }
                                    if covered {
                                        emit_term(&format!(
                                            "      ✂️ [PEAK AT CROP EDGE] {}.{} = \"{}\" | 라벨 봉우리가 자기 출처 크롭의 테두리에서 패치 한 칸 이내입니다. 봉우리를 포함했다는 사실만으로는 값이 온전하다는 증거가 되지 않습니다. 값이 크롭 경계에서 잘렸을 수 있으므로 재판독 대상에 넣습니다.",
                                            hm.category, field,
                                            final_data_map.get(field).and_then(|v| v.as_str()).unwrap_or("")
                                        ));
                                    }
                                    verify_fields.push(field.clone());
                                    cands.push((hm.category.clone(), field.clone(), *patch, *z));
                                    continue;
                                }
                                if filled_peaks.contains(patch) {
                                    pruned.push(field.clone());
                                    continue;
                                }
                                let hit_rate = crate::utils::score_dynamics::adaptive_baseline(&format!("vision.recovery_hit.v2.{}", field))
                                    .map(|(m, _)| m);
                                if hit_rate.map_or(false, |m| m <= 0.0) {
                                    history_skip.push(field.clone());
                                    continue;
                                }
                                cands.push((hm.category.clone(), field.clone(), *patch, *z * hit_rate.unwrap_or(1.0)));
                            }
                        }
                        if !pruned.is_empty() {
                            emit_term(&format!(
                                "  ✂️ [RECOVERY PRUNED] 봉우리 칸이 이미 채워진 필드의 봉우리와 같은 빈 필드 {}개를 제외합니다 (그 칸의 라벨은 이미 다른 값의 출처입니다): {:?}",
                                pruned.len(), pruned
                            ));
                        }
                        if !history_skip.is_empty() {
                            emit_term(&format!(
                                "  📉 [RECOVERY HISTORY SKIP] SDS vision.recovery_hit 이력상 복구가 한 번도 성공하지 못한 필드를 제외합니다: {:?}",
                                history_skip
                            ));
                        }
                        if !verify_fields.is_empty() {
                            emit_term(&format!(
                                "  🔁 [PEAK VERIFY] 값을 추출한 크롭이 자기 라벨 봉우리를 포함하지 않은 필드 {:?} 를 봉우리에서 다시 읽어 검증합니다.",
                                verify_fields
                            ));
                        }
                        let (verify_cands, empty_cands): (Vec<_>, Vec<_>) = cands
                            .iter()
                            .cloned()
                            .partition(|(_, f, _, _)| verify_fields.iter().any(|v| v == f));
                        let shared_peaks: Vec<usize> = {
                            let mut count: std::collections::HashMap<usize, usize> =
                                std::collections::HashMap::new();
                            for hm in heatmaps.iter() {
                                for (_, patch, _) in hm.field_peaks.iter() {
                                    *count.entry(*patch).or_insert(0) += 1;
                                }
                            }
                            count.into_iter().filter(|(_, n)| *n >= 2).map(|(p, _)| p).collect()
                        };
                        let (shared_cands, solo_cands): (Vec<_>, Vec<_>) = empty_cands
                            .iter()
                            .cloned()
                            .partition(|(_, _, p, _)| shared_peaks.iter().any(|x| x == p));
                        if !shared_cands.is_empty() {
                            emit_term(&format!(
                                "  🤝 [RECOVERY SHARED POOL] 봉우리를 다른 필드와 공유해 순위가 밀린 빈 필드 {}개에 별도 창 몫을 배정합니다: {:?} — 공유는 좌표 경쟁의 결과일 뿐 그 필드의 z 가 낮다는 뜻이 아닙니다. 한 줄로 세우면 공유 필드는 구조적으로 영원히 복구되지 않습니다.",
                                shared_cands.len(),
                                shared_cands.iter().map(|(_, f, _, z)| format!("{}(z {:+.2})", f, z)).take(8).collect::<Vec<_>>()
                            ));
                        }
                        let budget_of = |list: &Vec<(String, String, usize, f32)>, label: &str| -> usize {
                            if list.is_empty() { return 0; }
                            let mut seats: Vec<usize> = Vec::new();
                            for (_, _, p, _) in list.iter() {
                                if !seats.iter().any(|x| x == p) { seats.push(*p); }
                            }
                            let picked = seats.len().min(recovery_ceiling);
                            emit_term(&format!(
                                "    📐 [RECOVERY BUDGET / {}] 후보 {}개 | 서로 다른 봉우리 칸 {}개 → 창 {}개 (상한 {}회는 이 문서가 이미 지불한 크롭 호출 수입니다). z 평균+표준편차 게이트를 철회합니다. 원소가 둘뿐인 풀에서는 그 게이트가 수학적으로 항상 최댓값과 같아 정확히 하나만 통과시켰고, 열다섯 개 풀에서도 봉우리를 공유해 순위가 밀린 필드를 0.06 차이로 잘라냈습니다. 같은 칸을 가리키는 필드는 한 창에 묶이므로 창 수를 후보 수가 아니라 칸 수로 세면 호출이 늘지 않습니다.",
                                label, list.len(), seats.len(), picked, recovery_ceiling
                            ));
                            crate::utils::score_dynamics::record_baseline("vision.recovery_budget", picked as f32);
                            picked
                        };
                        let mut windows = crate::model::merge::plan_recovery_windows(
                            &verify_cands,
                            grid.grid_rows,
                            grid.grid_cols,
                            grid.orig_width,
                            grid.orig_height,
                            budget_of(&verify_cands, "PEAK VERIFY"),
                        );
                        for (pool, pool_label) in [(&solo_cands, "EMPTY SOLO"), (&shared_cands, "EMPTY SHARED")] {
                            let b = budget_of(pool, pool_label);
                            if b == 0 { continue; }
                            for w in crate::model::merge::plan_recovery_windows(
                                pool,
                                grid.grid_rows,
                                grid.grid_cols,
                                grid.orig_width,
                                grid.orig_height,
                                b,
                            ) {
                                // 🌟 [WINDOW OVERLAP MERGE] 픽셀 완전 일치만 보던 검사를
                                //    중심 포함 관계로 바꾸고, 충돌 시 버리는 대신 합칩니다.
                                //    실측에서 창 6·7 이 서로의 중심을 품은 채 따로 호출되었고,
                                //    창 7 만 읽은 "INVOICE TOTAL"→"2000.00" 이 존재했습니다.
                                //    버리면 그 정답이 사라지므로 합쳐서 한 번에 읽습니다.
                                let hit = windows.iter().position(|(bx, _)| {
                                    crate::model::merge::recovery_window_merge(*bx, w.0).is_some()
                                });
                                match hit {
                                    Some(i) => {
                                        let u = match crate::model::merge::recovery_window_merge(windows[i].0, w.0) {
                                            Some(u) => u,
                                            None => { windows.push(w); continue; }
                                        };
                                        let added: Vec<String> = w.1.iter()
                                            .filter(|(_, f, _)| !windows[i].1.iter().any(|(_, x, _)| x == f))
                                            .map(|(_, f, _)| f.clone())
                                            .collect();
                                        emit_term(&format!(
                                            "    🔗 [RECOVERY WINDOW OVERLAP MERGE] px({},{})-({},{}) 와 px({},{})-({},{}) 는 서로의 중심을 품고 있습니다. 두 창을 px({},{})-({},{}) 하나로 합치고 필드 {:?} 를 편입합니다. 겹치는 두 창은 같은 지면을 두 번 읽어 호출만 늘리는데, 버리면 그쪽 창만 읽은 라벨↔값 쌍이 통째로 사라집니다. 합친 사각형이 따로 읽을 때보다 픽셀을 더 먹지 않을 때만 병합합니다.",
                                            windows[i].0.0, windows[i].0.1, windows[i].0.2, windows[i].0.3,
                                            w.0.0, w.0.1, w.0.2, w.0.3,
                                            u.0, u.1, u.2, u.3, added
                                        ));
                                        windows[i].0 = u;
                                        for f in w.1.into_iter() {
                                            if windows[i].1.iter().any(|(_, x, _)| *x == f.1) { continue; }
                                            windows[i].1.push(f);
                                        }
                                        crate::utils::score_dynamics::record_baseline("vision.window_overlap_merge", 1.0);
                                    }
                                    None => windows.push(w),
                                }
                            }
                        }
                        if windows.is_empty() {
                            emit_term("  ⚪ [FIELD RECOVERY] 비어 있으면서 자기 라벨 봉우리를 가진 필드가 없습니다.");
                        } else {
                            emit_term(&format!(
                                "  🩺 [FIELD RECOVERY] 후보 {}개 (빈 필드 {} = 단독 {} + 공유 {} · 재검증 {}) | 창 하나에 필드 하나로 소형 크롭 {}개를 다시 읽습니다.",
                                cands.len(), empty_cands.len(), solo_cands.len(), shared_cands.len(),
                                verify_cands.len(), windows.len()
                            ));
                            emit_term(&format!(
                                "    📖 [RECOVERY LABEL BANK] 크롭 루프 앞에서 세운 스키마 필드 {}개의 라벨 뱅크를 그대로 재사용합니다. 메인 루프의 쌍 라우팅이 남긴 라벨 근거 {}건을 복구 창의 REROUTE KEEP 판정에 이어받습니다.",
                                gate_banks.len(), pair_evidence.len()
                            ));
                            let mut label_evidence: std::collections::HashMap<String, f32> =
                                pair_evidence.clone();
                            for (wi, (bbox, fields)) in windows.into_iter().enumerate() {
                                if cancel_token
                                    .as_ref()
                                    .map_or(false, |t| t.load(std::sync::atomic::Ordering::Relaxed))
                                {
                                    break;
                                }
                                let (lg, _, _) = legibility.count_in_bbox(bbox, grid.orig_width, grid.orig_height);
                                if lg == 0 { continue; }
                                let micro_plan = crate::models::siglip2::vision_crop::CropPlan {
                                    category: fields[0].0.clone(),
                                    bbox,
                                    score: fields[0].2,
                                    margin: 0.0,
                                    patch_count: 0,
                                    top_field: fields[0].1.clone(),
                                    owned_patches: 0,
                                    twin_of: String::new(),
                                };
                                let micro = crate::models::siglip2::vision_crop::crop_region_clamped(
                                    &dynamic_image, &micro_plan, 512, height_baseline, &emit_term,
                                );
                                let micro_verify = micro.clone();
                                let defs: Vec<(String, String)> = fields
                                    .iter()
                                    .map(|(_, f, _)| (f.clone(), crate::parsing::trade_field_definition(&language, f)))
                                    .collect();
                                emit_term(&format!(
                                    "    🔎 [RECOVERY CROP {}] px({},{})-({},{}) | 필드 {:?}",
                                    wi + 1, bbox.0, bbox.1, bbox.2, bbox.3,
                                    fields.iter().map(|(c, f, z)| format!("{}.{}(z {:+.2})", c, f, z)).collect::<Vec<_>>()
                                ));

                                let pair_mode = fields.len() >= 2;
                                let prompt = if pair_mode {
                                    emit_term(&format!(
                                        "    🏷️ [PAIR READ] 창 {}: 축 {}개를 각각 묻는 대신 인쇄된 라벨↔값 쌍을 전부 옮겨 적게 합니다. 정의가 한 줄뿐인 축을 여러 개 나열하면 2B 모델이 '이 값이 어느 축인가' 를 스스로 판정해야 하고, 날짜 축 7개처럼 정의가 서로 구별되지 않으면 전부 null 을 돌려줍니다. 라벨→축 배정은 라벨 코사인 게이트의 일이므로 모델에게서 그 일을 빼앗습니다.",
                                        wi + 1, fields.len()
                                    ));
                                    crate::parsing::get_trade_pair_read_prompt(&detected_type, &defs)
                                } else {
                                    crate::parsing::get_trade_recovery_prompt(&detected_type, &defs)
                                };
                                let res = self.chat_with_qwen3_5_image_spinner(
                                    "You are a highly precise document data extraction assistant.",
                                    &prompt,
                                    Some(micro),
                                    app_handle,
                                    "extraction-progress",
                                    json!({
                                        "category": format!("Vision (Recovery {})", wi + 1),
                                        "summary": if pair_mode { "Transcribing label/value pairs..." } else { "Re-reading empty fields..." }
                                    }),
                                    if pair_mode { 384 } else { 160 },
                                    cancel_token.clone(),
                                    Some(task_id.clone()),
                                    None
                                ).await?;
                                let raw_parsed = crate::parsing::parse_json_from_llm(&res);
                                // 🌟 [EXTRA FIELDS] 창이 묻지 않았지만 읽힌 라벨이 가리킨 축입니다.
                                //    아래 확정 루프는 이 축들도 창 축과 똑같은 게이트를 통과시킵니다.
                                let mut extra_fields: Vec<(String, String, f32)> = Vec::new();
                                let parsed = if !pair_mode {
                                    raw_parsed
                                } else {
                                    let pairs: Vec<(String, String)> = raw_parsed
                                        .get("pairs")
                                        .and_then(|v| v.as_array())
                                        .map(|arr| {
                                            arr.iter()
                                                .filter_map(|e| {
                                                    let l = e.get("label").and_then(|x| x.as_str())?.trim().to_string();
                                                    let v = e
                                                        .get("value")
                                                        .and_then(|x| match x {
                                                            Value::String(s) => Some(s.trim().to_string()),
                                                            Value::Number(n) => Some(n.to_string()),
                                                            _ => None,
                                                        })
                                                        .unwrap_or_default();
                                                    if l.is_empty() || v.is_empty() { return None; }
                                                    if crate::model::merge::is_schema_echo(&v) { return None; }
                                                    Some((l, v))
                                                })
                                                .collect()
                                        })
                                        .unwrap_or_default();
                                    if pairs.is_empty() {
                                        emit_term(&format!(
                                            "      ⚪ [PAIR READ EMPTY] 창 {} 에서 읽어낸 라벨↔값 쌍이 없습니다.",
                                            wi + 1
                                        ));
                                        Value::Object(serde_json::Map::new())
                                    } else {
                                        emit_term(&format!(
                                            "      🏷️ [PAIR READ] 창 {} 에서 쌍 {}건을 읽었습니다: {:?}",
                                            wi + 1,
                                            pairs.len(),
                                            pairs.iter().map(|(l, v)| format!("\"{}\"→\"{}\"", l, v)).take(8).collect::<Vec<_>>()
                                        ));
                                        crate::utils::score_dynamics::record_baseline(
                                            "vision.pair_read_count",
                                            pairs.len() as f32,
                                        );
                                        let labels: Vec<String> = pairs.iter().map(|(l, _)| l.clone()).collect();
                                        let pair_embs = self
                                            .get_embedding_batch(labels)
                                            .await
                                            .unwrap_or_else(|_| vec![Vec::new(); pairs.len()]);
                                        let window_fields: Vec<String> =
                                            fields.iter().map(|(_, f, _)| f.clone()).collect();
                                        let (routed, route_logs, _) = crate::model::merge::route_pairs_to_fields_exact(
                                            &pairs, &pair_embs, &window_fields, &gate_banks, &label_exact,
                                        );
                                        for line in route_logs.iter() { emit_term(line); }
                                        let mut obj = serde_json::Map::new();
                                        for r in routed.iter() {
                                            emit_term(&format!(
                                                "      🧭 [PAIR ROUTE{}] \"{}\" → {} = \"{}\" | 스키마 {}축 전체와 경쟁시켜 중립점수 {:+.4} 로 확정했습니다. 창은 '어디를 볼지' 를 정한 좌표 근거일 뿐이고, 읽어낸 라벨은 '그것이 무엇인지' 를 말하는 직접 근거입니다. 좌표 근거로 직접 근거를 가두면 창 안에 정답 축이 없을 때 반드시 오배정이 생깁니다.",
                                                if r.in_window { "" } else { " / OUT OF WINDOW" },
                                                r.label, r.field, r.value, gate_banks.len(), r.own
                                            ));
                                            if !r.in_window {
                                                let c = crate::logic::trade_field_category(&r.field).to_string();
                                                if !extra_fields.iter().any(|(_, f, _)| *f == r.field) {
                                                    extra_fields.push((c, r.field.clone(), r.own));
                                                }
                                            }
                                            obj.insert(
                                                r.field.clone(),
                                                json!({ "label": r.label, "value": r.value }),
                                            );
                                        }
                                        crate::utils::score_dynamics::record_baseline(
                                            "vision.pair_route_ratio",
                                            routed.len() as f32 / pairs.len().max(1) as f32,
                                        );
                                        Value::Object(obj)
                                    }
                                };
                                // 🌟 [EFFECTIVE FIELDS] 창이 물은 축 + 라벨이 데려온 축.
                                //    두 집합에 같은 게이트(라벨 근거 / 값 형식 / 선점 / 이송)를 적용해야
                                //    창 밖 축만 검증이 무른 경로가 생기지 않습니다.
                                let eff_fields: Vec<(String, String, f32)> = {
                                    let mut v = fields.clone();
                                    for e in extra_fields.into_iter() {
                                        if v.iter().any(|(_, f, _)| *f == e.1) { continue; }
                                        emit_term(&format!(
                                            "      ➕ [WINDOW FIELD EXPAND] 창 {} 이 묻지 않았지만 읽힌 라벨이 가리킨 축 '{}'({}) 를 확정 대상에 편입합니다.",
                                            wi + 1, e.1, e.0
                                        ));
                                        v.push(e);
                                    }
                                    v
                                };
                                for (cat, field, _) in eff_fields.iter() {
                                    let is_verify = verify_fields.iter().any(|f| f == field);
                                    let current = final_data_map
                                        .get(field)
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("")
                                        .to_string();
                                    let hit_axis = format!("vision.recovery_hit.v2.{}", field);
                                    let node = parsed.get(field);
                                    let label = node
                                        .and_then(|n| n.get("label"))
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.trim().to_string())
                                        .unwrap_or_default();
                                    let value = node
                                        .and_then(|n| n.get("value"))
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.trim().to_string())
                                        .unwrap_or_default();
                                    if value.is_empty() {
                                        if is_verify {
                                            emit_term(&format!(
                                                "      ⚪ [VERIFY KEEP] {}.{} = \"{}\" | 봉우리 재판독이 비어 기존 값을 유지합니다.",
                                                cat, field, current
                                            ));
                                        } else {
                                            crate::utils::score_dynamics::record_baseline(&hit_axis, 0.0);
                                            emit_term(&format!(
                                                "      ⚪ [RECOVERY NULL] {}.{} | 이 영역에 값이 없다고 답했습니다.",
                                                cat, field
                                            ));
                                        }
                                        continue;
                                    }
                                    // 🌟 [RECOVERY OCCUPIED] 이미 확정된 축에 다른 값이 들어오면
                                    //    merge_extracted 의 '기존 스칼라 유지' 규칙이 조용히 버립니다.
                                    //    그 전에 끊어야 blind read 호출 1회와 오해를 부르는
                                    //    ✅ [RECOVERED] 로그, 그리고 접지 주장 오염이 사라집니다.
                                    //    실측: 창 8 의 "CONSIGNEE VAT/EORI"→amount 가 창 4 의
                                    //    "INVOICE TOTAL"→amount 를 덮으려다 여기서 멈춥니다.
                                    if !is_verify
                                        && !current.is_empty()
                                        && !current.eq_ignore_ascii_case(&value)
                                    {
                                        crate::utils::score_dynamics::record_baseline(&hit_axis, 0.0);
                                        crate::utils::score_dynamics::record_confusion(
                                            field, field, 0.0,
                                        );
                                        emit_term(&format!(
                                            "      ⚪ [RECOVERY OCCUPIED] {}.{} 는 이미 \"{}\" 로 확정되어 있습니다. 이 창이 읽은 \"{}\" 는 같은 축을 두고 뒤에 온 주장이므로 채택하지 않습니다. 먼저 온 값이 라벨 근거와 함께 들어왔다면 순서가 곧 강도입니다.",
                                            cat, field, current, value
                                        ));
                                        continue;
                                    }
                                    let mut blind_confirmed = false;
                                    if label.is_empty() {
                                        let definition = crate::parsing::trade_field_definition(&language, field);
                                        let blind_prompt = crate::parsing::get_trade_blind_read_prompt(&detected_type, field, &definition);
                                        let blind_res = self.chat_with_qwen3_5_image_spinner(
                                            "You are a highly precise document data extraction assistant.",
                                            &blind_prompt,
                                            Some(micro_verify.clone()),
                                            app_handle,
                                            "extraction-progress",
                                            json!({
                                                "category": format!("Vision (Recovery Verify {})", wi + 1),
                                                "summary": format!("Confirming {}...", field)
                                            }),
                                            96,
                                            cancel_token.clone(),
                                            Some(task_id.clone()),
                                            None
                                        ).await?;
                                        let blind_value = crate::parsing::parse_json_from_llm(&blind_res)
                                            .get("value")
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.trim().to_string())
                                            .unwrap_or_default();
                                        if crate::model::merge::same_printed_token(&value, &blind_value) {
                                            blind_confirmed = true;
                                            crate::utils::score_dynamics::record_baseline("vision.labelless_confirm", 1.0);
                                            emit_term(&format!(
                                                "      ✅ [LABELLESS CONFIRMED] {}.{} = \"{}\" | 라벨을 읽지 못했지만 기대 필드명 없이 같은 창을 다시 읽어도 같은 토큰이 인쇄되어 있습니다. 라벨이 값과 다른 칸에 있거나 창 경계 밖일 뿐이므로 값을 버리지 않습니다.",
                                                cat, field, value
                                            ));
                                        } else {
                                            crate::utils::score_dynamics::record_baseline("vision.labelless_confirm", 0.0);
                                            if !is_verify {
                                                crate::utils::score_dynamics::record_baseline(&hit_axis, 0.0);
                                            }
                                            emit_term(&format!(
                                                "      🚫 [LABELLESS DROP] {}.{} = \"{}\" | 라벨을 읽지 못했고, 기대 필드명 없이 다시 읽으면 \"{}\" 입니다. 인쇄되지 않은 값으로 보고 폐기합니다.",
                                                cat, field, value,
                                                if blind_value.is_empty() { "null" } else { blind_value.as_str() }
                                            ));
                                            continue;
                                        }
                                    }
                                    if crate::model::merge::is_schema_echo(&value)
                                        || value.eq_ignore_ascii_case(&label)
                                        || crate::parsing::is_printed_label_echo(&value, &language)
                                        || crate::parsing::is_printed_label_fragment(&value, &language)
                                    {
                                        if !is_verify { crate::utils::score_dynamics::record_baseline(&hit_axis, 0.0); }
                                        emit_term(&format!(
                                            "      🚫 [RECOVERY LABEL AS VALUE] {}.{} = \"{}\" | 값 자리에 라벨이 들어왔습니다.",
                                            cat, field, value
                                        ));
                                        continue;
                                    }
                                    if is_verify
                                        && (value.eq_ignore_ascii_case(&current) || crate::model::merge::same_printed_token(&value, &current))
                                    {
                                        emit_term(&format!(
                                            "      ✅ [VERIFY CONFIRMED] {}.{} = \"{}\" | 자기 라벨 봉우리에서 다시 읽어도 같은 값입니다.",
                                            cat, field, current
                                        ));
                                        continue;
                                    }
                                    let claimed = collect_claimed(&final_data_map);
                                    if let Some((owner, _)) = claimed.iter().find(|(k, v)| k != field && v.eq_ignore_ascii_case(&value)) {
                                        if !is_verify { crate::utils::score_dynamics::record_baseline(&hit_axis, 0.0); }
                                        emit_term(&format!(
                                            "      🚫 [RECOVERY CLAIMED] {}.{} = \"{}\" | 이미 '{}' 가 확정한 값입니다.",
                                            cat, field, value, owner
                                        ));
                                        continue;
                                    }
                                    let (ok, own, rival, rival_field) = if blind_confirmed {
                                        (true, 0.0f32, 0.0f32, String::new())
                                    } else {
                                        let label_emb = self.get_embedding(label.clone()).await.unwrap_or_default();
                                        crate::model::merge::recovery_label_gate(&label_emb, field, &gate_banks)
                                    };
                                    let evidence = if blind_confirmed {
                                        "라벨 미판독 + 기대 필드명 없는 재판독 일치".to_string()
                                    } else {
                                        format!(
                                            "라벨 \"{}\" (자기 중립점수 {:+.4} vs 최강 경쟁 '{}' {:+.4})",
                                            label,
                                            own,
                                            if rival_field.is_empty() { "-" } else { rival_field.as_str() },
                                            rival
                                        )
                                    };
                                    let mut ok = ok;
                                    if !ok && !rival_field.is_empty() {
                                        let rival_in_window = eff_fields.iter().any(|(_, f, _)| *f == rival_field);
                                        let (pass, why) = crate::utils::ai_utils::window_assign_verdict(
                                            own, rival, !rival_in_window,
                                        );
                                        if pass {
                                            emit_term(&format!(
                                                "      🧷 [WINDOW ARGMAX] {}.{} = \"{}\" | 스키마 전체로는 '{}'({:+.4}) 가 라벨 argmax 이지만 그 축은 이 창에서 묻지 않았고, 이 축의 자기 중립점수는 {:+.4} 입니다. 근거: {}. 이 창이 물은 필드 {:?} 안에서 1위이므로 통과시킵니다.",
                                                cat, field, value, rival_field, rival, own, why,
                                                eff_fields.iter().map(|(_, f, _)| f.clone()).collect::<Vec<_>>()
                                            ));
                                            crate::utils::score_dynamics::record_baseline("vision.window_argmax", 1.0);
                                            ok = true;
                                        } else if !rival_in_window {
                                            crate::utils::score_dynamics::record_baseline("vision.window_argmax", 0.0);
                                            emit_term(&format!(
                                                "      🚫 [WINDOW ARGMAX BLOCKED] {}.{} = \"{}\" | 이 창이 그 축 하나만 물었으므로 창 안 argmax 는 자동으로 자기 자신입니다. 자기 중립점수 {:+.4} vs 경쟁 축 '{}' {:+.4} — {}. 양수라는 이유만으로 통과시키면 '이름' 계열 축 전부가 평균 위에 서는 라벨(SIGNATORY NAME)이 창을 연 축으로 흘러듭니다. 아래 REROUTE 로 소유 축을 찾습니다.",
                                                cat, field, value, own, rival_field, rival, why
                                            ));
                                        }
                                    }
                                    if !ok {
                                        let fmt_ok = crate::utils::ai_utils::value_matches_format(
                                            crate::utils::ai_utils::detect_field_format(&rival_field),
                                            &value,
                                        );
                                        let in_schema = schema_fields.iter().any(|f| f == &rival_field);
                                        if rival_field.is_empty() || !in_schema || !fmt_ok {
                                            if !is_verify { crate::utils::score_dynamics::record_baseline(&hit_axis, 0.0); }
                                            emit_term(&format!(
                                                "      🚫 [RECOVERY LABEL GATE] {}.{} = \"{}\" | {} — 자기 필드가 argmax 가 아니고, 이길 필드로 옮길 수도 없어 폐기합니다. (스키마 소속 {} · 값 형식 {})",
                                                cat, field, value, evidence, in_schema, fmt_ok
                                            ));
                                            continue;
                                        }
                                        let incumbent = final_data_map
                                            .get(&rival_field)
                                            .and_then(|v| v.as_str())
                                            .map(|s| s.trim().to_string())
                                            .unwrap_or_default();
                                        let incumbent_ev = label_evidence.get(&rival_field).copied();
                                        if !incumbent.is_empty() {
                                            if incumbent.eq_ignore_ascii_case(&value) {
                                                emit_term(&format!(
                                                    "      ⚪ [REROUTE SAME] {}.{} 의 값이 이미 '{}' 에 같은 문자열로 들어 있습니다. 중복 기록하지 않습니다.",
                                                    cat, field, rival_field
                                                ));
                                                continue;
                                            }
                                            if incumbent_ev.map_or(false, |e| e >= rival) {
                                                emit_term(&format!(
                                                    "      ⚪ [REROUTE KEEP] {}.{} = \"{}\" 를 '{}' 로 옮기려 했으나, 그 자리의 \"{}\" 가 더 강한 라벨 근거({:+.4} ≥ {:+.4})를 갖고 있어 유지합니다.",
                                                    cat, field, value, rival_field, incumbent,
                                                    incumbent_ev.unwrap_or(f32::MIN), rival
                                                ));
                                                continue;
                                            }
                                            grounding_claims.retain(|g| {
                                                !(g.field == rival_field && g.value.eq_ignore_ascii_case(&incumbent))
                                            });
                                        }
                                        let rcat = crate::logic::trade_field_category(&rival_field);
                                        let write_cat = if rcat.is_empty() { cat.as_str() } else { rcat };
                                        let mut rpatch = serde_json::Map::new();
                                        rpatch.insert(rival_field.clone(), json!(value.clone()));
                                        let rpatch = Value::Object(rpatch);
                                        record_grounding_claims(&mut grounding_claims, write_cat, &rpatch, bbox);
                                        final_data_map.insert(rival_field.clone(), json!(value.clone()));
                                        if !rcat.is_empty() && !crate::logic::is_trade_array_category(rcat) {
                                            let slot = final_data_map
                                                .entry(rcat.to_string())
                                                .or_insert_with(|| Value::Object(serde_json::Map::new()));
                                            if let Some(o) = slot.as_object_mut() {
                                                o.insert(rival_field.clone(), json!(value.clone()));
                                            }
                                        } else if !rcat.is_empty() {
                                            // 🌟 [ARRAY ROW WRITE] 배열 카테고리로 이송된 스칼라를 행에도 넣습니다.
                                            //    행이 하나뿐일 때만 기입합니다. 여럿이면 '어느 행인가' 의 근거가 없습니다.
                                            let wrote = crate::model::merge::write_into_single_row(
                                                &mut final_data_map, rcat, &rival_field, &value,
                                            );
                                            emit_term(&format!(
                                                "      {} [ARRAY ROW WRITE] '{}' 는 배열 카테고리 '{}' 의 축입니다. {}",
                                                if wrote { "✅" } else { "⚪" }, rival_field, rcat,
                                                if wrote {
                                                    "행이 하나뿐이라 그 행에 채웠습니다. 루트에만 두면 자연어 변환이 같은 당사자의 사실을 서로 다른 절로 쪼갭니다.".to_string()
                                                } else {
                                                    "행이 없거나 둘 이상이라 어느 행인지 단정할 근거가 없습니다. 루트에만 둡니다.".to_string()
                                                }
                                            ));
                                        }
                                        label_evidence.insert(rival_field.clone(), rival);
                                        crate::utils::score_dynamics::record_field_seen(&rival_field);
                                        crate::utils::score_dynamics::record_field_assigned(&rival_field, rival);
                                        crate::utils::score_dynamics::record_confusion(&rival_field, field, rival - own);
                                        crate::utils::score_dynamics::record_baseline("vision.recovery_reroute", 1.0);
                                        emit_term(&format!(
                                            "      🔀 [RECOVERY REROUTE] {}.{} 가 아니라 '{}' 로 확정합니다. 값 \"{}\" | {} | 이전 값 \"{}\" (근거 {}) — 읽힌 라벨이 가리키는 필드가 정답이고, 그 자리에 라벨 근거 없이 먼저 들어온 값은 교체 대상입니다.",
                                            cat, field, rival_field, value, evidence,
                                            if incumbent.is_empty() { "없음" } else { incumbent.as_str() },
                                            match incumbent_ev { Some(e) => format!("{:+.4}", e), None => "없음".to_string() }
                                        ));
                                        continue;
                                    }
                                    let mut patch = serde_json::Map::new();
                                    patch.insert(field.clone(), json!(value.clone()));
                                    let patch = Value::Object(patch);
                                    if is_verify {
                                        grounding_claims.retain(|g| !(g.field == *field && g.value.eq_ignore_ascii_case(&current)));
                                        record_grounding_claims(&mut grounding_claims, cat, &patch, bbox);
                                        final_data_map.insert(field.clone(), json!(value.clone()));
                                        let slot = final_data_map
                                            .entry(cat.clone())
                                            .or_insert_with(|| Value::Object(serde_json::Map::new()));
                                        if let Some(o) = slot.as_object_mut() {
                                            o.insert(field.clone(), json!(value.clone()));
                                        }
                                        label_evidence.insert(field.clone(), own);
                                        emit_term(&format!(
                                            "      🔁 [VERIFY REPLACED] {}.{}: \"{}\" → \"{}\" | {}",
                                            cat, field, current, value, evidence
                                        ));
                                        continue;
                                    }
                                    crate::utils::score_dynamics::record_baseline(&hit_axis, 1.0);
                                    record_grounding_claims(&mut grounding_claims, cat, &patch, bbox);
                                    // 🌟 [ARRAY ROW WRITE] 배열 카테고리에 merge_extracted 를 그대로 태우면
                                    //    ARRAY COERCE 가 축 하나만 담은 새 행을 만들어 같은 당사자가 두 행으로 갈립니다.
                                    //    행이 하나뿐이면 그 행에 채우고, 그럴 수 없을 때만 기존 병합에 맡깁니다.
                                    let row_done = crate::logic::is_trade_array_category(cat)
                                        && crate::model::merge::write_into_single_row(
                                            &mut final_data_map, cat, field, &value,
                                        );
                                    if row_done {
                                        final_data_map.insert(field.clone(), json!(value.clone()));
                                        emit_term(&format!(
                                            "      ✅ [ARRAY ROW WRITE] {}.{} 를 기존 행 1건에 채웠습니다. 새 행을 만들면 같은 당사자의 사실이 두 레코드로 갈립니다.",
                                            cat, field
                                        ));
                                    } else {
                                        merge_extracted(&mut final_data_map, cat, &patch, &emit_term);
                                    }
                                    label_evidence.insert(field.clone(), own);
                                    emit_term(&format!(
                                        "      ✅ [RECOVERED] {}.{} = \"{}\" | {}",
                                        cat, field, value, evidence
                                    ));
                                }
                            }
                        }
                    }

                    if !array_deferred.is_empty() {
                        let overlaps = |a: (u32, u32, u32, u32), b: (u32, u32, u32, u32)| -> bool {
                            a.0 < b.2 && b.0 < a.2 && a.1 < b.3 && b.1 < a.3
                        };
                        let mut owner_cats: Vec<String> = Vec::new();
                        for d in array_deferred.iter() {
                            if !owner_cats.iter().any(|c| *c == d.3) {
                                owner_cats.push(d.3.clone());
                            }
                        }
                        for rcat in owner_cats.iter() {
                            let group: Vec<&(String, String, String, String, f32, (u32, u32, u32, u32))> =
                                array_deferred.iter().filter(|d| d.3 == *rcat).collect();
                            let rows_now = final_data_map
                                .get(rcat.as_str())
                                .and_then(|v| v.as_array())
                                .map(|a| a.len())
                                .unwrap_or(0);
                            if rows_now == 0 {
                                let mut seen_fields: Vec<&str> = Vec::new();
                                let mut dup = false;
                                for d in group.iter() {
                                    if seen_fields.iter().any(|f| *f == d.2.as_str()) {
                                        dup = true;
                                    }
                                    seen_fields.push(d.2.as_str());
                                }
                                let mut reach: Vec<bool> = vec![false; group.len()];
                                let mut stack: Vec<usize> = vec![0];
                                reach[0] = true;
                                while let Some(i) = stack.pop() {
                                    for j in 0..group.len() {
                                        if !reach[j] && overlaps(group[i].5, group[j].5) {
                                            reach[j] = true;
                                            stack.push(j);
                                        }
                                    }
                                }
                                let connected = reach.iter().all(|r| *r);
                                if dup || !connected {
                                    crate::utils::score_dynamics::record_baseline("vision.array_owner_resolved", 0.0);
                                    emit_term(&format!(
                                        "  ⚪ [ARRAY OWNER RESOLVE SKIP] '{}' 는 복구가 끝난 뒤에도 행이 0개인데 미뤄 둔 쌍 {:?} 가 {}. 한 행으로 묶을 근거가 없어 버립니다.",
                                        rcat,
                                        group.iter().map(|d| format!("\"{}\"→{}=\"{}\"", d.0, d.2, d.1)).collect::<Vec<_>>(),
                                        if dup { "같은 축을 두 번 주장합니다 (서로 다른 당사자)" } else { "서로 겹치지 않는 지면에서 왔습니다" }
                                    ));
                                    continue;
                                }
                                let mut row = serde_json::Map::new();
                                for d in group.iter() {
                                    row.insert(d.2.clone(), json!(d.1.clone()));
                                    let mut p = serde_json::Map::new();
                                    p.insert(d.2.clone(), json!(d.1.clone()));
                                    record_grounding_claims(&mut grounding_claims, rcat, &Value::Object(p), d.5);
                                    pair_evidence.insert(d.2.clone(), d.4);
                                    pair_label.insert(d.2.clone(), d.0.clone());
                                    crate::utils::score_dynamics::record_field_seen(&d.2);
                                    crate::utils::score_dynamics::record_field_assigned(&d.2, d.4);
                                }
                                crate::utils::score_dynamics::record_baseline("vision.array_owner_resolved", 1.0);
                                emit_term(&format!(
                                    "  ✅ [ARRAY OWNER RESOLVE / NEW ROW] '{}' 는 복구가 끝난 뒤에도 행이 0개입니다. 미뤄 둔 쌍 {:?} 는 서로 겹치는 지면에서 왔고 같은 축을 두 번 주장하지 않으므로 한 당사자의 사실로 보고 행 하나로 만듭니다.",
                                    rcat,
                                    group.iter().map(|d| format!("\"{}\"→{}=\"{}\"", d.0, d.2, d.1)).collect::<Vec<_>>()
                                ));
                                merge_extracted(&mut final_data_map, rcat, &Value::Object(row), &emit_term);
                                continue;
                            }
                            if rows_now != 1 {
                                crate::utils::score_dynamics::record_baseline("vision.array_owner_resolved", 0.0);
                                emit_term(&format!(
                                    "  ⚪ [ARRAY OWNER RESOLVE SKIP] '{}' 는 행이 {}개라 미뤄 둔 쌍 {}건이 어느 행의 것인지 단정할 수 없습니다.",
                                    rcat, rows_now, group.len()
                                ));
                                continue;
                            }
                            for d in group.iter() {
                                let (label, value, field, _, own, bbox) = *d;
                                let occupied = final_data_map
                                    .get(rcat.as_str())
                                    .and_then(|v| v.as_array())
                                    .and_then(|a| a.first())
                                    .and_then(|r| r.get(field.as_str()))
                                    .map(|v| match v {
                                        Value::Null => false,
                                        Value::String(s) => !s.trim().is_empty(),
                                        _ => true,
                                    })
                                    .unwrap_or(false);
                                if occupied {
                                    continue;
                                }
                                let near = grounding_claims
                                    .iter()
                                    .any(|g| g.category == *rcat && overlaps(g.bbox, *bbox));
                                if !near {
                                    crate::utils::score_dynamics::record_baseline("vision.array_owner_resolved", 0.0);
                                    emit_term(&format!(
                                        "  ⚪ [ARRAY OWNER RESOLVE SKIP] \"{}\" → {}.{} = \"{}\" | '{}' 의 유일한 행을 만든 지면과 이 쌍을 읽은 크롭이 겹치지 않습니다. 다른 당사자의 사실일 수 있어 채우지 않습니다.",
                                        label, rcat, field, value, rcat
                                    ));
                                    continue;
                                }
                                let claimed = collect_claimed(&final_data_map);
                                if let Some((owner, _)) = claimed
                                    .iter()
                                    .find(|(k, v)| k != field && v.eq_ignore_ascii_case(value))
                                {
                                    emit_term(&format!(
                                        "  🚫 [ARRAY OWNER RESOLVE / CLAIMED] \"{}\" → {}.{} = \"{}\" | 이미 '{}' 가 확정한 값입니다.",
                                        label, rcat, field, value, owner
                                    ));
                                    continue;
                                }
                                if !crate::model::merge::write_into_single_row(&mut final_data_map, rcat, field, value) {
                                    continue;
                                }
                                let mut p = serde_json::Map::new();
                                p.insert(field.clone(), json!(value.clone()));
                                record_grounding_claims(&mut grounding_claims, rcat, &Value::Object(p), *bbox);
                                pair_evidence.insert(field.clone(), *own);
                                pair_label.insert(field.clone(), label.clone());
                                crate::utils::score_dynamics::record_field_seen(field);
                                crate::utils::score_dynamics::record_field_assigned(field, *own);
                                crate::utils::score_dynamics::record_baseline("vision.array_owner_resolved", 1.0);
                                emit_term(&format!(
                                    "  ✅ [ARRAY OWNER RESOLVE] \"{}\" → {}.{} = \"{}\" | 크롭 루프에서는 '{}' 의 행이 없어 미뤄 두었는데, 복구가 끝난 지금 행이 정확히 하나이고 그 행을 만든 지면이 이 쌍의 크롭과 겹칩니다. 같은 당사자의 사실이므로 그 행에 채웁니다 (중립점수 {:+.4}).",
                                    label, rcat, field, value, rcat, own
                                ));
                            }
                        }
                    }

                    {
                        let text_of = |v: &Value| -> String {
                            match v {
                                Value::String(s) => s.trim().to_string(),
                                Value::Number(n) => n.to_string(),
                                _ => String::new(),
                            }
                        };
                        let scalar_text_of = |m: &serde_json::Map<String, Value>, f: &str| -> String {
                            let c = crate::logic::trade_field_category(f);
                            let in_cat = m
                                .get(c)
                                .and_then(|o| o.get(f))
                                .map(|v| text_of(v))
                                .unwrap_or_default();
                            if !in_cat.is_empty() {
                                return in_cat;
                            }
                            m.get(f).map(|v| text_of(v)).unwrap_or_default()
                        };
                        let tax_text = scalar_text_of(&final_data_map, "amount_tax");
                        let tax = crate::model::merge::printed_number(&tax_text);
                        let mut row_sum = 0.0f64;
                        let mut row_n = 0usize;
                        for key in ["items", "line_items"] {
                            if let Some(arr) = final_data_map.get(key).and_then(|v| v.as_array()) {
                                let mut s = 0.0f64;
                                let mut c = 0usize;
                                for r in arr.iter() {
                                    let t = r.get("total_price").map(|v| text_of(v)).unwrap_or_default();
                                    if let Some(x) = crate::model::merge::printed_number(&t) {
                                        s += x;
                                        c += 1;
                                    }
                                }
                                if c > row_n {
                                    row_sum = s;
                                    row_n = c;
                                }
                            }
                        }
                        let base = [
                            if row_n > 0 { Some(row_sum) } else { None },
                            crate::model::merge::printed_number(&scalar_text_of(&final_data_map, "amount_subtotal")),
                            crate::model::merge::printed_number(&scalar_text_of(&final_data_map, "amount")),
                            crate::model::merge::printed_number(&scalar_text_of(&final_data_map, "grand_total_amount")),
                        ]
                        .into_iter()
                        .flatten()
                        .filter(|v| *v > 0.0)
                        .fold(None, |acc: Option<f64>, v| Some(acc.map_or(v, |a| a.max(v))));
                        let label = pair_label.get("amount_tax").cloned().unwrap_or_default();
                        if let (Some(t), Some(b)) = (tax, base) {
                            if b > 0.0 && t > b && !label.is_empty() {
                                let emb = self.get_embedding(label.clone()).await.unwrap_or_default();
                                let ranking = crate::model::merge::label_axis_scores(&emb, &gate_banks);
                                let target = ranking
                                    .iter()
                                    .find(|(f, z)| {
                                        *z > 0.0
                                            && f.as_str() != "amount_tax"
                                            && schema_fields.iter().any(|s| s == f)
                                            && crate::model::merge::row_axis_aggregate_related("amount_tax", f)
                                            && !crate::logic::is_trade_array_category(crate::logic::trade_field_category(f))
                                            && crate::utils::ai_utils::query_value_format(f)
                                                != crate::utils::ai_utils::FieldFormat::Numeric
                                            && crate::utils::ai_utils::value_matches_format(
                                                crate::utils::ai_utils::query_value_format(f),
                                                &tax_text,
                                            )
                                            && scalar_text_of(&final_data_map, f).is_empty()
                                    })
                                    .cloned();
                                crate::utils::score_dynamics::record_baseline(
                                    "vision.money_id_reroute",
                                    if target.is_some() { 1.0 } else { 0.0 },
                                );
                                match target {
                                    Some((field, z)) => {
                                        let tax_cat = crate::logic::trade_field_category("amount_tax").to_string();
                                        let tcat = crate::logic::trade_field_category(&field).to_string();
                                        final_data_map.remove("amount_tax");
                                        if let Some(o) = final_data_map.get_mut(&tax_cat).and_then(|v| v.as_object_mut()) {
                                            o.remove("amount_tax");
                                        }
                                        let mut p = serde_json::Map::new();
                                        p.insert(field.clone(), json!(tax_text.clone()));
                                        merge_extracted(&mut final_data_map, &tcat, &Value::Object(p), &emit_term);
                                        for g in grounding_claims.iter_mut() {
                                            if g.field == "amount_tax" && g.value.trim() == tax_text {
                                                g.field = field.clone();
                                                g.category = tcat.clone();
                                            }
                                        }
                                        pair_label.remove("amount_tax");
                                        pair_label.insert(field.clone(), label.clone());
                                        pair_evidence.remove("amount_tax");
                                        pair_evidence.insert(field.clone(), z);
                                        crate::utils::score_dynamics::record_field_seen(&field);
                                        crate::utils::score_dynamics::record_field_assigned(&field, z);
                                        emit_term(&format!(
                                            "  💱 [MONEY → ID REROUTE] 'amount_tax' = \"{}\" 가 이 문서의 가장 큰 금액 축(행 합계·소계·총액) {} 보다도 큽니다. 세액은 어느 총계보다도 클 수 없으므로 이 자리에 인쇄된 것은 금액이 아니라 등록번호입니다. 라벨 \"{}\" 를 스키마 전체와 다시 재어, 세액과 이름 토큰을 공유하는 비금액 축 가운데 비어 있고 값 형식이 맞는 1위 {}.{} (중립점수 {:+.4}) 로 옮깁니다. 금액 정합 단계가 지우기 전에 옮기므로 번호가 저장본에 남습니다.",
                                            tax_text, b, label, tcat, field, z
                                        ));
                                    }
                                    None => {
                                        emit_term(&format!(
                                            "  💱 [MONEY → ID REROUTE / NO AXIS] 'amount_tax' = \"{}\" 가 이 문서의 가장 큰 금액 축 {} 보다도 크지만, 라벨 \"{}\" 의 순위에서 세액과 이름 토큰을 공유하면서 비어 있고 형식이 맞는 비금액 축을 찾지 못했습니다. 금액 정합 단계의 폐기 판정에 맡깁니다.",
                                            tax_text, b, label
                                        ));
                                    }
                                }
                            }
                        }
                    }
                    extracted_data = Value::Object(final_data_map);
                    if let Some(m) = extracted_data.as_object_mut() {
                        let n = self
                            .remap_off_schema_axes(
                                m, &mut grounding_claims, &detected_type, &language, &emit_term,
                            )
                            .await;
                        if n > 0 {
                            emit_term(&format!(
                                "  ✅ [SCHEMA AXIS MAP] 스키마 밖 키 {}건을 같은 개념의 스키마 축으로 옮겼습니다. 접지 주장의 필드명도 함께 갱신했으므로 STEP 6 의 폐기 판정이 어긋나지 않습니다.",
                                n
                            ));
                        }
                    }
                    crate::models::siglip2::layout_cache::LAYOUT_CACHE.commit(
                        layout_hit.as_ref(),
                        layout_fresh.as_ref(),
                        &grid,
                        &legibility,
                        true,
                        &detected_type,
                        Some(&verdict),
                        &extracted_data,
                        &emit_term,
                    );
                }

            } else {
                // ============================================================
                // 🛒 [Commerce 모드] SigLIP2 히트맵 + 정밀 크롭
                // ============================================================
                emit_term("[STAGE-2] 🛒 Commerce Mode: SigLIP2 Heatmap Pipeline...");
                let commerce_page_type = "goods";
                // 🌟 [SDS SCOPE] 커머스 경로도 1차 키를 확정합니다.
                //    이 줄이 없으면 스코프가 'vision|unknown|' 에 머물러
                //    상품 이미지와 무역 서식의 히트맵 확산도가 한 통계에 섞이고,
                //    V-1 의 확산 게이트(중앙값+MAD) 기준선이 오염됩니다.
                crate::utils::score_dynamics::refine_primary(commerce_page_type);
                // 🌟 [SCOPED LOCK + LAZY TEXT] trade 분기와 동일한 셀프 데드락 방지 구조를
                //    with_siglip_text 가 그대로 제공하며, 캐시 미스가 없으면 인코더를 올리지 않습니다.
                let reused_heatmaps = layout_hit
                    .as_ref()
                    .filter(|h| h.reuse())
                    .and_then(|h| h.heatmaps_for(commerce_page_type));
                let mut heatmaps = match reused_heatmaps {
                    Some(hm) => {
                        emit_term(&format!(
                            "  🗂️ [LAYOUT CACHE / HEATMAP REUSE] 신뢰 템플릿의 커머스 열 히트맵 {}개를 그대로 씁니다. 아레나와 크롭 계획은 이 문서의 판독 지도로 다시 계산합니다.",
                            hm.len()
                        ));
                        hm
                    }
                    None => {
                        let hm = self
                            .with_siglip_text("column heatmaps (commerce)", |m| {
                                crate::models::siglip2::vision_encoder::build_column_heatmaps(
                                    m, &grid, commerce_page_type, &language, Some(&legibility), &[], &emit_term
                                )
                            })
                            .await
                            .map_err(|e| anyhow::anyhow!("Commerce heatmap failed: {}", e))?;
                        let snap = crate::models::siglip2::layout_cache::snapshot(&hm);
                        if let Some(h) = layout_hit.as_ref() {
                            crate::models::siglip2::layout_cache::LAYOUT_CACHE.observe(h, commerce_page_type, &snap, &emit_term);
                        }
                        layout_fresh = Some(snap);
                        hm
                    }
                };

                {
                    let mut protect: Vec<&str> =
                        crate::logic::TRADE_ARRAY_CATEGORIES.to_vec();
                    protect.push(crate::logic::TRADE_IDENTITY_CATEGORY);
                    let arena = crate::models::siglip2::nms_arena::run_arena(
                        &heatmaps, &grid, &legibility, &protect, &emit_term,
                    );
                    crate::utils::score_dynamics::record_baseline(
                        "vision.arena_rounds",
                        arena.rounds as f32,
                    );
                    crate::models::siglip2::nms_arena::apply_arena(
                        &mut heatmaps, &arena, &emit_term,
                    );
                }

                let commerce_height_baseline =
                    crate::models::siglip2::vision_crop::measure_doc_text_height(
                        &dynamic_image, &emit_term,
                    );
                let plans = crate::models::siglip2::vision_crop::plan_crops(
                    &heatmaps,
                    &grid,
                    &legibility,
                    crate::logic::TRADE_ARRAY_CATEGORIES,
                    crate::logic::TRADE_IDENTITY_CATEGORY,
                    crate::logic::TRADE_IDENTITY_FIELD,
                    &emit_term,
                );

                // 🌟 [VRAM STAGE] 커머스 경로도 여기서 SigLIP2 임무가 끝납니다.
                //    아래 두 분기(폴백 단일 호출 / 크롭 루프) 모두 Qwen3.5 를 올리므로
                //    분기 이전에 반환해야 두 경로가 동일한 VRAM 여유를 갖습니다.
                self.release_siglip2("commerce STEP 1~4 complete, before Qwen3.5").await;

                if plans.is_empty() {
                    // 히트맵 실패 → 기존 단일 호출 폴백
                    emit_term("  🛟 [FALLBACK] 크롭 영역 없음. 전체 화면 단일 호출로 전환.");
                    let prompt = crate::parsing::get_image_extraction_prompt("kr", &language, "tracking", "");
                    let (_track_bias, track_prej) = crate::parsing::get_vision_tracking_bias(&language);
                    let result_str = self.chat_with_qwen3_5_image_spinner(
                        "You are a precise commerce and logistics extraction assistant.", &prompt, Some(dynamic_image.clone()), app_handle, "extraction-progress",
                        json!({ "category": "Vision Analysis", "summary": "Analyzing commerce tracking/goods..." }), 1024, cancel_token.clone(), Some(task_id.clone()), Some(&track_prej)
                    ).await?;
                    extracted_data = crate::parsing::parse_json_from_llm(&result_str);
                    record_grounding_claims(
                        &mut grounding_claims,
                        "goods",
                        &extracted_data,
                        (0, 0, grid.orig_width, grid.orig_height),
                    );
                } else {
                    emit_term(&format!("[STAGE-5] 🤖 커머스 크롭 {}개 정제 추출", plans.len()));
                    let mut merged = serde_json::Map::new();
                    let all_fields = crate::parsing::get_detail_schema_fields(commerce_page_type, "", &language);

                    for (idx, plan) in plans.iter().enumerate() {
                        if cancel_token.as_ref().map_or(false, |t| t.load(std::sync::atomic::Ordering::Relaxed)) {
                            return Ok(());
                        }

                        let fields: Vec<(String, String)> = all_fields.iter()
                            .filter(|(name, _, _, _)| {
                                crate::logic::trade_field_category(name) == plan.category
                            })
                            .map(|(name, desc, _, _)| (name.clone(), desc.clone()))
                            .collect();

                        if fields.is_empty() { continue; }

                        let (lg_cnt, il_cnt, bl_cnt) =
                            legibility.count_in_bbox(plan.bbox, grid.orig_width, grid.orig_height);
                        crate::utils::score_dynamics::record_baseline(
                            "vision.crop_legible_patches",
                            lg_cnt as f32,
                        );
                        if lg_cnt == 0 {
                            emit_term(&format!(
                                "    🚫 [EMPTY CROP SKIP] '{}' 는 판독 가능 패치가 0개입니다 (판독불가 {} / 여백 {}). Qwen 호출을 생략합니다.",
                                plan.category, il_cnt, bl_cnt
                            ));
                            crate::utils::score_dynamics::record_baseline("vision.empty_crop_skip", 1.0);
                            continue;
                        }
                        crate::utils::score_dynamics::record_baseline("vision.empty_crop_skip", 0.0);

                        let crop = crate::models::siglip2::vision_crop::crop_region_clamped(
                            &dynamic_image, plan, 512, commerce_height_baseline, &emit_term,
                        );

                        emit_term(&format!(
                            "    📤 [{}] {}x{} 크롭 전송 ({}개 필드)",
                            plan.category, crop.width(), crop.height(), fields.len()
                        ));

                        // 🌟 [ALREADY CLAIMED] 커머스도 동일. 가격과 배송비가 섞이는 사고를 막습니다.
                        let claimed = collect_claimed(&merged);

                        let prompt = crate::parsing::get_commerce_crop_prompt(
                            commerce_page_type,
                            &fields,
                            &language,
                            &plan.top_field,
                            plan.score,
                            &claimed,
                        );

                        let res = self.chat_with_qwen3_5_image_spinner(
                            "You are a precise commerce extraction assistant.",
                            &prompt,
                            Some(crop),
                            app_handle,
                            "extraction-progress",
                            json!({ "category": format!("Commerce Crop {}/{}", idx + 1, plans.len()), "summary": format!("Extracting {}...", plan.category) }),
                            1024,
                            cancel_token.clone(),
                            Some(task_id.clone()),
                            None
                        ).await?;

                        let parsed = crate::parsing::parse_json_from_llm(&res);
                        record_claim_violations(
                            &claimed,
                            &parsed,
                            &plan.category,
                            &emit_term,
                        );
                        record_grounding_claims(
                            &mut grounding_claims,
                            &plan.category,
                            &parsed,
                            plan.bbox,
                        );
                        if let Some(v) = parsed.as_object() {
                            merge_extracted(&mut merged, &plan.category, &Value::Object(v.clone()), &emit_term);
                        }
                    }
                    extracted_data = Value::Object(merged);
                    crate::models::siglip2::layout_cache::LAYOUT_CACHE.commit(
                        layout_hit.as_ref(),
                        layout_fresh.as_ref(),
                        &grid,
                        &legibility,
                        false,
                        commerce_page_type,
                        cached_verdict.as_ref(),
                        &extracted_data,
                        &emit_term,
                    );
                }
            }

            if let Some(m) = extracted_data.as_object() {
                let dropped = crate::models::siglip2::value_grounding::retain_merged_claims(
                    &mut grounding_claims,
                    m,
                );
                if !dropped.is_empty() {
                    emit_term(&format!(
                        "  🧹 [CLAIM PRUNE / STAGE-6] 저장본에 없는 주장 {}건을 접지 검증 전에 뺍니다: {:?} — 저장되지 않은 값을 검증하면 폐기 판정이 아무 데도 적용되지 않은 채 SDS 접지 분포만 오염되고, 같은 값을 두 축이 나눠 가진 것처럼 보여 중복 소유 경쟁이 헛돌았습니다.",
                        dropped.len(),
                        dropped
                            .iter()
                            .map(|c| format!("{}.{}=\"{}\"", c.category, c.field, c.value))
                            .take(8)
                            .collect::<Vec<_>>()
                    ));
                }
            }
            if !grounding_claims.is_empty() {
                emit_term(&format!(
                    "[STAGE-6] 🔬 추출값 {}건 접지 검증 (SigLIP2 텍스트 ↔ 이미지 패치)",
                    grounding_claims.len()
                ));

                let mut verdicts = crate::models::siglip2::value_grounding::verify_claims_v2(
                    &grounding_claims,
                    grid.grid_rows,
                    grid.grid_cols,
                    grid.orig_width,
                    grid.orig_height,
                    &legibility,
                    &language,
                    &emit_term,
                );

                {
                    let survivors: Vec<crate::models::siglip2::value_grounding::GroundingClaim> =
                        grounding_claims
                            .iter()
                            .filter(|c| {
                                !verdicts.iter().any(|v| {
                                    !v.accepted && v.field == c.field && v.value.trim() == c.value.trim()
                                })
                            })
                            .cloned()
                            .collect();
                    let v1_history =
                        crate::utils::score_dynamics::adaptive_baseline("vision.grounding_v1_doc_reject");
                    let v1_claims = crate::utils::score_dynamics::adaptive_baseline_n("vision.grounding_v1_reject")
                        .filter(|(m, sd, n)| *n >= 2 && *m - 1.96 * *sd / (*n as f32).sqrt() > 0.5);
                    if survivors.is_empty() {
                        emit_term("  ⚪ [VALUE GROUNDING v1 SKIP] v2 를 통과한 주장이 없어 패치 코사인 관측을 건너뜁니다.");
                    } else if let Some((reject_mean, reject_sd)) = v1_history.filter(|(m, _)| *m > 0.5) {
                        crate::utils::score_dynamics::record_baseline("vision.grounding_v1_retired", 1.0);
                        emit_term(&format!(
                            "  ⏭️ [VALUE GROUNDING v1 RETIRED] 이 서식 스코프에서 v1 관측을 거친 문서들의 문서별 폐기 비율이 평균 {:.3} (σ {:.3}) 입니다. 환각은 소수여야 하는데 문서마다 과반을 폐기하는 게이트는 환각이 아니라 짧은 값 문자열 자체를 거르고 있다는 뜻이므로, 관측을 더 쌓아도 게이트로 켤 근거가 생기지 않습니다. Qwen3.5 반환 → SigLIP2 텍스트 인코더 부착 → 임베딩 재적재 비용을 이번 문서부터 지불하지 않습니다.",
                            reject_mean, reject_sd
                        ));
                    } else if let Some((claim_mean, claim_sd, claim_n)) = v1_claims {
                        crate::utils::score_dynamics::record_baseline("vision.grounding_v1_retired", 1.0);
                        emit_term(&format!(
                            "  ⏭️ [VALUE GROUNDING v1 RETIRED / CLAIMS] v2 로 접지가 확인된 주장 {}건 가운데 v1 이 버리자고 한 비율이 평균 {:.3} (σ {:.3}) 이고 95% 하한 {:.3} 도 과반입니다. 문서 단위 표본(vision.grounding_v1_doc_reject)은 아직 모자라지만 주장 단위로는 v1 이 환각이 아니라 인쇄된 짧은 값 자체를 거른다는 결론이 이미 섭니다. SigLIP2 텍스트 인코더 부착과 임베딩 재적재 비용을 이번 문서부터 지불하지 않습니다.",
                            claim_n, claim_mean, claim_sd, claim_mean - 1.96 * claim_sd / (claim_n as f32).sqrt()
                        ));
                    } else {
                        emit_term(&format!(
                            "  🔬 [VALUE GROUNDING v1 / OBSERVE] v2 를 통과한 {}건을 패치 코사인으로 한 번 더 관측합니다. v2 는 출처 사각형 안에 글자가 있는지만 세므로 그 자리에 인쇄되지 않은 문자열도 통과합니다. 이번 회차는 관측만 하고 값을 폐기하지 않습니다. (Qwen3.5 반환 + SigLIP2 텍스트 인코더 1회 부착 비용이 듭니다)",
                            survivors.len()
                        ));
                        self.deep_purge_resources().await;
                        let ready = self.ensure_siglip2_ext(false, true).await;
                        let probe = match ready {
                            Err(e) => Err(e),
                            Ok(_) => {
                                self.with_siglip_text("value grounding v1 (stage 6)", |m| {
                                    Ok(crate::models::siglip2::value_grounding::verify_claims(
                                        &survivors,
                                        &grid.patches,
                                        grid.grid_rows,
                                        grid.grid_cols,
                                        grid.orig_width,
                                        grid.orig_height,
                                        &legibility,
                                        |t| {
                                            crate::models::siglip2::vision_encoder::encode_phrases_ephemeral(
                                                m,
                                                &[t.to_string()],
                                            )
                                            .ok()
                                            .and_then(|v| v.into_iter().next())
                                            .unwrap_or_default()
                                        },
                                        &|_: &str| {},
                                    ))
                                })
                                .await
                            }
                        };
                        self.release_siglip2("value grounding v1 observe complete").await;
                        match probe {
                            Ok(list) => {
                                let mut would_drop: Vec<String> = Vec::new();
                                let mut held = 0usize;
                                for v in list.iter() {
                                    if v.gate == crate::models::siglip2::value_grounding::VerdictGate::Held {
                                        held += 1;
                                        continue;
                                    }
                                    crate::utils::score_dynamics::record_baseline(
                                        "vision.grounding_v1_in",
                                        v.surprisal_in,
                                    );
                                    crate::utils::score_dynamics::record_baseline(
                                        "vision.grounding_v1_reject",
                                        if v.accepted { 0.0 } else { 1.0 },
                                    );
                                    if !v.accepted {
                                        would_drop.push(format!(
                                            "{}.{}=\"{}\" (in {:+.4} / out {:+.4} / {})",
                                            v.category, v.field, v.value, v.surprisal_in, v.surprisal_out, v.reason
                                        ));
                                    }
                                }
                                let judged = list.len().saturating_sub(held);
                                if judged > 0 {
                                    crate::utils::score_dynamics::record_baseline(
                                        "vision.grounding_v1_doc_reject",
                                        would_drop.len() as f32 / judged as f32,
                                    );
                                }
                                if would_drop.is_empty() {
                                    emit_term(&format!(
                                        "  ✅ [VALUE GROUNDING v1 / OBSERVE] 판정 {}건(보류 {}건) 전부 패치 코사인으로도 접지되었습니다. 이 문서에서는 폐기 게이트를 켜도 잃는 값이 없습니다.",
                                        list.len().saturating_sub(held), held
                                    ));
                                } else {
                                    emit_term(&format!(
                                        "  👁️ [VALUE GROUNDING v1 / OBSERVE] 폐기 게이트를 켰다면 {}건이 사라졌을 것입니다: {:?} — 값을 실제로 버리기 전에 이 목록이 환각만 담고 있는지 사람이 확인해야 합니다. SigLIP2 가 짧은 고유명사를 패치와 대조하는 능력은 이 코드베이스에서 측정된 적이 없습니다.",
                                        would_drop.len(),
                                        would_drop.iter().take(8).collect::<Vec<_>>()
                                    ));
                                }
                            }
                            Err(e) => emit_term(&format!(
                                "  ⚪ [VALUE GROUNDING v1 SKIP] SigLIP2 텍스트 인코더를 올리지 못해 관측을 건너뜁니다: {}",
                                e
                            )),
                        }
                    }
                }

                let dup_groups = crate::model::merge::cross_field_duplicate_groups(&grounding_claims, &verdicts);
                if !dup_groups.is_empty() {
                    let bank_type = if is_trade_doc { "shipping_doc" } else { "goods" };
                    let mut texts: Vec<String> = Vec::new();
                    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                    for (value, owners) in dup_groups.iter() {
                        if seen.insert(value.clone()) {
                            texts.push(value.clone());
                        }
                        for (_, field, _) in owners.iter() {
                            let (phrases, _) = crate::model::merge::owner_label_bank(&language, bank_type, field);
                            for p in phrases {
                                if seen.insert(p.clone()) {
                                    texts.push(p);
                                }
                            }
                        }
                    }
                    let mut embs: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
                    for part in texts.chunks(200) {
                        let e = self
                            .get_embedding_batch(part.to_vec())
                            .await
                            .unwrap_or_else(|_| vec![Vec::new(); part.len()]);
                        embs.extend(e);
                    }
                    let lookup: std::collections::HashMap<String, Vec<f32>> =
                        texts.into_iter().zip(embs.into_iter()).collect();
                    let owner_verdicts = crate::model::merge::resolve_cross_field_duplicates(
                        &dup_groups, &lookup, &language, bank_type, &emit_term,
                    );
                    verdicts.extend(owner_verdicts);
                }

                {
                    const FREE_TEXT_AXES: [&str; 2] = ["special_instructions", "marks_numbers"];
                    let mut cands: Vec<(String, String, String)> = Vec::new();
                    if let Some(root) = extracted_data.as_object() {
                        for (cat, node) in root.iter() {
                            let obj = match node.as_object() { Some(o) => o, None => continue };
                            for f in FREE_TEXT_AXES.iter() {
                                let s = match obj.get(*f).and_then(|v| v.as_str()) {
                                    Some(s) => s.trim(),
                                    None => continue,
                                };
                                if s.chars().count() < 24 && s.split_whitespace().count() < 6 {
                                    continue;
                                }
                                cands.push((cat.clone(), (*f).to_string(), s.to_string()));
                            }
                        }
                    }
                    if !cands.is_empty() {
                        let decl = crate::logic::anchor_phrases(
                            crate::logic::DECLARATION_BOILERPLATE_ANCHOR,
                            crate::logic::DECLARATION_BOILERPLATE_ANCHOR_ML,
                        );
                        let instr = crate::logic::anchor_phrases(
                            crate::logic::HANDLING_INSTRUCTION_ANCHOR,
                            crate::logic::HANDLING_INSTRUCTION_ANCHOR_ML,
                        );
                        let n_val = cands.len();
                        let n_decl = decl.len();
                        let mut texts: Vec<String> = cands.iter().map(|(_, _, v)| v.clone()).collect();
                        texts.extend(decl);
                        texts.extend(instr);
                        let mut embs: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
                        for part in texts.chunks(200) {
                            let e = self
                                .get_embedding_batch(part.to_vec())
                                .await
                                .unwrap_or_else(|_| vec![Vec::new(); part.len()]);
                            embs.extend(e);
                        }
                        let decl_bank: Vec<Vec<f32>> = embs[n_val..n_val + n_decl]
                            .iter()
                            .filter(|e| !e.is_empty())
                            .cloned()
                            .collect();
                        let instr_bank: Vec<Vec<f32>> = embs[n_val + n_decl..]
                            .iter()
                            .filter(|e| !e.is_empty())
                            .cloned()
                            .collect();
                        if decl_bank.is_empty() || instr_bank.is_empty() {
                            emit_term("  ⚪ [DECLARATION GATE SKIP] 서식 전문 또는 화물 지시문 앵커를 임베딩하지 못했습니다. 한쪽 뱅크만으로 판정하려면 절대 임계가 필요해지므로 판정하지 않습니다.");
                        } else {
                            let instr_coh = crate::utils::ai_utils::bank_internal_cohesion(&instr_bank);
                            let decl_coh = crate::utils::ai_utils::bank_internal_cohesion(&decl_bank);
                            for (i, (cat, field, value)) in cands.iter().enumerate() {
                                let q = match embs.get(i) {
                                    Some(q) if !q.is_empty() => q,
                                    _ => continue,
                                };
                                let own = crate::utils::ai_utils::max_pool_sim(q, &instr_bank);
                                let prej = crate::utils::ai_utils::max_pool_sim(q, &decl_bank);
                                crate::utils::score_dynamics::record_baseline("vision.declaration_gap", prej - own);
                                if prej > own {
                                    emit_term(&format!(
                                        "    📜 [DECLARATION BOILERPLATE] {}.{} = \"{}\" | 서식 전문 {:.4} 가 화물 지시문 {:.4} 를 앞섭니다 (뱅크 결속 전문 {:.3} / 지시문 {:.3}). 수출자 확인 문구는 같은 서식이면 모든 문서에 똑같이 인쇄되므로, 이 축에 들어가는 순간 그 축의 변별력이 0 이 되고 자연어 변환을 타고 청크까지 색인되어 실제 지시문을 밀어냅니다. 폐기합니다.",
                                        cat, field, value, prej, own, decl_coh, instr_coh
                                    ));
                                    crate::utils::score_dynamics::record_baseline("vision.declaration_boilerplate", 1.0);
                                    verdicts.push(crate::models::siglip2::value_grounding::GroundingVerdict {
                                        category: cat.clone(),
                                        field: field.clone(),
                                        value: value.clone(),
                                        surprisal_in: 0.0,
                                        surprisal_out: 0.0,
                                        top_patch: 0,
                                        top_legible: true,
                                        accepted: false,
                                        gate: crate::models::siglip2::value_grounding::VerdictGate::Prejudice,
                                        reason: "서식 전문 (수출자 확인 문구)".to_string(),
                                    });
                                } else {
                                    emit_term(&format!(
                                        "    ✅ [DECLARATION GATE KEEP] {}.{} = \"{}\" | 화물 지시문 {:.4} 가 서식 전문 {:.4} 이상입니다. 이 문서에만 있는 지시문으로 보고 유지합니다.",
                                        cat, field, value, own, prej
                                    ));
                                    crate::utils::score_dynamics::record_baseline("vision.declaration_boilerplate", 0.0);
                                }
                            }
                        }
                    }
                }

                if let Some(map) = extracted_data.as_object_mut() {
                    apply_grounding_verdicts(map, &verdicts, &emit_term);
                    if is_trade_doc {
                        crate::model::merge::drop_row_echo_columns(map, &emit_term);
                        crate::model::merge::reconcile_monetary_axes(map, &emit_term);
                        crate::model::merge::reconcile_package_axes(map, &emit_term);
                        crate::model::merge::reconcile_weight_basis(map, &emit_term);
                        let doc_code = map
                            .get("header")
                            .and_then(|h| h.get("doc_type"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        crate::model::merge::reroute_closed_vocab_values(map, &doc_code, &emit_term);
                    }
                } else {
                    emit_term("  ⚪ [GROUNDING APPLY SKIP] 추출 결과가 객체가 아니라 폐기 판정을 적용할 수 없습니다.");
                }
            }

            // 🌟 [VRAM STAGE-FINAL] 비전 벡터 저장 완료.
            let mode_name = if is_trade_doc { "Trade Document" } else { "Commerce" };
            emit_term(&format!("[STAGE-2] Generating vision insights for {} mode...", mode_name));

            emit_term("\n=======================================");
            emit_term(&format!("[DEBUG-VISION] 🤖 AI Raw Response Extracted."));
            emit_term("=======================================\n");

            if is_trade_doc {
                let nested_cur = extracted_data
                    .get("financials")
                    .and_then(|f| f.get("currency"))
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !crate::model::merge::is_schema_echo(s));
                if let (Some(c), Some(obj)) = (nested_cur, extracted_data.as_object_mut()) {
                    let root_empty = obj
                        .get("currency")
                        .and_then(|v| v.as_str())
                        .map_or(true, |s| s.trim().is_empty());
                    if root_empty {
                        obj.insert("currency".to_string(), json!(c));
                    }
                }
                crate::scheduler::trading::normalize_trading_data(&mut extracted_data, &language);
                {
                    let read_axis = |name: &str| -> Option<String> {
                        extracted_data
                            .get(name)
                            .cloned()
                            .or_else(|| {
                                extracted_data.as_object().and_then(|o| {
                                    o.values()
                                        .filter_map(|v| v.as_object())
                                        .find_map(|inner| inner.get(name).cloned())
                                })
                            })
                            .and_then(|v| match v {
                                Value::String(s) if !s.trim().is_empty() => Some(s),
                                Value::Number(n) => Some(n.to_string()),
                                _ => None,
                            })
                    };
                    let mut shown: Vec<String> = Vec::new();
                    let mut non_iso: Vec<String> = Vec::new();
                    for axis in [
                        "issue_date", "expiry_date", "etd", "eta",
                        "departure_date", "arrival_date", "due_date",
                        "latest_shipment_date", "valid_until",
                        "transaction_date", "declaration_date", "clearance_date",
                    ] {
                        let v = match read_axis(axis) { Some(v) => v, None => continue };
                        let iso = v.len() >= 10
                            && v.as_bytes()[4] == b'-'
                            && v.as_bytes()[7] == b'-'
                            && v.chars().take(4).all(|c| c.is_ascii_digit());
                        shown.push(format!("{}=\"{}\"{}", axis, v, if iso { "" } else { " ⚠" }));
                        if !iso { non_iso.push(axis.to_string()); }
                    }
                    if shown.is_empty() {
                        emit_term("  📅 [DATE NORMALIZE] 저장 직전 시점에 날짜 축이 하나도 없습니다. 회수 단계에서 날짜를 얻지 못했다는 뜻이므로, 이 문서에 대한 기간 조건은 어떤 값을 넣어도 통과하지 못합니다.");
                    } else if non_iso.is_empty() {
                        emit_term(&format!(
                            "  📅 [DATE NORMALIZE] 날짜 축 {}개가 모두 ISO 형식입니다: {:?}. 질의의 기간 조건은 ISO 문자열로 비교하므로 이 형식이어야만 만납니다.",
                            shown.len(), shown
                        ));
                    } else {
                        emit_term(&format!(
                            "  ⚠️ [DATE NORMALIZE] 날짜 축 {:?} 가 ISO 형식이 아닙니다 (전체: {:?}). 인쇄 원문이 그대로 남았다는 뜻이며, 이 상태로는 기간 조건이 문자열 비교로 떨어져 영원히 통과하지 못합니다. 정규화가 이 축 이름에 걸리지 않았는지, 루트 승격에서 이름이 바뀌었는지 확인해야 합니다.",
                            non_iso, shown
                        ));
                    }
                    crate::utils::score_dynamics::record_baseline(
                        "vision.date_iso_ratio",
                        if shown.is_empty() {
                            0.0
                        } else {
                            (shown.len() - non_iso.len()) as f32 / shown.len() as f32
                        },
                    );
                }
            }
            if !is_trade_doc {
                let lang_code = crate::utils::bias_schema::lang_code_of(&language);
                crate::scheduler::trading::normalize_trading_data(&mut extracted_data, &lang_code);
                let raw_status = extracted_data
                    .get("status")
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default();
                let status_note = if raw_status.is_empty() {
                    "상태 값 없음".to_string()
                } else if crate::logic::parse_status(&raw_status) != 0 {
                    format!("상태 '{}' 는 이미 캐노니컬 키", raw_status)
                } else {
                    let lower = raw_status.to_lowercase();
                    let canon = if crate::logic::parse_status(&lower) != 0 {
                        Some((lower, "소문자 일치".to_string()))
                    } else {
                        crate::utils::ai_utils::status_canonical_exact(&raw_status, "goods", None)
                    };
                    match canon {
                        Some((key, route)) => {
                            if let Some(o) = extracted_data.as_object_mut() {
                                o.insert("status".to_string(), json!(key.clone()));
                            }
                            format!("상태 '{}' → '{}' ({})", raw_status, key, route)
                        }
                        None => format!("상태 '{}' 는 닫힌 상태 어휘와 정확히 맞지 않아 원문을 둡니다", raw_status),
                    }
                };
                emit_term(&format!(
                    "  🧾 [COMMERCE NORMALIZE / IMAGE] 텍스트 경로와 같은 저장 형식으로 맞춥니다: 날짜 축은 ISO, 금액·수량 축은 수치, 통화는 문서 언어 '{}' 기준 코드 | {} | 질의의 기간·수치·상태 조건이 이미지 문서와도 같은 비교로 만납니다.",
                    lang_code, status_note
                ));
            }
            let nl = crate::parsing::json_to_natural_language(&extracted_data);
            let doc_type_owned: String = if is_trade_doc {
                let raw = extracted_data.get("header")
                    .and_then(|h| h.get("doc_type"))
                    .and_then(|s| s.as_str())
                    .or_else(|| extracted_data.get("doc_type").and_then(|s| s.as_str()))
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .unwrap_or("shipping_doc");
                let code = crate::logic::doc_type_to_code(raw);
                match crate::utils::bias_schema::canonical_trade_doc_code(&code) {
                    Some(c) => c.to_string(),
                    None => raw.to_string(),
                }
            } else {
                "goods".to_string()
            };
            let doc_type: &str = doc_type_owned.as_str();
            let masked_nl = nl.clone(); // 마스킹은 백엔드 push_data 단계에서 동적으로 수행됩니다.

            let item_digest = crate::utils::hash::digest(&nl);

            {
                let mut q35_guard = self.qwen3_5_generator.lock().await;
                if let Some(gen) = q35_guard.as_mut() {
                    if gen.vision_capable() && gen.is_vision_jit_capable() && gen.vision_resident() {
                        let _ = gen.set_vision_active(false);
                        emit_term("[VISION-JIT] Vision pipeline complete. mmproj weights released before embedding stage.");
                    }
                }
            }

            emit_term("[STAGE-3] Syncing extracted data to LanceDB...");

            // 🌟 [CRITICAL FIX 2] 5단계 마무리를 위한 저장 스텝(4단계) UI 추가!
            let payload_save = json!({ "task_id": task_id.clone(), "category": "Saving", "summary": "Syncing to database...", "spinner": "⠋" });
            let _ = app_handle.emit("extraction-progress", &payload_save);
            crate::utils::logger::log_task_progress(app_handle, &task_id, &payload_save);

            let store_guard = store_mutex.lock().await;
            if let Some(db) = store_guard.as_ref() {
                let from_addr = "0x0000000000000000000000000000000000000000";
                let team_id = crate::utils::hash::hash_id(from_addr);
                let hashed_cc = if is_trade_doc {
                    crate::parsing::trading_cc()
                } else {
                    crate::utils::hash::hash_id("local.commerce")
                };

                let auto_identity = if item_digest.is_empty() {
                    task_id.clone()
                } else {
                    format!("AUTO-{}-{}", doc_type, item_digest)
                };
                let raw_no_owned: String = if is_trade_doc {
                    let (resolved_no, _resolved_idx, _is_fallback) =
                        crate::parsing::resolve_trade_doc_identity(&doc_type, &extracted_data, &language);
                    
                    emit_term(&format!(
                        "  🔑 [DOC IDENTITY] resolve_trade_doc_identity 결과: '{}' (폴백: {})",
                        resolved_no, resolved_no.is_empty()
                    ));
                    
                    if !resolved_no.is_empty() {
                        resolved_no
                    } else {
                        // 폴백: header / 루트 직접 탐색 (기존 경로 유지)
                        let from_header = extracted_data.get("header")
                            .and_then(|h| h.get("document_number").or_else(|| h.get("doc_number")))
                            .and_then(|s| s.as_str());
                        let from_root = extracted_data.get("document_number")
                            .or_else(|| extracted_data.get("doc_number"))
                            .or_else(|| extracted_data.get("tracking_number"))
                            .and_then(|s| s.as_str());
                        
                        from_header
                            .or(from_root)
                            .map(|s| s.trim().to_string())
                            .filter(|s| !crate::model::merge::is_schema_echo(s))
                            .unwrap_or_else(|| auto_identity.clone())
                    }
                } else {
                    extracted_data.get("tracking_number")
                        .and_then(|s| s.as_str())
                        .map(|s| s.trim().to_string())
                        .filter(|s| !crate::model::merge::is_schema_echo(s))
                        .unwrap_or_else(|| auto_identity.clone())
                };
                let raw_no: &str = raw_no_owned.as_str();
                emit_term(&format!("[STAGE-3] 문서 식별자 확정: '{}' (내용 기반 결정론 폴백 여부: {})",
                    raw_no, raw_no == auto_identity.as_str()));

                let table_name = "items"; 
                
                let clean_no = crate::utils::hash::normalize_identifier(raw_no);
                let index_val = crate::scheduler::entity_key_index(doc_type, &team_id, &hashed_cc, raw_no);
                let hashed_id = crate::scheduler::entity_id(&team_id, index_val);
                let mut ref_val = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, hashed_cc, clean_no));
                emit_term(&format!(
                    "  🔑 [VISION IDENTITY] {} '{}' → index {} · id '{}' | 텍스트 경로(trading.rs)·커머스 목록·상세와 같은 entity_key_index 식입니다. 같은 서식이 이미지와 텍스트 두 경로로 들어와도 한 행으로 합쳐지고, 다른 서식의 rel_ 축이 같은 index 로 착지합니다.",
                    doc_type, raw_no, index_val, hashed_id
                ));

                let mut final_data = if extracted_data.is_object() { extracted_data.clone() } else { json!({ "raw_output": extracted_data }) };
                final_data.as_object_mut().unwrap().insert("index".to_string(), json!(index_val));
                final_data.as_object_mut().unwrap().insert("id".to_string(), json!(hashed_id));
                // 🌟 [CRITICAL FIX] 이미지 추출 결과에도 모드 필터를 위한 mode 값을 명시적으로 주입합니다.
                // 🌟 [MODE PARITY] MODE REROUTE(commerce→trading) 가 발화하면 저장 모드도
                //    실제로 실행된 파이프라인과 일치해야 합니다. search_mode 원본("commerce")을
                //    그대로 저장하면 문서는 commerce 목록에만 박히고, trading 목록의
                //    `mode = 'shipping'` 필터에서는 영원히 0건이라 UI 에 아무것도 안 나옵니다.
                //    hashed_cc 가 이미 is_trade_doc 을 쓰는 것과 동일한 규칙으로 통일합니다.
                //    · shipping 태스크            → is_trade_doc=true  → "shipping" (동작 불변)
                //    · commerce + 무역 서식 감지  → is_trade_doc=true  → "shipping" (리라우트 일치)
                //    · commerce + 상품/택배 라벨  → is_trade_doc=false → "commerce" (동작 불변)
                final_data.as_object_mut().unwrap().insert(
                    "mode".to_string(),
                    json!(if is_trade_doc { "shipping" } else { "commerce" }),
                );
                final_data.as_object_mut().unwrap().insert("text".to_string(), json!(nl));
                final_data.as_object_mut().unwrap().insert("masked_text".to_string(), json!(masked_nl));

                if is_trade_doc {
                    // 잎을 끌어올릴 중첩 그룹. 배열(line_items/containers)은 아래에서 따로 처리합니다.
                    const TRADE_GROUPS: [&str; 8] = crate::logic::TRADE_FLATTEN_GROUPS;

                    // bias.json 의 path_alias 를 역방향(alias -> canonical)으로 사용합니다.
                    // build_dexie_plan 은 canonical 로 조건을 모으므로,
                    // 저장 시점에도 canonical 이름으로 올려야 두 방향이 만납니다.
                    fn canonical_name(raw: &str) -> String {
                        crate::utils::bias_schema::canonical_field_name(raw)
                    }

                    let mut hoisted: Vec<String> = Vec::new();

                    for group in TRADE_GROUPS.iter() {
                        let src = match extracted_data.get(*group).and_then(|v| v.as_object()) {
                            Some(o) => o.clone(),
                            None => continue,
                        };
                        let obj = final_data.as_object_mut().unwrap();
                        for (k, v) in src {
                            if v.is_null() { continue; }
                            if let Some(s) = v.as_str() {
                                if crate::model::merge::is_schema_echo(s) { continue; }
                            }
                            let name = canonical_name(&k);
                            // 이미 채워진 축은 덮어쓰지 않습니다. (아래 식별자 블록이 우선)
                            if obj.get(&name).map_or(false, |x| !x.is_null()) { continue; }
                            obj.insert(name.clone(), v.clone());
                            hoisted.push(name);
                        }
                    }

                    // ── 문서 식별자 : no(레거시 commerce 축)와 doc_number(trading 축)를 동시 유지 ──
                    {
                        let obj = final_data.as_object_mut().unwrap();
                        let dnum = obj.get("doc_number").cloned()
                            .or_else(|| obj.get("document_number").cloned())
                            .unwrap_or(json!(""));
                        obj.insert("no".to_string(), dnum.clone());
                        obj.insert("doc_number".to_string(), dnum);
                        if obj.get("doc_type").map_or(true, |v| v.as_str().unwrap_or("").is_empty()) {
                            obj.insert("doc_type".to_string(), json!(doc_type));
                        }
                    }

                    // ── 배열 축 : 첫 원소만 대표 축으로 승격 ──
                    //    (전체 목록은 data.containers / data.items 배열에 그대로 남습니다)
                    for (arr_key, promote) in [
                        ("containers", vec!["container_number", "seal_number"]),
                        ("items", vec!["hs_code"]),
                    ] {
                        let arr = match extracted_data.get(arr_key).and_then(|v| v.as_array()) {
                            Some(a) => a.clone(),
                            None => continue,
                        };
                        let obj = final_data.as_object_mut().unwrap();
                        for field in promote {
                            if obj.get(field).map_or(false, |x| !x.is_null()) { continue; }
                            if let Some(v) = arr.iter().find_map(|it| it.get(field)) {
                                obj.insert(field.to_string(), v.clone());
                                hoisted.push(field.to_string());
                            }
                        }
                    }

                    // 🌟 [LEGACY MIRROR] 기존 소비처가 line_items 를 읽으므로 items 를 그대로 복사합니다.
                    //    generate_rich_summary / merge_json_manual 등 텍스트 경로가
                    //    line_items 키를 전제하고 있어, 키를 통일하면서 그쪽이 끊기지 않게 합니다.
                    //    원본은 items 이고 line_items 는 읽기 전용 사본입니다.
                    {
                        let items_arr = final_data.get("items").cloned()
                            .or_else(|| extracted_data.get("items").cloned());
                        if let Some(v) = items_arr {
                            if v.as_array().map(|a| !a.is_empty()).unwrap_or(false) {
                                final_data.as_object_mut().unwrap()
                                    .insert("line_items".to_string(), v);
                                emit_term("[TRADING FLATTEN v3] 🔁 items 배열을 line_items 로 미러했습니다. (레거시 소비처 호환)");
                            }
                        }
                    }

                    emit_term(&format!(
                        "[TRADING FLATTEN v3] data 루트로 승격한 축 {}개: {:?}",
                        hoisted.len(),
                        hoisted.iter().take(12).collect::<Vec<_>>()
                    ));
                }

                let doc_bcc = crate::scheduler::entity_bcc(doc_type, &hashed_cc);
                let prev_json: Option<Value> = db
                    .get_item_by_id(table_name, &hashed_id)
                    .await
                    .ok()
                    .flatten()
                    .and_then(|d| serde_json::from_str::<Value>(&d.json_data).ok());
                let prev_prior = crate::utils::canonical::ledger_prior(prev_json.as_ref());
                let mut vision_stats: std::collections::HashMap<String, (i64, i64, i64)> = std::collections::HashMap::new();
                let mut vision_linked = 0usize;
                let mut vision_drafted = 0usize;
                if is_trade_doc {
                    let (_, _, trade_ref) = crate::parsing::trading_envelope(&team_id, doc_type, &final_data, &hashed_id);
                    ref_val = trade_ref;
                    relay_plan = crate::parsing::plan_trade_relays(doc_type, &extracted_data, &language);
                    let mut forward_cols: Vec<String> = Vec::new();
                    for (target_type, relay_key) in relay_plan.iter() {
                        if relay_key.search_field != "doc_number" {
                            continue;
                        }
                        let link_value = final_data
                            .get(&relay_key.source_field)
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        if crate::model::merge::is_schema_echo(&link_value) {
                            continue;
                        }
                        let target_index = crate::scheduler::entity_key_index(target_type, &team_id, "", &link_value);
                        let col = crate::logic::trading_index_column(target_type);
                        if let Some(o) = final_data.as_object_mut() {
                            o.insert(col.clone(), json!(target_index));
                        }
                        forward_cols.push(format!("{}={}", col, target_index));
                    }
                    if !forward_cols.is_empty() {
                        emit_term(&format!(
                            "  🔑 [VISION RELAY INDEX] 정방향 연결 축 {:?} 을 저장 전에 새깁니다. trading.rs 가 rel_{{코드}} 에 상대 index 를 넣는 것과 같은 규약이라 검색의 relay_out 이 두 경로 문서를 똑같이 풉니다.",
                            forward_cols
                        ));
                    }
                    let mut carried: Vec<String> = Vec::new();
                    if prev_prior == crate::utils::canonical::LedgerPrior::Placeholder {
                        if let (Some(prev), Some(obj)) = (prev_json.as_ref().and_then(|v| v.as_object()), final_data.as_object_mut()) {
                            for (k, v) in prev.iter() {
                                if !(k.starts_with("rel_") || k.starts_with("reference_")) {
                                    continue;
                                }
                                if crate::utils::canonical::relay_value_is_placeholder(k, v) {
                                    continue;
                                }
                                let blank_here = obj
                                    .get(k)
                                    .map_or(true, |cur| crate::utils::canonical::relay_value_is_placeholder(k, cur));
                                if blank_here {
                                    obj.insert(k.clone(), v.clone());
                                    carried.push(k.clone());
                                }
                            }
                        }
                    }
                    if !carried.is_empty() {
                        crate::utils::score_dynamics::record_baseline("vision.relay_placeholder_carry", carried.len() as f32);
                        emit_term(&format!(
                            "  🧷 [VISION PLACEHOLDER CARRY] '{}' 자리에 다른 서식이 먼저 만든 자리 초안이 있었습니다. 초안이 들고 있던 연결 축 {:?} 을 원본으로 옮긴 뒤 저장합니다. 옮기지 않으면 원본 저장이 초안을 덮어 역방향 연결이 사라집니다.",
                            hashed_id, carried
                        ));
                    }
                    if let Some(o) = final_data.as_object_mut() {
                        o.insert(crate::utils::canonical::LEDGER_KEY.to_string(), json!("count"));
                    }
                    crate::scheduler::relay_ledger::add_delta(
                        &mut vision_stats,
                        doc_type,
                        crate::utils::canonical::ledger_delta(prev_prior, true),
                    );
                }
                if let Some(o) = final_data.as_object_mut() {
                    o.insert(
                        "updated_at".to_string(),
                        json!(chrono::Utc::now().timestamp_millis()),
                    );
                }
                let vision_vec: Option<Vec<f32>> = if grid.pooled.len() == 1152 {
                    Some(grid.pooled.clone())
                } else {
                    None
                };
                let _ = db.upsert_item(
                    table_name,
                    &hashed_id,
                    doc_type,
                    final_data.clone(),
                    None,
                    vision_vec,
                    Some(from_addr),
                    Some(&team_id),
                    Some(&hashed_cc),
                    Some(&doc_bcc),
                    Some(&ref_val),
                    Some(&item_digest)
                ).await;

                {
                    let chunk_cancel = cancel_token
                        .clone()
                        .unwrap_or_else(|| Arc::new(AtomicBool::new(false)));
                    let chunk_bcc = crate::utils::hash::hash_id(&format!("{}{}", doc_type, hashed_cc));
                    let translit_demand = crate::model::lang_llm::translit_demand(&final_data);
                    let (_translit_engine, translit_status) = crate::model::lang_llm::await_engine(
                        &language,
                        app_handle,
                        &task_id,
                        &chunk_cancel,
                        translit_demand,
                    )
                    .await;
                    emit_term(&format!("  {}", translit_status));
                    match crate::scheduler::indexing::index_item_chunks(
                        db,
                        self,
                        &hashed_id,
                        doc_type,
                        &language,
                        &final_data,
                        true,
                        &hashed_cc,
                        &chunk_bcc,
                        &ref_val,
                        if is_trade_doc { "shipping" } else { "commerce" },
                        "",
                        &chunk_cancel,
                        app_handle,
                        &task_id,
                        false,
                    ).await {
                        Ok(n) => emit_term(&format!(
                            "  🧩 [VISION CHUNK INDEX] item_id='{}' | 청크 {}건 인덱싱 완료 (doc_type='{}'). 이 단계가 없으면 문서가 FTS 와 비전 벡터로만 회수되어, 질의의 속성 힌트와 크로스링구얼·음차 트랙이 붙을 자리가 없습니다. 음차 별칭도 이 회차에서 함께 만듭니다 — 영문 서식의 값('T-Shirt')을 한글 질의('티셔츠')가 별칭 트랙으로 맞추려면 이 문서의 별칭 행이 있어야 합니다.",
                            hashed_id, n, doc_type
                        )),
                        Err(e) => emit_term(&format!(
                            "  ⚠️ [VISION CHUNK INDEX] 청크 인덱싱에 실패했습니다: {}. 문서 저장 자체는 끝났으므로 파이프라인은 계속 진행합니다.",
                            e
                        )),
                    }
                }

                let mut relay_starved: Vec<String> = Vec::new();

                // 🌟 relay_plan 을 if is_trade_doc 블록 외부에서 선언하여
                //    블록 내부와 외부 모두에서 접근 가능하게 합니다.
                

                if is_trade_doc {
                    if relay_plan.is_empty() {
                        emit_term("  ⚪ [RELAY v4] 릴레이 키가 확보되지 않아 릴레이를 건너뜁니다.");
                    } else {
                        emit_term(&format!(
                            "  🔗 [RELAY v4] 릴레이 계획 {}건: {:?}",
                            relay_plan.len(),
                            relay_plan.iter().map(|(t, k)| format!("{}←{}('{}')", t, k.role, k.source_field)).collect::<Vec<_>>()
                        ));
                    }
                    for (target_type, relay_key) in &relay_plan {
                        // 🌟 [SEARCH FIELD FIX v5] source_field와 search_field를 분리합니다.
                        //    - source_field: 내 문서에서 값을 가져온 필드 (진단용)
                        //    - search_field: 상대 문서에서 검색할 필드명
                        //    기존에는 둘 다 source_field로 동일하여 자기 자신의 필드에서 검색하여
                        //    항상 SELF-SKIP 되었습니다.
                        let search_field = &relay_key.search_field;
                        let source_field = &relay_key.source_field;
                        let link_value = relay_key.raw.clone();
                        if crate::model::merge::is_schema_echo(&link_value) {
                            continue;
                        }
                        let link_value = final_data.get(source_field)
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        if crate::model::merge::is_schema_echo(&link_value) {
                            relay_starved.push(format!("{}←{}(빈 키)", target_type, source_field));
                            continue;
                        }
                        emit_term(&format!(
                            "  🔗 [TRADE RELAY] {} → {} | {}='{}' 로 연결 검색...",
                            doc_type, target_type, search_field, link_value
                        ));

                        let is_forward = search_field.as_str() == "doc_number" || search_field.as_str() == "no";
                        let mut found_target: Option<(String, Value)> = None;
                        if is_forward {
                            let direct_index = crate::scheduler::entity_key_index(target_type, &team_id, "", &link_value);
                            let direct_id = crate::scheduler::entity_id(&team_id, direct_index);
                            if let Ok(Some(doc)) = db.get_item_by_id("items", &direct_id).await {
                                let direct_code = crate::logic::doc_type_to_code(&doc.r#type);
                                if direct_code != *target_type && doc.r#type != *target_type {
                                    crate::utils::score_dynamics::record_baseline("vision.relay_type_guard", 1.0);
                                    emit_term(&format!(
                                        "  🔀 [TRADE RELAY / TYPE GUARD] {} → {} (index={}) 자리의 문서 '{}' 가 '{}' 타입입니다. index 는 서식 코드를 해시에 포함하므로 이 충돌은 기존 데이터 오염입니다. 연결도 초안 생성도 하지 않습니다.",
                                        doc_type, target_type, direct_index, direct_id, doc.r#type
                                    ));
                                    continue;
                                }
                                if let Ok(parsed) = serde_json::from_str::<Value>(&doc.json_data) {
                                    emit_term(&format!(
                                        "  🔍 [TRADE RELAY / DIRECT ID] {} → {} | 번호 '{}' 로 index {} 를 재현해 id '{}' 를 바로 찾았습니다.",
                                        doc_type, target_type, link_value, direct_index, direct_id
                                    ));
                                    found_target = Some((doc.id.clone(), parsed));
                                }
                            }
                        }
                        if found_target.is_none() {
                            let escaped_link = link_value.replace('\'', "''");
                            let filter = format!(
                                "(data LIKE '%\"{}\":\"{}\"%') OR (data LIKE '%\"{}\":[%\"{}\"%')",
                                search_field, escaped_link, search_field, escaped_link
                            );
                            let relay_search = db.get_all_items("items", 10, 0, Some(filter)).await;
                            match relay_search {
                                Ok(docs) => {
                                    for doc in docs {
                                        if doc.id == hashed_id {
                                            continue;
                                        }
                                        let found_doc_type = doc.r#type.clone();
                                        let found_code = crate::logic::doc_type_to_code(&found_doc_type);
                                        let parsed: Value = match serde_json::from_str(&doc.json_data) {
                                            Ok(v) => v,
                                            Err(_) => continue,
                                        };
                                        let data_doc_type = parsed.get("doc_type")
                                            .and_then(|v| v.as_str())
                                            .unwrap_or("");
                                        let data_code = crate::logic::doc_type_to_code(data_doc_type);
                                        let type_matches = if found_code == *target_type {
                                            true
                                        } else if data_code == *target_type {
                                            true
                                        } else if found_doc_type == *target_type {
                                            true
                                        } else if data_doc_type == *target_type {
                                            true
                                        } else {
                                            false
                                        };
                                        if !type_matches {
                                            continue;
                                        }
                                        let field_ok = match parsed.get(search_field) {
                                            Some(Value::String(s)) => s.trim() == link_value,
                                            Some(Value::Array(a)) => a
                                                .iter()
                                                .any(|x| x.as_str().map_or(false, |s| s.trim() == link_value)),
                                            _ => false,
                                        };
                                        if !field_ok {
                                            continue;
                                        }
                                        found_target = Some((doc.id, parsed));
                                        break;
                                    }
                                },
                                Err(e) => {
                                    emit_term(&format!(
                                        "  ⚠️ [TRADE RELAY v4] {} 검색 실패: {:?}",
                                        target_type, e
                                    ));
                                    continue;
                                }
                            }
                        }
                        match found_target {
                            Some((existing_id, mut ej)) => {
                                let mut needs_update = false;
                                let reverse_field = crate::logic::trade_reference_field_of(doc_type)
                                    .unwrap_or("");
                                if !reverse_field.is_empty() {
                                    let my_doc_number = final_data
                                        .get("doc_number")
                                        .and_then(|v| v.as_str())
                                        .map(|s| s.trim().to_string())
                                        .filter(|s| !crate::model::merge::is_schema_echo(s));
                                    if let Some(my_doc_number) = my_doc_number {
                                        let existing_ref = ej.get(reverse_field).and_then(|v| v.as_str()).unwrap_or("");
                                        if crate::model::merge::is_schema_echo(existing_ref) {
                                            ej.as_object_mut().unwrap().insert(reverse_field.to_string(), json!(my_doc_number));
                                            needs_update = true;
                                        }
                                    }
                                }
                                let relay_col = crate::logic::trading_index_column(doc_type);
                                let their_relay = ej.get(&relay_col).and_then(|v| v.as_u64()).unwrap_or(0);
                                if their_relay == 0 {
                                    ej.as_object_mut().unwrap().insert(relay_col.clone(), json!(index_val));
                                    needs_update = true;
                                }
                                // 물류 정보 상호 보완 (vessel, pol, pod, etd, eta)
                                for field in ["vessel", "voyage_number", "pol", "pod", "etd", "eta"] {
                                    let my_val = extracted_data.get("logistics")
                                        .and_then(|l| l.get(field))
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    if crate::model::merge::is_schema_echo(my_val) { continue; }
                                    let their_val = ej.get("logistics")
                                        .and_then(|l| l.get(field))
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    if crate::model::merge::is_schema_echo(their_val) {
                                        if let Some(logistics_obj) = ej.get_mut("logistics").and_then(|l| l.as_object_mut()) {
                                            logistics_obj.insert(field.to_string(), json!(my_val));
                                            needs_update = true;
                                        }
                                    }
                                }
                                // 화물 정보 상호 보완 (container_number, seal_number)
                                for field in ["container_number", "seal_number"] {
                                    let my_val = extracted_data.get("containers")
                                        .and_then(|c| c.as_array())
                                        .and_then(|arr| arr.first())
                                        .and_then(|c| c.get(field))
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    if crate::model::merge::is_schema_echo(my_val) { continue; }
                                    let their_containers = ej.get("containers").and_then(|c| c.as_array());
                                    let their_has = their_containers.map_or(false, |arr| {
                                        arr.iter().any(|c| c.get(field).and_then(|v| v.as_str()).map_or(false, |v| v == my_val))
                                    });
                                    if !their_has {
                                        if let Some(containers_arr) = ej.get_mut("containers").and_then(|c| c.as_array_mut()) {
                                            if containers_arr.is_empty() {
                                                containers_arr.push(json!({ field: my_val }));
                                            } else if let Some(first) = containers_arr.first_mut() {
                                                if let Some(obj) = first.as_object_mut() {
                                                    if obj.get(field).and_then(|v| v.as_str()).unwrap_or("").is_empty() {
                                                        obj.insert(field.to_string(), json!(my_val));
                                                    }
                                                }
                                            }
                                            needs_update = true;
                                        }
                                    }
                                }
                                vision_linked += 1;
                                let found_prior = crate::utils::canonical::ledger_prior(Some(&ej));
                                if found_prior == crate::utils::canonical::LedgerPrior::Placeholder {
                                    if let Some(obj) = ej.as_object_mut() {
                                        let slot = obj
                                            .entry(crate::utils::canonical::RELAY_ORIGIN_KEY.to_string())
                                            .or_insert_with(|| json!([]));
                                        if !slot.is_array() {
                                            *slot = json!([]);
                                        }
                                        if let Some(arr) = slot.as_array_mut() {
                                            if !arr.iter().any(|x| x.as_str() == Some(doc_type)) {
                                                arr.push(json!(doc_type));
                                                needs_update = true;
                                            }
                                        }
                                    }
                                }
                                if needs_update {
                                    match found_prior {
                                        crate::utils::canonical::LedgerPrior::Placeholder => {}
                                        crate::utils::canonical::LedgerPrior::Draft => {
                                            ej.as_object_mut().unwrap().insert(crate::utils::canonical::LEDGER_KEY.to_string(), json!("count"));
                                            ej.as_object_mut().unwrap().insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                                            crate::scheduler::relay_ledger::add_delta(
                                                &mut vision_stats,
                                                target_type,
                                                crate::utils::canonical::ledger_delta(found_prior, true),
                                            );
                                        }
                                        _ => {
                                            ej.as_object_mut().unwrap().insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                                        }
                                    }
                                    let merged_text = crate::parsing::json_to_natural_language(&ej);
                                    ej.as_object_mut().unwrap().insert("text".to_string(), json!(merged_text));
                                    ej.as_object_mut().unwrap().insert("masked_text".to_string(), json!(merged_text.clone()));
                                    let _ = db.upsert_item(
                                        "items", &existing_id, target_type, ej, None,
                                        None,
                                        Some(from_addr), Some(&team_id), Some(&hashed_cc),
                                        Some(&crate::scheduler::entity_bcc(target_type, &hashed_cc)),
                                        Some(&ref_val), None
                                    ).await;
                                    emit_term(&format!(
                                        "  ✅ [TRADE RELAY v4] 기존 {} 문서 '{}' ({:?}) 에 {} 연결 축 병합 완료.",
                                        target_type, existing_id, found_prior, doc_type
                                    ));
                                }
                            },
                            None => {
                                let search_key = search_field.to_string();
                                if search_key != "doc_number" && search_key != "no" {
                                    crate::utils::score_dynamics::record_baseline("vision.relay_reverse_no_draft", 1.0);
                                    emit_term(&format!(
                                        "  ⚪ [TRADE RELAY v4 / REVERSE NO-DRAFT] {} ← {}='{}' 는 역방향 참조입니다. 이 값은 상대 서식의 문서번호가 아니라 이 문서의 번호라서 상대 자리 초안의 식별자가 될 수 없습니다. 상대 원본이 들어오면 그쪽의 정방향 릴레이가 이 문서를 찾아 연결합니다.",
                                        target_type, search_key, link_value
                                    ));
                                    continue;
                                }
                                let draft_index = crate::scheduler::entity_key_index(target_type, &team_id, "", &link_value);
                                let draft_id = crate::scheduler::entity_id(&team_id, draft_index);
                                if let Ok(Some(occupant)) = db.get_item_by_id("items", &draft_id).await {
                                    crate::utils::score_dynamics::record_baseline("vision.relay_id_occupied", 1.0);
                                    emit_term(&format!(
                                        "  ⚪ [TRADE RELAY v4 / ID OCCUPIED] {} 자리 '{}' 에 이미 '{}' 타입 문서가 있습니다. index 는 서식 코드를 해시에 포함하므로 이 자리는 같은 서식의 같은 번호여야 합니다. 덮어쓰지 않습니다.",
                                        target_type, draft_id, occupant.r#type
                                    ));
                                    continue;
                                }
                                let reverse_field = crate::logic::trade_reference_field_of(doc_type).unwrap_or("");
                                let my_number = final_data
                                    .get("doc_number")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.trim().to_string())
                                    .filter(|s| !crate::model::merge::is_schema_echo(s))
                                    .unwrap_or_else(|| raw_no.to_string());
                                let mut draft_data = json!({});
                                if let Some(obj) = draft_data.as_object_mut() {
                                    obj.insert("id".to_string(), json!(draft_id.clone()));
                                    obj.insert("type".to_string(), json!(target_type));
                                    obj.insert("index".to_string(), json!(draft_index));
                                    obj.insert(search_key.clone(), json!(link_value.clone()));
                                    if !reverse_field.is_empty() && reverse_field != search_key.as_str() && !my_number.starts_with("AUTO-") {
                                        obj.insert(reverse_field.to_string(), json!(my_number.clone()));
                                    }
                                    obj.insert(crate::logic::trading_index_column(doc_type), json!(index_val));
                                    obj.insert("doc_type".to_string(), json!(target_type));
                                    obj.insert("updated_at".to_string(), json!(0));
                                    obj.insert("mode".to_string(), json!("shipping"));
                                    obj.insert(crate::utils::canonical::LEDGER_KEY.to_string(), json!("placeholder"));
                                    obj.insert(crate::utils::canonical::RELAY_ORIGIN_KEY.to_string(), json!([doc_type]));
                                    obj.insert("text".to_string(), json!(format!("{} draft (ref: {} = {})", target_type, search_key, link_value)));
                                }
                                let _ = db.upsert_item(
                                    "items", &draft_id, target_type, draft_data, None,
                                    None,
                                    Some(from_addr), Some(&team_id), Some(&hashed_cc),
                                    Some(&crate::scheduler::entity_bcc(target_type, &hashed_cc)),
                                    Some(&ref_val), None
                                ).await;
                                vision_drafted += 1;
                                crate::scheduler::relay_ledger::add_delta(
                                    &mut vision_stats,
                                    target_type,
                                    crate::utils::canonical::LEDGER_PLACEHOLDER_DELTA,
                                );
                                emit_term(&format!(
                                    "  📝 [TRADE RELAY v4] {} 자리 초안 '{}' 생성 ({}: '{}', index={}, 역참조 {}='{}'). 원본이 텍스트·이미지 어느 경로로 들어와도 같은 id 로 착지해 초안이 해소됩니다.",
                                    target_type, draft_id, search_key, link_value, draft_index, reverse_field, my_number
                                ));
                            },
                        }
                    }
                }

                if relay_plan.is_empty() {
                    emit_term("  ⚪ [TRADE RELAY v4] 릴레이 키가 확보되지 않았습니다. 추출 결과에서 유효한 참조 번호가 없습니다.");
                } else {
                    let linked = relay_plan.iter()
                        .filter(|(_, k)| !k.raw.is_empty() && k.raw != "N/A")
                        .count();

                    emit_term(&format!(
                        "  ✅ [TRADE RELAY v4 SUMMARY] 계획 {}건 | 유효 키 {}건 | 역할: {:?}",
                        relay_plan.len(),
                        linked,
                        relay_plan.iter().map(|(t, k)| format!("{}:{}", t, k.role)).collect::<Vec<_>>()
                    ));
                }
                if is_trade_doc {
                    let now_ms_metrics = chrono::Utc::now().timestamp_millis();
                    let mut metrics_doc = final_data.clone();
                    if let Some(o) = metrics_doc.as_object_mut() {
                        if o.get("type").is_none() {
                            o.insert("type".to_string(), json!(doc_type));
                        }
                        if o.get("created_at").is_none() {
                            o.insert("created_at".to_string(), json!(now_ms_metrics));
                        }
                        let rel_keys: Vec<String> = o.keys().filter(|k| k.starts_with("rel_")).cloned().collect();
                        for k in rel_keys {
                            o.remove(&k);
                        }
                    }
                    let _ = crate::utils::metrics::update_team_base_metrics(db, &team_id, &hashed_cc, &vec![metrics_doc], vision_stats.clone()).await;
                    emit_term(&format!(
                        "  📊 [VISION LEDGER] {} '{}' 저장 전 상태 {:?} | 기존 문서 연결 {}건 · 자리 초안 {}건 | 원장 변화 {:?} → 팀 base(cc=trading) 에 반영했습니다. 텍스트 경로와 같은 팀·cc 라 두 경로의 count 가 한곳에 모입니다.",
                        doc_type, hashed_id, prev_prior, vision_linked, vision_drafted, vision_stats
                    ));
                }
                let _ = db.update_task_status(&task_id, 9).await;
                let _ = db.update_message_status(&task_id, 9, Some("Extraction Complete")).await;
            }
            
            emit_term("[SUCCESS] Task Completed. Data saved.");
            
            let payload = json!({ 
               "task_id": task_id.clone(),
               "category": "Done", "summary": "Analysis Complete", "spinner": "✅", "data": extracted_data
            });
            
            // 🌟 [CRITICAL FIX] Done 상태를 파일에도 확실히 기록하여 상세페이지 복구 시 100% 출력되게 합니다!
            crate::utils::logger::log_task_progress(app_handle, &task_id, &payload);
            
            crate::utils::sync_utils::notify_new_task();

            // 🌟 [SDS] 비전 태스크 경계에서 관측을 확정합니다.
            //
            //  ── 왜 함수 끝이 아니라 여기인가 ──
            //   이 함수의 본문 마지막은
            //     if let Ok(img) = image::open(...) { ... Ok(()) } else { Ok(()) }
            //   이고, 이 if/else 자체가 함수의 꼬리 표현식(반환값)입니다.
            //   그 뒤에 문장을 붙이면 if/else 가 '문장' 이 되어 값 타입이 ()
            //   이어야 하는데 실제로는 Result<()> 라 E0308 로 컴파일이 깨지고,
            //   설령 통과해도 위 분기에서 이미 반환되므로 도달하지 못합니다.
            //   따라서 성공 분기의 Ok(()) '직전' 이 유일하게 올바른 위치입니다.
            //
            //  ── 취소·에러 경로를 덮지 못하는 것은 손실이 아닙니다 ──
            //   본문 중간에 `return Ok(())`(사용자 취소) 와 `?`(에러 전파) 가 있어
            //   그 경로는 이 지점을 지나지 않습니다. 그러나
            //     · enter_scope 는 다음 태스크 진입 시 스코프를 덮어쓰고
            //     · flush 를 놓친 관측은 DIRTY=true 로 메모리에 남아
            //       다음 태스크의 flush 또는 unload_model / 앱 종료 flush 가 기록합니다.
            //   즉 유실이 아니라 '지연' 이며, 그래서 Drop 가드를 도입하지 않습니다.
            emit_term(&format!("[ENGINE] {}", crate::utils::score_dynamics::report()));
            crate::utils::score_dynamics::flush();
            crate::utils::score_dynamics::leave_scope();
            emit_term(&format!("[ENGINE] ✅ Image extraction pipeline complete for Task: {}", task_id));
            Ok(())
        } else {
            // 🌟 [SDS] 이미지 파일을 열지 못한 경로입니다.
            //    관측이 하나도 없으므로 flush 는 불필요하고 스코프만 내립니다.
            //    (flush 는 dirty 가 false 면 어차피 파일을 쓰지 않습니다)
            crate::utils::score_dynamics::leave_scope();
            Ok(())
        }
    }

    async fn remap_off_schema_axes<E: Fn(&str)>(
        &self,
        map: &mut serde_json::Map<String, Value>,
        claims: &mut Vec<crate::models::siglip2::value_grounding::GroundingClaim>,
        doc_type: &str,
        doc_lang: &str,
        emit: E,
    ) -> usize {
        use crate::utils::ai_utils::{cosine_similarity, detect_field_format, semantic_anchor_text, value_matches_format, FieldFormat};

        let compat = |a: FieldFormat, b: FieldFormat| -> bool {
            if a == b { return true; }
            matches!(
                (a, b),
                (FieldFormat::Text, FieldFormat::Address)
                    | (FieldFormat::Address, FieldFormat::Text)
                    | (FieldFormat::Numeric, FieldFormat::Identifier)
                    | (FieldFormat::Identifier, FieldFormat::Numeric)
            )
        };

        let schema: Vec<String> = crate::parsing::get_detail_schema_fields(doc_type, "", doc_lang)
            .into_iter()
            .map(|(f, _, _, _)| f)
            .filter(|f| !f.contains(','))
            .collect();
        if schema.is_empty() { return 0; }

        let mut orphans: Vec<(String, String)> = Vec::new();
        for (k, v) in map.iter() {
            if v.is_object() || v.is_array() || v.is_null() { continue; }
            let s = match v {
                Value::String(s) => s.trim().to_string(),
                Value::Number(n) => n.to_string(),
                _ => continue,
            };
            if s.is_empty() || crate::model::merge::is_schema_echo(&s) { continue; }
            if schema.iter().any(|f| f == k) { continue; }
            let (known, cat) = crate::model::merge::trade_schema_owner_of(doc_type, k);
            if !known || !cat.is_empty() { continue; }
            orphans.push((k.clone(), s));
        }
        if orphans.is_empty() { return 0; }

        let empty_at = |m: &serde_json::Map<String, Value>, f: &str| -> bool {
            m.get(f).map_or(true, |x| {
                x.is_null() || x.as_str().map(|s| s.trim().is_empty()).unwrap_or(false)
            })
        };

        let orphan_texts: Vec<String> = orphans
            .iter()
            .map(|(k, _)| semantic_anchor_text(doc_lang, doc_type, k))
            .collect();
        let schema_groups: Vec<Vec<String>> = schema
            .iter()
            .map(|f| {
                let (mut ph, _) = crate::model::merge::owner_label_bank(doc_lang, doc_type, f);
                for p in crate::utils::ai_utils::split_bias_phrases_full(&semantic_anchor_text(doc_lang, doc_type, f)) {
                    if crate::utils::ai_utils::is_value_example_phrase(&p) { continue; }
                    if !ph.iter().any(|e| e.eq_ignore_ascii_case(&p)) { ph.push(p); }
                }
                ph
            })
            .collect();
        let orphan_embs = match self.get_embedding_batch(orphan_texts).await {
            Ok(e) if e.len() == orphans.len() => e,
            _ => {
                emit("  ⚪ [SCHEMA AXIS MAP SKIP] 앵커 임베딩을 만들지 못해 스키마 밖 축을 그대로 둡니다.");
                return 0;
            }
        };
        let schema_heads: Vec<Vec<f32>> = self
            .ship_embed_phrase_groups(&schema_groups)
            .await
            .iter()
            .map(|bank| crate::model::merge::bank_centroid(bank))
            .collect();

        let mut moved = 0usize;
        for (oi, (key, raw)) in orphans.iter().enumerate() {
            let q = &orphan_embs[oi];
            if q.iter().all(|&x| x == 0.0) { continue; }
            let want = detect_field_format(key);
            let multiline = raw.lines().filter(|l| !l.trim().is_empty()).count() >= 2;
            let mut scored: Vec<(String, f32)> = Vec::new();
            for (si, f) in schema.iter().enumerate() {
                if !empty_at(map, f) { continue; }
                if multiline && detect_field_format(f) != FieldFormat::Address { continue; }
                if !compat(want, detect_field_format(f)) { continue; }
                if !value_matches_format(detect_field_format(f), raw) { continue; }
                let cat = crate::logic::trade_field_category(f);
                if cat.is_empty() || crate::logic::is_trade_array_category(cat) { continue; }
                let e = &schema_heads[si];
                if e.is_empty() { continue; }
                scored.push((f.clone(), cosine_similarity(q, e)));
            }
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            if scored.is_empty() {
                emit(&format!(
                    "  ⚪ [SCHEMA AXIS MAP] '{}' 를 받아 줄 빈 스키마 축이 하나도 없습니다. 루트에만 남겨 둡니다.",
                    key
                ));
                continue;
            }
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            if scored.len() < 3 {
                let strict = multiline && (scored.len() == 1 || scored[0].1 > scored[1].1);
                if !strict {
                    emit(&format!(
                        "  ⚪ [SCHEMA AXIS MAP] '{}' 를 받아 줄 빈 스키마 축이 {}개뿐이라 자기 분포로 이상치를 판정할 수 없습니다. 루트에만 남겨 둡니다.",
                        key, scored.len()
                    ));
                    continue;
                }
                emit(&format!(
                    "  🧭 [SCHEMA AXIS MAP / MULTILINE] '{}' 은 줄바꿈으로 나뉜 주소 블록이라 후보를 주소 축 {}개로 좁혔습니다. 표본이 적어 분포 판정은 불가능하지만 형태가 이미 축의 종류를 확정했으므로 엄격 argmax 로 '{}'({:.4}) 를 채택합니다.",
                    key, scored.len(), scored[0].0, scored[0].1
                ));
            } else {
                let tail: Vec<f32> = scored[1..].iter().map(|(_, s)| *s).collect();
                let n = tail.len() as f32;
                let mean = tail.iter().sum::<f32>() / n;
                let sd = (tail.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n)
                    .sqrt();
                if sd <= 1e-6 || scored[0].1 - mean < sd {
                    emit(&format!(
                        "  ⚪ [SCHEMA AXIS MAP] '{}' 의 최고 후보 '{}'({:.4}) 가 나머지 평균 {:.4} 에서 표준편차 {:.4} 만큼 떨어지지 못했습니다. 어느 축이라고 단정할 근거가 없으므로 루트에만 남겨 둡니다.",
                        key, scored[0].0, scored[0].1, mean, sd
                    ));
                    continue;
                }
            }

            let target = scored[0].0.clone();
            let cat = crate::logic::trade_field_category(&target).to_string();
            let val = map.remove(key).unwrap_or(json!(raw.clone()));
            map.insert(target.clone(), val.clone());
            let slot = map
                .entry(cat.clone())
                .or_insert_with(|| Value::Object(serde_json::Map::new()));
            if let Some(o) = slot.as_object_mut() {
                o.insert(target.clone(), val);
            }
            for c in claims.iter_mut() {
                if c.field == *key && c.value.trim() == raw.trim() {
                    c.field = target.clone();
                    c.category = cat.clone();
                }
            }
            crate::utils::score_dynamics::record_baseline("vision.schema_axis_map", 1.0);
            emit(&format!(
                "  🧭 [SCHEMA AXIS MAP] '{}' = \"{}\" → {}.{} (앵커 코사인 {:.4}). 이 키는 '{}' 서식의 로드된 스키마에 이름이 없지만 그 개념의 축은 존재합니다. 이름 완전일치만 보면 값이 루트에만 남아, 자연어 변환은 존재하지 않는 절을 만들고 청크 인덱싱의 스키마 화이트리스트가 그 절을 다시 폐기합니다. 읽어낸 값이 저장은 되고도 검색 경로에서는 존재하지 않게 되는 지점입니다.",
                key, raw, cat, target, scored[0].1, doc_type
            ));
            moved += 1;
        }
        moved
    }

    pub async fn chat_with_qwen3_5_image_spinner(
        &self, 
        system: &str,       
        user_input: &str,   
        image: Option<DynamicImage>,
        _app_handle: &tauri::AppHandle,
        _event_name: &str,
        mut base_payload: Value,
        max_tokens: usize,
        cancellation_token: Option<Arc<AtomicBool>>,
        session_id: Option<String>,
        semantic_prejudice: Option<&str>   // 🌟 추가
    ) -> anyhow::Result<String> {
        // [VISION-DYNAMIC] 🌟 target_size 로직 삭제하고 바로 bool 전달
        self.ensure_qwen3_5(image.is_some()).await?;

        // [FIX] Inject task_id from session_id if it's a task reference
        if let Some(ref sid) = session_id {
            if sid.starts_with("task_") || sid.starts_with("img_") {
                if let Some(obj) = base_payload.as_object_mut() {
                    obj.insert("task_id".to_string(), json!(sid));
                }
            }
        }

        // [LOG] Save to task history if task_id exists
        if let Some(task_id) = base_payload.get("task_id").and_then(|v| v.as_str()) {
            crate::utils::logger::log_task_progress(_app_handle, task_id, &base_payload); // 기존 변수명이 app_handle이면 app_handle로 사용
        }
        
        // 🌟 [CRITICAL FIX] 화면에 실시간 진행률(퍼센트)을 쏘아 보내는 코드를 복구합니다!
        let _ = _app_handle.emit(_event_name, &base_payload); // 기존 변수명이 app_handle이면 app_handle, _event_name이면 _event_name 사용
        
        let mut q35_gen_guard = self.qwen3_5_generator.lock().await;
        let gen = q35_gen_guard.as_mut().ok_or_else(|| anyhow!("Qwen 3.5 Generator is unloaded"))?;
        
        let mut content_parts = Vec::new();
        
        if let Some(img) = image {
            let mut buf = Cursor::new(Vec::new());
            img.write_to(&mut buf, image::ImageFormat::Png)?;
            let b64 = BASE64_STANDARD.encode(buf.into_inner());
            let url = format!("data:image/png;base64,{}", b64);
            
            content_parts.push(ChatCompletionRequestMessageContentPart::ImageURL(
                ChatCompletionRequestMessageContentPartImage {
                    image_url: ImageURL { url, detail: None }
                }
            ));
        }

        // User Text 할당
        content_parts.push(ChatCompletionRequestMessageContentPart::Text(
            ChatCompletionRequestMessageContentPartText { text: user_input.to_string() }
        ));

        // System 메시지 명시적 생성
        let system_message = ChatCompletionRequestMessage::System(crate::openai_types::ChatCompletionRequestSystemMessage {
            content: system.to_string(),
            name: None,
        });

        // User 메시지 명시적 생성
        let user_message = ChatCompletionRequestUserMessage {
            content: ChatCompletionRequestUserMessageContent::Array(content_parts),
            name: None,
        };

        // 파라미터 세팅
        let params = ChatCompletionParameters {
            messages: vec![system_message, ChatCompletionRequestMessage::User(user_message)],
            model: "qwen3.5".to_string(),
            max_tokens: Some(max_tokens as u32),
            temperature: Some(0.0),
            top_p: Some(0.95),
            ..Default::default()
        };
        
        // 🌟 [KV SESSION PER CALL] 비전 호출마다 KV 디렉터리를 새로 씁니다.
        //  같은 task_id 세션을 크롭마다 재사용하면, SSD 계획으로 흘러간 앞선 호출이
        //  DirectStorage 로 열어 둔 b0 의 어텐션 층 파일을 다음 호출이 지우고 다시 만들다
        //  delete-pending 이름에 부딪혀 '액세스가 거부되었습니다 (os error 5)' 가 납니다.
        //  크롭마다 프롬프트가 달라 세션 접두 캐시가 재사용될 여지도 없습니다.
        //  base_payload.task_id 는 원래 session_id 로 이미 주입되었으므로 로그 귀속은 바뀌지 않습니다.
        let kv_session: Option<String> = session_id.as_ref().map(|sid| {
            format!("{}_v{}", sid, VISION_KV_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
        });
        let kv_dir = kv_session
            .as_ref()
            .map(|s| crate::utils::paths::get_kv_dir(Some(_app_handle)).join(s));
        let out = gen.generate(
            params,
            cancellation_token.clone(),
            kv_session,
            Some("inference".to_string()),
            None,
            semantic_prejudice
        ).await.map_err(|e| anyhow!("Qwen 3.5 Inference failed: {}", e));
        if let Some(d) = kv_dir {
            let _ = std::fs::remove_dir_all(&d);
        }
        out
    }
}
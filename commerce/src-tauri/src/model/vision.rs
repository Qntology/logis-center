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

            if !is_trade_doc {
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

                // 마진 부족 시에만 LLM 재판정 1회
                if verdict.code_margin < 0.15 && verdict.code_candidates.len() > 1 {
                    emit_term(&format!(
                        "  🤝 [TIE BREAK] 코드 마진 {:+.4} 가 임계 미만. LLM 재판정 1회 수행.",
                        verdict.code_margin
                    ));
                    let prompt = crate::parsing::get_trade_doc_classification_prompt_with_evidence(
                        &verdict.group,
                        &verdict.code_candidates,
                    );
                    let type_res = self.chat_with_qwen3_5_image_spinner(
                        "You are a document classifier.", &prompt, Some(dynamic_image.clone()), app_handle, "extraction-progress",
                        json!({ "category": "Vision (Step 2)", "summary": "Verifying document type..." }), 64, cancel_token.clone(), Some(task_id.clone()), None
                    ).await?;
                    if let Some(v) = crate::parsing::parse_json_from_llm(&type_res).get("doc_type").and_then(|d| d.as_str()) {
                        if verdict.code_candidates.iter().any(|(c, _)| c == v) {
                            emit_term(&format!("  ✅ [TIE BREAK] LLM 판정 '{}' 채택.", v));
                            detected_type = v.to_string();
                        } else {
                            emit_term(&format!(
                                "  🚫 [TIE BREAK] LLM 이 후보 밖 '{}' 반환. 비전 판정 '{}' 유지.",
                                v, detected_type
                            ));
                        }
                    }
                }

                emit_term(&format!("✅ Document identified as: **{}** (group: {})", detected_type, verdict.group));
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
                    let mut heatmaps = self
                        .with_siglip_text("column heatmaps (trade)", |m| {
                            crate::models::siglip2::vision_encoder::build_column_heatmaps(
                                m, &grid, &detected_type, &language, Some(&legibility), &title_prej, &emit_term
                            )
                        })
                        .await
                        .map_err(|e| anyhow::anyhow!("Heatmap build failed: {}", e))?;
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
                    final_data_map.insert("header".to_string(), json!({"doc_type": detected_type}));
                    final_data_map.insert("parties".to_string(), json!({}));
                    final_data_map.insert("logistics".to_string(), json!({}));
                    final_data_map.insert("conditions".to_string(), json!({}));
                    final_data_map.insert("financials".to_string(), json!({}));
                    final_data_map.insert("cargo".to_string(), json!({}));
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

                        let (tile_count, _why) = crate::models::siglip2::vision_crop::decide_tile_count(
                            plan,
                            &heatmaps,
                            &grid,
                            &legibility,
                            crate::logic::TRADE_ARRAY_CATEGORIES,
                            &emit_term,
                        );
                        let row_tile_cat = crate::logic::TRADE_ARRAY_CATEGORIES
                            .iter()
                            .any(|c| *c == plan.category.as_str());
                        let tiles = if row_tile_cat && tile_count > 1 {
                            crate::models::siglip2::vision_crop::plan_row_tiles(
                                &dynamic_image, plan.bbox, &emit_term
                            )
                            .unwrap_or_else(|| {
                                crate::models::siglip2::vision_crop::plan_overlap_tiles(
                                    plan.bbox, tile_count, 0.25
                                )
                            })
                        } else {
                            crate::models::siglip2::vision_crop::plan_overlap_tiles(
                                plan.bbox, tile_count, 0.25
                            )
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
                            let is_array_cat = crate::logic::TRADE_ARRAY_CATEGORIES
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

                            let prompt = crate::parsing::get_trade_crop_prompt(
                                &plan.category,
                                &detected_type,
                                &plan.top_field,
                                plan.score,
                                &claimed,
                            );
                            let verify_crop = crop.clone();

                            let tile_res = self.chat_with_qwen3_5_image_spinner(
                                "You are a highly precise document data extraction assistant.",
                                &prompt,
                                Some(crop),
                                app_handle,
                                "extraction-progress",
                                json!({
                                    "category": format!("Vision (Crop {}/{}{})", idx + 1, plans.len(), tile_tag),
                                    "summary": format!("Extracting {}...", plan.category)
                                }),
                                1024,
                                cancel_token.clone(),
                                Some(task_id.clone()),
                                None
                            ).await?;

                            let mut tile_json = crate::parsing::parse_json_from_llm(&tile_res);
                            if !is_array_cat {
                                let echo_fields: Vec<(String, String)> = tile_json
                                    .as_object()
                                    .map(|o| {
                                        o.iter()
                                            .filter_map(|(k, v)| {
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
                                    .unwrap_or_default();
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
                            }
                            record_claim_violations(
                                &claimed,
                                &tile_json,
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
                        const FIELD_RECOVERY_BUDGET: usize = 4;
                        let mut cands: Vec<(String, String, usize, f32)> = Vec::new();
                        for hm in heatmaps.iter() {
                            if crate::logic::TRADE_ARRAY_CATEGORIES.iter().any(|c| *c == hm.category.as_str()) { continue; }
                            for (field, patch, z) in hm.field_peaks.iter() {
                                if field.starts_with("__") || field == "doc_type" { continue; }
                                let filled = final_data_map
                                    .get(field)
                                    .map(|v| !(v.is_null() || v.as_str().map(|s| s.trim().is_empty()).unwrap_or(false)))
                                    .unwrap_or(false);
                                if filled { continue; }
                                let legible = legibility.verdict.get(*patch).copied()
                                    == Some(crate::models::siglip2::legibility::PatchLegibility::Legible);
                                if !legible { continue; }
                                cands.push((hm.category.clone(), field.clone(), *patch, *z));
                            }
                        }
                        let windows = crate::model::merge::plan_recovery_windows(
                            &cands,
                            grid.grid_rows,
                            grid.grid_cols,
                            grid.orig_width,
                            grid.orig_height,
                            FIELD_RECOVERY_BUDGET,
                        );
                        if windows.is_empty() {
                            emit_term("  ⚪ [FIELD RECOVERY] 비어 있으면서 자기 라벨 봉우리를 가진 필드가 없습니다.");
                        } else {
                            emit_term(&format!(
                                "  🩺 [FIELD RECOVERY] 빈 필드 후보 {}개 | 봉우리 주변 소형 크롭 {}개를 다시 읽습니다 (상한 {}회).",
                                cands.len(), windows.len(), FIELD_RECOVERY_BUDGET
                            ));
                            let schema_fields: Vec<String> = crate::parsing::get_detail_schema_fields(&detected_type, "", &language)
                                .into_iter()
                                .map(|(f, _, _, _)| f)
                                .filter(|f| f != "id,link" && f != "status" && f != "doc_type")
                                .collect();
                            let mut gate_banks: Option<Vec<(String, Vec<Vec<f32>>, Vec<f32>)>> = None;
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
                                let micro = crate::models::siglip2::vision_crop::crop_region(&dynamic_image, &micro_plan, 512);
                                let defs: Vec<(String, String)> = fields
                                    .iter()
                                    .map(|(_, f, _)| (f.clone(), crate::parsing::trade_field_definition(&language, f)))
                                    .collect();
                                emit_term(&format!(
                                    "    🔎 [RECOVERY CROP {}] px({},{})-({},{}) | 필드 {:?}",
                                    wi + 1, bbox.0, bbox.1, bbox.2, bbox.3,
                                    fields.iter().map(|(c, f, z)| format!("{}.{}(z {:+.2})", c, f, z)).collect::<Vec<_>>()
                                ));
                                let prompt = crate::parsing::get_trade_recovery_prompt(&detected_type, &defs);
                                let res = self.chat_with_qwen3_5_image_spinner(
                                    "You are a highly precise document data extraction assistant.",
                                    &prompt,
                                    Some(micro),
                                    app_handle,
                                    "extraction-progress",
                                    json!({
                                        "category": format!("Vision (Recovery {})", wi + 1),
                                        "summary": "Re-reading empty fields..."
                                    }),
                                    160,
                                    cancel_token.clone(),
                                    Some(task_id.clone()),
                                    None
                                ).await?;
                                let parsed = crate::parsing::parse_json_from_llm(&res);
                                for (cat, field, _) in fields.iter() {
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
                                    if value.is_empty() || label.is_empty() {
                                        emit_term(&format!(
                                            "      ⚪ [RECOVERY NULL] {}.{} | 이 영역에 해당 라벨과 값이 없다고 답했습니다.",
                                            cat, field
                                        ));
                                        continue;
                                    }
                                    if crate::model::merge::is_schema_echo(&value)
                                        || value.eq_ignore_ascii_case(&label)
                                        || crate::parsing::is_printed_label_echo(&value, &language)
                                        || crate::parsing::is_printed_label_fragment(&value, &language)
                                    {
                                        emit_term(&format!(
                                            "      🚫 [RECOVERY LABEL AS VALUE] {}.{} = \"{}\" | 값 자리에 라벨이 들어왔습니다.",
                                            cat, field, value
                                        ));
                                        continue;
                                    }
                                    let claimed = collect_claimed(&final_data_map);
                                    if let Some((owner, _)) = claimed.iter().find(|(_, v)| v.eq_ignore_ascii_case(&value)) {
                                        emit_term(&format!(
                                            "      🚫 [RECOVERY CLAIMED] {}.{} = \"{}\" | 이미 '{}' 가 확정한 값입니다.",
                                            cat, field, value, owner
                                        ));
                                        continue;
                                    }
                                    if gate_banks.is_none() {
                                        let mut phr_all: Vec<String> = Vec::new();
                                        let mut per_field: Vec<(String, Vec<String>, Vec<f32>)> = Vec::new();
                                        for f in schema_fields.iter() {
                                            let (ph, wt) = crate::utils::ai_utils::label_phrase_bank(&language, "shipping_doc", f);
                                            for p in ph.iter() {
                                                if !phr_all.contains(p) { phr_all.push(p.clone()); }
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
                                        gate_banks = Some(banks);
                                    }
                                    let label_emb = self.get_embedding(label.clone()).await.unwrap_or_default();
                                    let banks = gate_banks.as_ref().map(|b| b.as_slice()).unwrap_or(&[]);
                                    let (ok, own, rival, rival_field) =
                                        crate::model::merge::recovery_label_gate(&label_emb, field, banks);
                                    if !ok {
                                        emit_term(&format!(
                                            "      🚫 [RECOVERY LABEL GATE] {}.{} = \"{}\" | 읽힌 라벨 \"{}\" 의 코사인: 자기 {:.4} vs '{}' {:.4} — 다른 필드의 라벨로 판정되어 폐기합니다.",
                                            cat, field, value, label, own, rival_field, rival
                                        ));
                                        continue;
                                    }
                                    let mut patch = serde_json::Map::new();
                                    patch.insert(field.clone(), json!(value.clone()));
                                    let patch = Value::Object(patch);
                                    record_grounding_claims(&mut grounding_claims, cat, &patch, bbox);
                                    merge_extracted(&mut final_data_map, cat, &patch, &emit_term);
                                    emit_term(&format!(
                                        "      ✅ [RECOVERED] {}.{} = \"{}\" | 라벨 \"{}\" (자기 cos {:.4} vs 최강 경쟁 '{}' {:.4})",
                                        cat, field, value, label, own, rival_field, rival
                                    ));
                                }
                            }
                        }
                    }

                    extracted_data = Value::Object(final_data_map);
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
                let mut heatmaps = self
                    .with_siglip_text("column heatmaps (commerce)", |m| {
                        crate::models::siglip2::vision_encoder::build_column_heatmaps(
                            m, &grid, commerce_page_type, &language, Some(&legibility), &[], &emit_term
                        )
                    })
                    .await
                    .map_err(|e| anyhow::anyhow!("Commerce heatmap failed: {}", e))?;

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

                        let crop = crate::models::siglip2::vision_crop::crop_region(
                            &dynamic_image, plan, 512
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

                let dup_groups = crate::model::merge::cross_field_duplicate_groups(&grounding_claims, &verdicts);
                if !dup_groups.is_empty() {
                    let bank_type = if is_trade_doc { "shipping_doc" } else { "goods" };
                    let mut texts: Vec<String> = Vec::new();
                    for (value, owners) in dup_groups.iter() {
                        if !texts.contains(value) {
                            texts.push(value.clone());
                        }
                        for (_, field, _) in owners.iter() {
                            let (phrases, _) = crate::utils::ai_utils::label_phrase_bank(&language, bank_type, field);
                            for p in phrases {
                                if !texts.contains(&p) {
                                    texts.push(p);
                                }
                            }
                        }
                    }
                    let embs = self.get_embedding_batch(texts.clone()).await.unwrap_or_default();
                    let lookup: std::collections::HashMap<String, Vec<f32>> =
                        texts.into_iter().zip(embs.into_iter()).collect();
                    let owner_verdicts = crate::model::merge::resolve_cross_field_duplicates(
                        &dup_groups, &lookup, &language, bank_type, &emit_term,
                    );
                    verdicts.extend(owner_verdicts);
                }

                if let Some(map) = extracted_data.as_object_mut() {
                    apply_grounding_verdicts(map, &verdicts, &emit_term);
                    if is_trade_doc {
                        crate::model::merge::drop_row_echo_columns(map, &emit_term);
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
                    .filter(|s| !s.is_empty() && s != "N/A" && s != "null");
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
            }
            let nl = crate::parsing::json_to_natural_language(&extracted_data);
            let doc_type = if is_trade_doc {
                extracted_data.get("header")
                    .and_then(|h| h.get("doc_type"))
                    .and_then(|s| s.as_str())
                    .or_else(|| extracted_data.get("doc_type").and_then(|s| s.as_str()))
                    .unwrap_or("shipping_doc")
            } else {
                "goods"
            };
            
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
                let hashed_cc = crate::utils::hash::hash_id(if is_trade_doc { "local.shipping" } else { "local.commerce" });

                // 식별자(ID) 추출 기준 분기
                // 🌟 [DOC NUMBER RESOLVE]
                //  ── 무엇이 문제였나 ──
                //   Slice & Merge 경로의 extracted_data 는 { header:{...}, parties:{...}, ... } 중첩이라
                //   루트에 document_number 가 없고, TRACKING Fast-Track 경로는 루트에 tracking_number 를 넣습니다.
                //   기존 코드는 무역 모드에서 '루트 document_number' 하나만 봤기 때문에
                //   두 경로 모두 항상 None → raw_no = task_id 였습니다.
                //   task_id 는 스캔마다 새로 생기므로 index/id/ref 가 매번 달라져
                //   같은 문서를 다시 스캔해도 upsert 가 아니라 신규 행이 계속 쌓였습니다.
                //  ── 탐색 순서 ──
                //   header.document_number → header.doc_number
                //   → 루트 document_number → 루트 doc_number → 루트 tracking_number
                //   "N/A" 는 LLM 이 '못 찾았다' 는 뜻으로 쓰는 값이라 식별자가 될 수 없습니다.
                let raw_no_owned: String = if is_trade_doc {
                    // 🌟 [DOC IDENTITY v3] parsing.rs 의 resolve_trade_doc_identity 가
                    //    접두어 완전일치 + 벡터 근거로 문서 식별자를 확정합니다.
                    //    기존은 header / 루트만 훑다가 없으면 즉시 task_id 폴백이었습니다.
                    //    그 결과 'BL-55432219' 가 r2~r3 에 인쇄되어 있어도
                    //    doc_number = "" → task_id 폴백 → 재스캔마다 다른 index 가 되어
                    //    같은 문서가 누적되었습니다.
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
                            .filter(|s| !s.is_empty() && s.as_str() != "N/A")
                            .unwrap_or_else(|| task_id.clone())
                    }
                } else {
                    extracted_data.get("tracking_number")
                        .and_then(|s| s.as_str())
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty() && s.as_str() != "N/A")
                        .unwrap_or_else(|| task_id.clone())
                };
                let raw_no: &str = raw_no_owned.as_str();
                emit_term(&format!("[STAGE-3] 문서 식별자 확정: '{}' (task_id 폴백 여부: {})",
                    raw_no, raw_no == task_id.as_str()));

                let table_name = "items"; 
                
                let clean_no = crate::utils::hash::normalize_identifier(raw_no);
                // 🌟 [RELAY INDEX v3] hash.rs 의 relay_index 를 사용합니다.
                //    기존은 `crc32(hash_id(type + clean_no))` 였는데,
                //    이 경로에는 `normalize_identifier` 의 전각 접기가 반영되지 않았습니다.
                //    `relay_index` 는 `normalize_identifier` 통과값을 받아
                //    전각 영숫자(ＣＩ－４３７２６)도 반각과 동일하게 취급합니다.
                let index_val = if crate::utils::hash::is_valid_relay_key(raw_no) {
                    crate::utils::hash::relay_index(raw_no)
                } else {
                    // 유효하지 않은 키(예: task_id 폴백)는 기존 경로 유지
                    crate::utils::hash::crc32(&crate::utils::hash::hash_id(&format!("{}{}", doc_type, clean_no)))
                };
                let hashed_id = crate::utils::hash::hash_id(&format!("{}{}", team_id, index_val));
                let ref_val = crate::utils::hash::hash_id(&format!("{}{}{}", team_id, hashed_cc, clean_no));

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
                    const TRADE_GROUPS: [&str; 6] =
                        ["header", "parties", "logistics", "financials", "conditions", "cargo"];

                    // bias.json 의 path_alias 를 역방향(alias -> canonical)으로 사용합니다.
                    // build_dexie_plan 은 canonical 로 조건을 모으므로,
                    // 저장 시점에도 canonical 이름으로 올려야 두 방향이 만납니다.
                    fn canonical_name(raw: &str) -> String {
                        let k = raw.trim();
                        if let Some(alias_obj) = crate::parsing::BIAS_DICT
                            .get("search_bridge")
                            .and_then(|sb| sb.get("path_alias"))
                            .and_then(|v| v.as_object())
                        {
                            for (canonical, list) in alias_obj {
                                if canonical == k { return canonical.clone(); }
                                if let Some(arr) = list.as_array() {
                                    if arr.iter().any(|a| a.as_str().map_or(false, |s| s == k)) {
                                        return canonical.clone();
                                    }
                                }
                            }
                        }
                        k.to_string()
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
                                // "N/A" 는 LLM 이 '못 찾았다' 는 뜻으로 쓰는 값이라 조건이 될 수 없습니다.
                                if s.trim().is_empty() || s == "N/A" { continue; }
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
                    table_name, // 분기된 테이블 적용
                    &hashed_id,
                    doc_type,
                    final_data.clone(),
                    None,
                    vision_vec,
                    Some(from_addr),
                    Some(&team_id),
                    Some(&hashed_cc),
                    Some(&crate::utils::hash::hash_id(&format!("{}{}", doc_type, hashed_cc))),
                    Some(&ref_val),
                    Some(&item_digest)
                ).await;

                let mut relay_starved: Vec<String> = Vec::new();

                // 🌟 relay_plan 을 if is_trade_doc 블록 외부에서 선언하여
                //    블록 내부와 외부 모두에서 접근 가능하게 합니다.
                

                if is_trade_doc {
                    relay_plan = crate::parsing::plan_trade_relays(&doc_type, &extracted_data, &language);
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
                        if link_value.is_empty() || link_value == "N/A" {
                            continue;
                        }
                        let link_value = final_data.get(source_field)
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .trim()
                            .to_string();
                        if link_value.is_empty() || link_value == "N/A" {
                            relay_starved.push(format!("{}←{}(빈 키)", target_type, source_field));
                            continue;
                        }
                        emit_term(&format!(
                            "  🔗 [TRADE RELAY] {} → {} | {}='{}' 로 연결 검색...",
                            doc_type, target_type, search_field, link_value
                        ));
                        // 🌟 [RELAY SEARCH v5] get_all_items로 여러 결과를 가져온 후,
                        //    자기 자신 제외 + 타입 검증으로 유효한 상대 문서를 찾습니다.
                        //    find_item_by_property는 첫 번째 결과만 반환하므로,
                        //    자기 자신이 먼저 나오면 무조건 SELF-SKIP 되는 문제를 해결합니다.
                        let filter = format!("data LIKE '%\"{}\":\"{}\"%'", search_field, link_value.replace('\'', "''"));
                        let relay_search = db.get_all_items("items", 10, 0, Some(filter)).await;
                        let mut found_target: Option<(String, Value)> = None;
                        match relay_search {
                            Ok(docs) => {
                                for doc in docs {
                                    // 🌟 [SELF-SEARCH GUARD] 자기 자신 제외
                                    if doc.id == hashed_id {
                                        continue;
                                    }
                                    // 🌟 [TYPE GUARD] 검색된 문서의 타입이 목표 타입과 일치해야 합니다.
                                    //    저장 시 type_은 전체 이름(예: "COMMERCIAL INVOICE")으로 설정되지만,
                                    //    릴레이 검색 시 target_type은 코드(예: "BL", "PL")입니다.
                                    //    따라서 전체 이름을 코드로 변환하여 비교합니다.
                                    let found_doc_type = doc.r#type.clone();
                                    let found_code = crate::logic::doc_type_to_code(&found_doc_type);
                                    // data JSON에서도 doc_type 확인
                                    let parsed: Value = match serde_json::from_str(&doc.json_data) {
                                        Ok(v) => v,
                                        Err(_) => continue,
                                    };
                                    let data_doc_type = parsed.get("doc_type")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    let data_code = crate::logic::doc_type_to_code(data_doc_type);
                                    // 타입 검증: 전체 이름 또는 코드 모두 매칭 시도
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
                                    // 🌟 [FIELD VALUE VERIFY] search_field 값이 정확히 일치하는지 확인
                                    let field_val = parsed.get(search_field)
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    if field_val != link_value {
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
                        match found_target {
                            Some((existing_id, mut ej)) => {
                                let mut needs_update = false;
                                // 🌟 [REVERSE REFERENCE INJECT] 현재 문서의 식별자를 타겟의 참조 필드에 역주입합니다.
                                //    역할 기반으로 역참조 필드명을 결정합니다.
                                let reverse_field = crate::logic::trade_reference_field_of(&doc_type)
                                    .unwrap_or("");
                                if !reverse_field.is_empty() {
                                    if let Some(my_doc_number) = extracted_data.get("doc_number").and_then(|v| v.as_str()) {
                                        if !my_doc_number.is_empty() && my_doc_number != "N/A" {
                                            let existing_ref = ej.get(reverse_field).and_then(|v| v.as_str()).unwrap_or("");
                                            if existing_ref.is_empty() || existing_ref == "N/A" {
                                                ej.as_object_mut().unwrap().insert(reverse_field.to_string(), json!(my_doc_number));
                                                needs_update = true;
                                            }
                                        }
                                    }
                                }
                                // 🌟 [RELAY INDEX CROSS-LINK] relay_index 를 타겟 문서의 봉투에 주입합니다.
                                //    이렇게 하면 두 문서가 같은 릴레이 축에서 서로를 찾을 수 있습니다.
                                let my_relay_idx = if crate::utils::hash::is_valid_relay_key(raw_no) {
                                    crate::utils::hash::relay_index(raw_no)
                                } else {
                                    0
                                };
                                if my_relay_idx > 0 {
                                    let relay_col = crate::logic::trading_index_column(&doc_type);
                                    let their_relay = ej.get(&relay_col).and_then(|v| v.as_u64()).unwrap_or(0);
                                    if their_relay == 0 {
                                        ej.as_object_mut().unwrap().insert(relay_col.clone(), json!(my_relay_idx));
                                        needs_update = true;
                                    }
                                }
                                // 물류 정보 상호 보완 (vessel, pol, pod, etd, eta)
                                for field in ["vessel", "voyage_number", "pol", "pod", "etd", "eta"] {
                                    let my_val = extracted_data.get("logistics")
                                        .and_then(|l| l.get(field))
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    if my_val.is_empty() || my_val == "N/A" { continue; }
                                    let their_val = ej.get("logistics")
                                        .and_then(|l| l.get(field))
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("");
                                    if their_val.is_empty() || their_val == "N/A" {
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
                                    if my_val.is_empty() || my_val == "N/A" { continue; }
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
                                if needs_update {
                                    ej.as_object_mut().unwrap().insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                                    let merged_text = crate::parsing::json_to_natural_language(&ej);
                                    ej.as_object_mut().unwrap().insert("text".to_string(), json!(merged_text));
                                    ej.as_object_mut().unwrap().insert("masked_text".to_string(), json!(merged_text.clone()));
                                    let _ = db.upsert_item(
                                        "items", &existing_id, target_type, ej, None,
                                        None,
                                        Some(from_addr), Some(&team_id), Some(&hashed_cc),
                                        Some(&crate::utils::hash::hash_id(&format!("{}{}", target_type, hashed_cc))),
                                        Some(&ref_val), None
                                    ).await;
                                    emit_term(&format!(
                                        "  ✅ [TRADE RELAY v4] 기존 {} 문서 '{}' 에 {} 정보 병합 완료.",
                                        target_type, existing_id, doc_type
                                    ));
                                }
                            },
                            None => {
                                // 🌟 [DRAFT v5] 미발견 시 draft 생성.
                                //    기존은 `relay_id(&link_value)` 로 타입 미반영 해시를 사용했습니다.
                                //    25건 릴레이가 전부 같은 `draft_id` 로 서로를 덮어쓰는 사고가 발생했습니다.
                                //    `relay_id` 에 `target_type` 을 전달하여 릴레이 대상마다 고유한 `draft_id` 를 부여합니다.
                                let draft_id = if crate::utils::hash::is_valid_relay_key(&link_value) {
                                    crate::utils::hash::relay_id(&link_value, target_type)
                                } else {
                                    crate::utils::hash::hash_id(&format!("{}{}{}", team_id, target_type, link_value))
                                };
                                let mut draft_data = json!({});
                                if let Some(obj) = draft_data.as_object_mut() {
                                    obj.insert("id".to_string(), json!(draft_id.clone()));
                                    obj.insert("type".to_string(), json!(target_type));
                                    // 🌟 [SEARCH FIELD FIX] draft에는 search_field(상대 문서의 검색 대상 필드)에 값을 넣습니다.
                                    //    기존에는 target_field(=source_field)로 넣어 방향이 뒤집혔습니다.
                                    obj.insert(search_field.to_string(), json!(link_value.clone()));
                                    obj.insert("doc_type".to_string(), json!(target_type));
                                    obj.insert("updated_at".to_string(), json!(0));
                                    obj.insert("mode".to_string(), json!("shipping"));
                                    obj.insert("text".to_string(), json!(format!("{} draft (ref: {} = {})", target_type, search_field, link_value)));
                                }
                                let _ = db.upsert_item(
                                    "items", &draft_id, target_type, draft_data, None,
                                    None,
                                    Some(from_addr), Some(&team_id), Some(&hashed_cc),
                                    Some(&crate::utils::hash::hash_id(&format!("{}{}", target_type, hashed_cc))),
                                    Some(&ref_val), None
                                ).await;
                                emit_term(&format!(
                                    "  📝 [TRADE RELAY v4] {} draft '{}' 생성 ({}: '{}').",
                                    target_type, draft_id, search_field, link_value
                                ));
                            },
                        }
                    }
                }

                // 🌟 [CRITICAL FIX] 이미지 데이터 저장 직후, DB의 Task와 Message 상태도 9(DONE)로 완전히 굳혀버립니다!
                // 🌟 [RELAY v4 SUMMARY] plan_trade_relays 기반 집계로 교체합니다.
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
                // 이 두 줄이 없어서 3초마다 UI가 이전 상태(1)를 DB에서 퍼와 덮어씌우고 있었습니다.
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
        
        gen.generate(
            params, 
            cancellation_token.clone(),
            session_id, // 🌟 SSD 저장 및 병합 캐시 활성화!
            Some("inference".to_string()),
            None, // 🌟 5번째 인자인 ignore_list 자리에 None을 명시적으로 추가합니다.
            semantic_prejudice  // 🌟 변경
        ).await.map_err(|e| anyhow!("Qwen 3.5 Inference failed: {}", e))
    }
}
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use serde_json::{json, Value};
use tauri::Emitter;

impl crate::model::LogisModel {

    // [신규] Analytic 파이프라인 (임시 Dummy 함수)
    pub async fn parse_analytic_query(&self, task_id: &str, app_handle: &tauri::AppHandle, query: String, language: &str, cancel_token: Arc<AtomicBool>) -> anyhow::Result<Value> {
        let app_handle_clone = app_handle.clone();
        let task_id_clone = task_id.to_string();
        let emit_term = move |msg: &str| {
            println!("{}", msg);
            let m = msg.to_string();
            let handle = app_handle_clone.clone();
            let tid = task_id_clone.clone();
            tokio::spawn(async move {
                use tauri::Emitter;
                let _ = handle.emit("task-console-log", serde_json::json!({"task_id": tid, "text": format!("{}\n", m)}));
            });
        };

        emit_term("\n=======================================");
        emit_term("[ENGINE] 🚀 Starting Analytic Search Pipeline (Draft Mode)...");

        // UI에 스피너 표기
        let payload = json!({ "task_id": task_id, "category": "Analytic", "summary": "Running mock analytics...", "spinner": "⠋" });
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::utils::logger::log_task_progress(app_handle, task_id, &payload);

        // 🌟 취소 버튼 즉시 반응 대응
        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) {
            emit_term("[ENGINE] 🛑 Task cancelled by user. Terminating safely.");
            return Ok(json!({ "context": [], "cancelled": true }));
        }

        // 🌟 [ANALYTIC CONTEXT v4]
        //  ── 무엇이 고쳐졌나 ──
        //   기존 더미는 type 을 "sales" 로 고정했습니다. 그런데
        //     · build_scope_filter 가 여기에 mode = 'analytic' 을 AND 로 붙이고
        //     · analytic 문서의 type 은 click / hover / change / report 뿐이라
        //   최종 SQL 이 `type = 'sales' AND mode = 'analytic'` 이 되어
        //   analytic 로컬 검색은 구조적으로 항상 0건이었습니다.
        //
        //  ── 왜 LLM 을 부르지 않는가 ──
        //   analytic 문서의 본문은 Cron Worker 가 이미 자연어 한 문장으로 구조화한
        //   action / summary 입니다. reindex_pending_embeddings 도 그 문장을
        //   그대로 벡터화합니다(analytic_text 우선). 질의도 문장, 저장도 문장이므로
        //   FTS + 벡터만으로 리콜이 성립하고, 조건화할 도메인 컬럼 자체가 없습니다.
        //   여기서 0.6B 를 부르면 VRAM 만 쓰고 얻는 것이 없습니다.
        emit_term(&format!("[STAGE-1] Building analytic context (no LLM required) for: '{}'", query));

        // 🌟 analytics D1(console-logis-center)이 실제로 발행하는 전 타입입니다.
        //    question / answer 는 채팅 말풍선 전용이라 검색 스코프에서 제외합니다.
        //    (main.ts 의 TYPE_SETS.analytic 과 반드시 같은 집합이어야 합니다)
        let analytic_types = vec!["report", "click", "hover", "change"];

        // 🌟 [UNASSIGNED] 질의 토큰을 그대로 넘겨 Dexie 가 keywords 가산점으로 씁니다.
        //    조건이 0개인 트랙이므로 이 축이 사실상 유일한 정밀도 신호입니다.
        let keywords: Vec<String> = query
            .split_whitespace()
            .map(|s| s.to_string())
            .collect();

        let ctx = json!([{
            "type": "report",
            "types": analytic_types,
            "text": query.clone(),
            "status": "",
            "substantial": "",
            "find": "",
            "condition": {},
            "alternates": {},
            "unassigned": keywords,
            "tier": "ANALYTIC"
        }]);

        let payload_done = json!({ "task_id": task_id, "category": "Done", "summary": "Analytic processing complete (Dummy).", "spinner": "✅" });
        let _ = app_handle.emit("extraction-progress", &payload_done);
        crate::utils::logger::log_task_progress(app_handle, task_id, &payload_done);

        emit_term("[SUCCESS] Analytic Search Pipeline Completed.");
        Ok(json!({ "context": ctx }))
    }

    /// 🌟 [ANALYTIC SEARCH QUERY v3 / VECTOR-FIRST NMS]
    ///  ── v2 의 결함 ──
    ///   time_intent / event_types 를 Qwen3.5 2B 가 '단독으로' 골랐습니다.
    ///   프롬프트에는 후보 목록만 있고 벡터 근거가 하나도 없었기 때문에,
    ///   실측 로그에서
    ///     · 질의에 기간 표현이 없는데 time_intent = "last_month" 로 창작되어
    ///       created_at >= 1782864000000 조건이 붙고 오늘 수집한 문서가 전량 탈락
    ///     · event_types 가 ["click"] 하나로 좁혀져
    ///       방금 구조화한 hover 3건 + report 1건이 스코프에서 제거
    ///   되면서 검색 결과가 구조적으로 0건이 되었습니다.
    ///
    ///  ── v3 구조 (commerce PLINKO 와 동일 계보) ──
    ///   ① 완전일치        : bias.json exact_match. 벡터·LLM 없이 확정
    ///   ② Stanza POS      : VERB/ADP/PUNCT 등 무의미 품사 판정 (NLP 모델)
    ///   ③ 슬라이딩 윈도우 : 1~4단어 청크 생성
    ///   ④ Max-Pool 코사인 : time / season / event 3개 뱅크와 구 단위 비교
    ///   ⑤ SURPRISAL 게이트: 뱅크 크기 편향 제거(√(2 ln N)).
    ///                       무작위 기대치를 못 넘으면 폐기 → '근거 없으면 조건 없음'
    ///   ⑥ NMS 배틀        : 겹치는 스팬 중 최고 점수만 생존
    ///   ⑦ 마진 판정        : 1위-2위가 사실상 동률일 때만 LLM 1회 재판정
    ///   ⑧ 기간 재확정      : LLM 이 준 날짜는 신뢰하지 않고 Rust 가 epoch 로 계산
    ///
    ///  ── 비용 ──
    ///   대부분의 질의에서 LLM 호출이 0회가 됩니다.
    ///   (임베딩 배치 3회 + Stanza POS 1회로 종료)
    ///   그리고 근거가 없으면 조건을 만들지 않으므로 스코프가 좁아지지 않습니다.
    pub async fn parse_analytic_search_query(
        &self,
        task_id: &str,
        app_handle: &tauri::AppHandle,
        query: String,
        language: &str,
        cancel: Arc<AtomicBool>,
    ) -> anyhow::Result<serde_json::Value> {
        use crate::utils::ai_utils::cosine_similarity as cos;

        let app_handle_clone = app_handle.clone();
        let tid = task_id.to_string();
        let emit_term = move |msg: &str| {
            println!("{}", msg);
            let _ = app_handle_clone.emit(
                "task-console-log",
                json!({ "task_id": tid, "text": format!("{}\n", msg) }),
            );
        };

        emit_term("\n[ANALYTIC-QUERY] 🔍 행동 로그 질의 파싱 시작 (v3 / Vector-First NMS)");
        emit_term(&format!("   질의: \"{}\"", query));

        let now_ms = chrono::Utc::now().timestamp_millis();
        let current_iso = chrono::DateTime::from_timestamp_millis(now_ms)
            .map(|dt| dt.naive_utc().format("%Y-%m-%dT%H:%M:%S").to_string())
            .unwrap_or_default();

        // =====================================================================
        // STEP 1 : bias.json 완전일치 + 접두일치 (벡터·LLM 없이 확정)
        // =====================================================================
        let (deterministic_time, _) = crate::parsing::get_deterministic_time_guide(&query, language);
        let (det_time_key, det_season_key) = crate::analytic::deterministic_time_keys(&query);
        let mut det_event_keys: Vec<String> = Vec::new();
        for w in query.split_whitespace() {
            // ① 완전일치
            if let Some(k) = crate::analytic::event_type_exact_match(w) {
                if !det_event_keys.iter().any(|e| e == &k) {
                    det_event_keys.push(k);
                }
                continue;
            }
            // 🌟 ② 접두일치 (교착어 대응)
            //    "클릭한게" 는 exact_match 에 없지만 "클릭" 이 접두이므로 click 으로 확정됩니다.
            if let Some(k) = crate::analytic::event_type_prefix_key(w) {
                if !det_event_keys.iter().any(|e| e == &k) {
                    emit_term(&format!(
                        "   ⚡ [EVENT PREFIX MATCH] '{}' → analytic_event_filters.{} (교착어 접두 확정)",
                        w, k
                    ));
                    det_event_keys.push(k);
                }
            }
        }
        if !det_time_key.is_empty() || !det_season_key.is_empty() || !det_event_keys.is_empty() {
            emit_term(&format!(
                "   ⚡ [EXACT MATCH] bias.json 일치 확정: time='{}' | season='{}' | events={:?}",
                det_time_key, det_season_key, det_event_keys
            ));
        }

        // =====================================================================
        // STEP 2 : Stanza 형태소 토큰화 (UPOS + Lemma)
        // =====================================================================
        let stanza_code = crate::analytic::stanza_lang_code(language);
        let tokens = crate::analytic::tokenize_query_with_morphology(&query, stanza_code).await;
        if tokens.is_empty() {
            emit_term("   ⚠️ [TOKENIZE] 분석 가능한 토큰이 없습니다. 전체 스코프로 검색합니다.");
        } else {
            emit_term(&format!(
                "   🧠 [STANZA POS] {:?}",
                tokens
                    .iter()
                    .map(|(w, t, l)| {
                        let tag = if t.is_empty() { "-".to_string() } else { t.clone() };
                        let lem = if l.is_empty() { "-".to_string() } else { l.clone() };
                        format!("{}(tag:{}, lemma:{})", w, tag, lem)
                    })
                    .collect::<Vec<_>>()
            ));
        }

        // 🌟 [DUAL AXIS] 드롭 대상 품사도 '청크 후보' 에는 남깁니다.
        //    '클릭한' 이 VERB 로 판정되어도 그것이 event 판정의 유일한 근거이기 때문입니다.
        //    드롭은 keywords / target 산출 시에만 적용합니다.
        const DROP_TAGS: [&str; 7] = ["VERB", "ADP", "PUNCT", "PART", "SCONJ", "CCONJ", "PRON"];
        let all_words: Vec<String> = tokens.iter().map(|(w, _, _)| w.clone()).collect();
        let content_flags: Vec<bool> = tokens
            .iter()
            .map(|(_, t, _)| !DROP_TAGS.iter().any(|d| d == t))
            .collect();

        // =====================================================================
        // 🌟 STEP 2-B : 형태소 변형 후보 산출 (교착어 원형 복원)
        // ---------------------------------------------------------------------
        //  commerce 는 같은 질의의 다른 토큰과 접두를 공유한다는 사실로 어간을 얻지만
        //  ("제품중에서" ↔ "제품"), analytic 질의는 형제 토큰이 없는 경우가 대부분입니다.
        //  Lemma → bias.json 접두 사전 → 문자 접두 n-gram 순서로 원형을 복원해
        //  슬라이딩 윈도우에 '검색 가능한 형태'를 함께 올립니다.
        // =====================================================================
        let morph_alts: Vec<Vec<String>> = tokens
            .iter()
            .map(|(w, _, l)| crate::analytic::morphological_variants(w, l))
            .collect();
        let morph_depth: usize = morph_alts.iter().map(|v| v.len()).max().unwrap_or(0);
        {
            let logged: Vec<String> = tokens
                .iter()
                .enumerate()
                .filter(|(i, _)| !morph_alts[*i].is_empty())
                .map(|(i, (w, _, _))| format!("{} → {:?}", w, morph_alts[i]))
                .collect();
            if logged.is_empty() {
                emit_term("   ⚪ [MORPH VARIANT] 원형 복원 후보가 없어 표면형만 사용합니다.");
            } else {
                emit_term(&format!("   ✂️ [MORPH VARIANT] 교착어 원형 복원: {:?}", logged));
            }
        }

        // =====================================================================
        // STEP 3 : 슬라이딩 윈도우 청크 생성 (1~6 단어) + 형태소 변형 청크
        //   Commerce의 2~8 윈도우를 참고하되, analytic은 짧은 질의가 많으므로
        //   1단어부터 시작하여 최대 6단어까지 확장합니다.
        //   🌟 같은 스팬(s,e)에 대해 '표면형 조합'과 '원형 조합'을 모두 올립니다.
        //      스팬이 동일하므로 NMS 배틀 / consumed 좌표계가 전혀 흔들리지 않습니다.
        // =====================================================================
        let mut chunk_texts: Vec<String> = Vec::new();
        let mut chunk_spans: Vec<(usize, usize)> = Vec::new();
        let mut seen_chunk: std::collections::HashSet<String> = std::collections::HashSet::new();

        for s in 0..all_words.len() {
            let max_e = all_words.len().min(s + 6);
            for e in (s + 1)..=max_e {
                // ① 표면형 조합
                let surface = all_words[s..e].join(" ");
                if !surface.trim().is_empty() {
                    let key = format!("{}|{}|{}", s, e, surface);
                    if seen_chunk.insert(key) {
                        chunk_texts.push(surface);
                        chunk_spans.push((s, e));
                    }
                }

                // ② 형태소 변형 조합 (깊이별로 순회)
                for d in 0..morph_depth {
                    let mut changed = false;
                    let mut parts: Vec<String> = Vec::with_capacity(e - s);
                    for i in s..e {
                        match morph_alts[i].get(d) {
                            Some(m) => {
                                changed = true;
                                parts.push(m.clone());
                            }
                            None => parts.push(all_words[i].clone()),
                        }
                    }
                    if !changed {
                        continue;
                    }
                    let mt = parts.join(" ");
                    if mt.trim().is_empty() {
                        continue;
                    }
                    let key = format!("{}|{}|{}", s, e, mt);
                    if seen_chunk.insert(key) {
                        chunk_texts.push(mt);
                        chunk_spans.push((s, e));
                    }
                }
            }
        }

        // =====================================================================
        // STEP 4 : 뱅크 구축 + 임베딩 (time / season / event)
        // =====================================================================
        self.check_embedding_downloaded().await?;
        self.ensure_embedding().await?;

        let mut bank_defs: Vec<(String, String, String)> =
            crate::utils::ai_utils::filter_category_phrases(&["time_filters", "season_filters"]);
        let mut prej_defs: Vec<(String, String, String)> =
            crate::utils::ai_utils::filter_category_prejudice_phrases(&[
                "time_filters",
                "season_filters",
            ]);
        // 🌟 [EVENT TYPE BANK] click/hover/change/report 도 슬라이딩 윈도우 NMS 배틀에 함께 올립니다.
        //    기존에는 질의 전체 벡터 1개로만 판정하여
        //    '클릭한거 뭐야' 에서 '클릭' 이 '뭐야' 와 섞여 신호가 희석되었습니다.
        //    슬라이딩 윈도우에 올리면 각 단어 윈도우가 독립적으로 경쟁하므로 이 문제가 없습니다.
        //
        //    🌟 [DUPLICATE LOOP FIX] 기존 코드는 동일한 적재 루프를 두 번 실행하여
        //    event 각 키의 구 개수 N 이 정확히 2배가 되어 있었습니다.
        //    SURPRISAL 은 √(2 ln N) 을 차감하므로 N 이 2배가 되면 event 키만
        //    구조적으로 불리해져(√(2 ln 24)=2.78 vs √(2 ln 12)=2.23) time/season 에 밀립니다.
        //    중복 삽입을 제거하고, 혹시 모를 사전 중복까지 구조적으로 차단합니다.
        for event_type in crate::analytic::ANALYTIC_SEARCH_TYPES.iter() {
            for p in crate::analytic::event_type_anchor_phrases(event_type) {
                if bank_defs
                    .iter()
                    .any(|(c, k, e)| c == "event" && k == event_type && e == &p)
                {
                    continue;
                }
                bank_defs.push(("event".to_string(), event_type.to_string(), p));
            }
            for p in crate::analytic::event_type_prejudice_phrases(event_type) {
                if prej_defs
                    .iter()
                    .any(|(c, k, e)| c == "event" && k == event_type && e == &p)
                {
                    continue;
                }
                prej_defs.push(("event".to_string(), event_type.to_string(), p));
            }
        }

        emit_term(&format!(
            "   📐 [BANK] 판정 구 {}개 | 편견 구 {}개 | 청크 후보 {}개",
            bank_defs.len(),
            prej_defs.len(),
            chunk_texts.len()
        ));

        let bank_texts: Vec<String> = bank_defs.iter().map(|(_, _, p)| p.clone()).collect();
        let prej_texts: Vec<String> = prej_defs.iter().map(|(_, _, p)| p.clone()).collect();

        let bank_embs: Vec<Vec<f32>> = if bank_texts.is_empty() {
            Vec::new()
        } else {
            let mut acc: Vec<Vec<f32>> = Vec::with_capacity(bank_texts.len());
            for part in bank_texts.chunks(200) {
                let e = self
                    .get_embedding_batch(part.to_vec())
                    .await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; part.len()]);
                acc.extend(e);
            }
            acc
        };
        let prej_embs: Vec<Vec<f32>> = if prej_texts.is_empty() {
            Vec::new()
        } else {
            let mut acc: Vec<Vec<f32>> = Vec::with_capacity(prej_texts.len());
            for part in prej_texts.chunks(200) {
                let e = self
                    .get_embedding_batch(part.to_vec())
                    .await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; part.len()]);
                acc.extend(e);
            }
            acc
        };
        let chunk_embs: Vec<Vec<f32>> = if chunk_texts.is_empty() {
            Vec::new()
        } else {
            self.get_embedding_batch(chunk_texts.clone())
                .await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; chunk_texts.len()])
        };

        // (category, key) 단위로 뱅크 인덱스를 묶습니다.
        let mut key_bank: std::collections::HashMap<(String, String), Vec<usize>> =
            std::collections::HashMap::new();
        for (i, (c, k, _)) in bank_defs.iter().enumerate() {
            key_bank.entry((c.clone(), k.clone())).or_default().push(i);
        }
        let mut key_prej: std::collections::HashMap<(String, String), Vec<usize>> =
            std::collections::HashMap::new();
        for (i, (c, k, _)) in prej_defs.iter().enumerate() {
            key_prej.entry((c.clone(), k.clone())).or_default().push(i);
        }

        // =====================================================================
        // STEP 5 : SURPRISAL 채점 (전역 기준선)
        //   surprisal = (max - μ_global)/σ_global - √(2 ln N)
        //
        //  ── 왜 전역 기준선으로 바꾸는가 ──
        //   직전 구현은 각 (category,key) 뱅크를 '자기 자신의' 평균/표준편차로 표준화했습니다.
        //   그런데 극값이론이 E[z of max] ≈ √(2 ln N) 을 예측하므로, 그 값을 다시 빼면
        //   질의와의 관련성과 무관하게 결과가 항상 0 부근으로 수렴합니다(판별력 0).
        //   (로그 실측: 청크 10개 전부 게이트 미통과 → [NMS CANDIDATE] 0건 →
        //    EVENT FALLBACK 4종 전체 → 의도한 click 스코프 축소가 영구 실패)
        //   ai_utils::surprisal_dual_scores 는 이미 같은 문제를 '모든 뱅크를 합친 전역 분포'
        //   기준선으로 해결해 두었으므로 동일 원리를 이식합니다.
        //   0 은 극값이론에서 유도된 값이므로 매직 상수가 아닙니다.
        // =====================================================================
        let global_baseline = |q: &Vec<f32>, embs: &Vec<Vec<f32>>| -> (f32, f32) {
            let mut pool: Vec<f32> = Vec::with_capacity(embs.len());
            for e in embs {
                if e.iter().all(|&v| v == 0.0) {
                    continue;
                }
                pool.push(cos(q, e));
            }
            if pool.len() < 2 {
                return (0.0f32, 1.0f32);
            }
            let n = pool.len() as f32;
            let mean: f32 = pool.iter().sum::<f32>() / n;
            let var: f32 = pool.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n;
            (mean, var.sqrt().max(1e-6))
        };

        let surprisal = |q: &Vec<f32>,
                         idxs: &Vec<usize>,
                         embs: &Vec<Vec<f32>>,
                         g_mean: f32,
                         g_sd: f32| -> (f32, f32) {
            let mut sims: Vec<f32> = Vec::with_capacity(idxs.len());
            for &i in idxs {
                let e = match embs.get(i) {
                    Some(v) => v,
                    None => continue,
                };
                if e.iter().all(|&v| v == 0.0) {
                    continue;
                }
                sims.push(cos(q, e));
            }
            if sims.is_empty() {
                return (f32::MIN, 0.0);
            }
            let n = sims.len() as f32;
            let mx = sims.iter().cloned().fold(f32::MIN, f32::max);
            let z = (mx - g_mean) / g_sd;
            let expect = (2.0 * n.max(2.0).ln()).sqrt();
            (z - expect, mx)
        };

        struct AnalyticSpan {
            start: usize,
            end: usize,
            text: String,
            category: String,
            key: String,
            score: f32,
            max_cos: f32,
            alts: Vec<(String, f32)>,
        }

        let mut candidates: Vec<AnalyticSpan> = Vec::new();
        // 🌟 [COVERAGE RESCUE POOL] 게이트를 통과하지 못했지만
        //    '전역 분포의 상위 1σ 꼬리' 에 든 최상위 후보를 따로 모아 둡니다.
        //    (통계적 사실 기반이며 새 매직 상수가 아닙니다)
        let mut rescue_pool: Vec<AnalyticSpan> = Vec::new();

        for (ci, (s, e)) in chunk_spans.iter().enumerate() {
            if cancel.load(std::sync::atomic::Ordering::Relaxed) {
                emit_term("[ANALYTIC-QUERY] 🛑 사용자 취소로 파싱을 중단합니다.");
                break;
            }
            let q = match chunk_embs.get(ci) {
                Some(v) => v,
                None => continue,
            };
            if q.iter().all(|&v| v == 0.0) {
                continue;
            }

            // 🌟 이 청크 하나에 대한 전역 기준선을 먼저 확정합니다.
            let (g_mean, g_sd) = global_baseline(q, &bank_embs);

            let mut scored: Vec<(String, String, f32, f32)> = Vec::new();
            for (ck, idxs) in key_bank.iter() {
                let (sur, own) = surprisal(q, idxs, &bank_embs, g_mean, g_sd);
                if sur == f32::MIN {
                    continue;
                }
                // 🌟 [PREJUDICE GATE] 경쟁 개념이 더 잘 설명하면 후보 자격 자체를 박탈합니다.
                //    점수에서 빼는 방식이 아니라 상대 우위 판정이므로 임계치가 없습니다.
                let prej = key_prej
                    .get(ck)
                    .map(|pi| {
                        let mut m = 0.0f32;
                        for &i in pi {
                            let ee = match prej_embs.get(i) {
                                Some(v) => v,
                                None => continue,
                            };
                            if ee.iter().all(|&v| v == 0.0) {
                                continue;
                            }
                            let sv = cos(q, ee);
                            if sv > m {
                                m = sv;
                            }
                        }
                        m
                    })
                    .unwrap_or(0.0);
                if prej >= own {
                    continue;
                }
                scored.push((ck.0.clone(), ck.1.clone(), sur, own));
            }
            if scored.is_empty() {
                continue;
            }
            scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            let (cat, key, sc, own) = scored[0].clone();

            let alts: Vec<(String, f32)> = scored
                .iter()
                .skip(1)
                .filter(|(c, _, _, _)| c == &cat)
                .take(3)
                .map(|(_, k, s, _)| (k.clone(), *s))
                .collect();

            // 🌟 [MULTI-CATEGORY COLLECT] Commerce의 intersecting_categories 처럼
            //    동일 카테고리 내 차순위뿐 아니라 타 카테고리 후보도 수집합니다.
            let mut multi_cats: Vec<(String, String, f32)> = Vec::new();
            multi_cats.push((cat.clone(), key.clone(), sc));
            for (c2, k2, s2, _) in scored.iter().skip(1).take(5) {
                if *c2 != cat && *s2 > 0.0 {
                    multi_cats.push((c2.clone(), k2.clone(), *s2));
                }
            }

            let span = AnalyticSpan {
                start: *s,
                end: *e,
                text: chunk_texts[ci].clone(),
                category: cat.clone(),
                key: key.clone(),
                score: sc,
                max_cos: own,
                alts,
            };

            // 🌟 [SURPRISAL GATE] 전역 기준선으로 바뀌었으므로 0 이 다시 의미를 갖습니다.
            //    surprisal > 0 = "N개를 무작위로 뽑은 기대 최댓값보다 실제로 더 가깝다"
            if sc > 0.0 {
                emit_term(&format!(
                    "   🎯 [NMS CANDIDATE] \"{}\" → {}.{} | Surprisal: {:+.4} | MaxCos: {:.4} | μ:{:.4} σ:{:.4} | MultiCats: {}",
                    chunk_texts[ci], cat, key, sc, own, g_mean, g_sd, multi_cats.len()
                ));
                candidates.push(span);
            } else if own > g_mean + g_sd {
                emit_term(&format!(
                    "   🛟 [RESCUE POOL] \"{}\" → {}.{} | Surprisal: {:+.4} (게이트 미통과) | MaxCos: {:.4} > μ+σ({:.4})",
                    chunk_texts[ci], cat, key, sc, own, g_mean + g_sd
                ));
                rescue_pool.push(span);
            }
        }

        // 🌟 [COVERAGE RESCUE] 게이트를 넘은 후보가 하나도 없으면
        //    '전역 분포 상위 1σ' 에 든 후보를 승격시켜 NMS 배틀을 살립니다.
        //    (승격하지 않으면 EVENT FALLBACK 4종 전체로 스코프가 무조건 넓어집니다)
        if candidates.is_empty() && !rescue_pool.is_empty() {
            rescue_pool.sort_by(|a, b| {
                b.max_cos
                    .partial_cmp(&a.max_cos)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            emit_term(&format!(
                "   🛟 [COVERAGE RESCUE] 게이트 통과 후보가 0건이라 상위 1σ 후보 {}건을 승격합니다.",
                rescue_pool.len()
            ));
            for r in rescue_pool.into_iter() {
                emit_term(&format!(
                    "      ↳ \"{}\" → {}.{} | Surprisal: {:+.4} | MaxCos: {:.4}",
                    r.text, r.category, r.key, r.score, r.max_cos
                ));
                candidates.push(r);
            }
        }

        // =====================================================================
        // STEP 6 : NMS 배틀 — Commerce의 계층적 흡수 + 갭 브리징 이식
        // =====================================================================
        candidates.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then((b.end - b.start).cmp(&(a.end - a.start)))
        });

        // 🌟 [NMS WITH ABSORPTION] Commerce의 NMS BATTLE과 동일하게,
        //    패배한 스팬의 카테고리/키 정보를 승자에게 병합합니다.
        //    이렇게 하면 '최근에 가장 많이'가 이겨도 '본게'의 이벤트 정보가
        //    승자 스팬에 흡수되어 keywords/target 산출 시 유실되지 않습니다.
        #[derive(Debug, Clone)]
        struct AbsorbedInfo {
            category: String,
            key: String,
            score: f32,
        }

        let mut winners: Vec<(AnalyticSpan, Vec<AbsorbedInfo>)> = Vec::new();
        for c in candidates.into_iter() {
            let mut is_overlapped = false;
            let mut winner_text = String::new();
            for (w, absorbed_list) in winners.iter_mut() {
                let overlaps = c.start < w.end && c.end > w.start;
                if overlaps {
                    is_overlapped = true;
                    winner_text = w.text.clone();
                    // 🌟 [ABSORPTION] 패배한 스팬의 카테고리/키를 승자에게 병합
                    let already_has = absorbed_list.iter().any(|a| a.category == c.category && a.key == c.key);
                    if !already_has {
                        absorbed_list.push(AbsorbedInfo {
                            category: c.category.clone(),
                            key: c.key.clone(),
                            score: c.score,
                        });
                        emit_term(&format!(
                            "   ♻️ [ABSORBED] \"{}\" ({}.{}) 의 정보가 승자 \"{}\" 에게 병합되었습니다.",
                            c.text, c.category, c.key, winner_text
                        ));
                    }
                    // 🌟 승자의 스팬 범위를 패배자까지 확장하여 커버리지 확보
                    if c.start < w.start {
                        w.start = c.start;
                        w.text = format!("{} {}", c.text, w.text);
                    }
                    if c.end > w.end {
                        w.end = c.end;
                        w.text = format!("{} {}", w.text, c.text);
                    }
                    if c.score > w.score {
                        w.score = c.score;
                        w.max_cos = c.max_cos;
                    }
                    break;
                }
            }
            if !is_overlapped {
                emit_term(&format!(
                    "   👑 [NMS WINNER] \"{}\" → {}.{} | Surprisal: {:+.4} | MaxCos: {:.4}",
                    c.text, c.category, c.key, c.score, c.max_cos
                ));
                winners.push((c, Vec::new()));
            } else {
                emit_term(&format!(
                    "   💀 [NMS DEFEAT] \"{}\" ({}.{}) 는 상위 스팬 '{}' 에 흡수되었습니다.",
                    c.text, c.category, c.key, winner_text
                ));
            }
        }

        // 🌟 [GAP BRIDGING] Commerce의 4차 패스와 동일하게,
        //    NMS에서 커버되지 않은 고아 단어를 승자 스팬에 흡수합니다.
        if !winners.is_empty() {
            emit_term("   🌉 [GAP BRIDGING] 고아 단어 구출 시작...");
            // 승자들을 시작 위치순으로 정렬
            winners.sort_by(|a, b| a.0.start.cmp(&b.0.start));

            // 왼쪽 끝 고아 단어 흡수
            if winners[0].0.start > 0 {
                let gap_text = all_words[0..winners[0].0.start].join(" ");
                emit_term(&format!(
                    "   🛠️ [LEFT EDGE] '{}' → '{}' 에 흡수",
                    gap_text, winners[0].0.text
                ));
                winners[0].0.start = 0;
                winners[0].0.text = format!("{} {}", gap_text, winners[0].0.text);
            }

            // 중간 갭 흡수 (양방향 점수 대결)
            for i in 0..(winners.len().saturating_sub(1)) {
                let gap_start = winners[i].0.end;
                let gap_end = winners[i + 1].0.start;
                if gap_start < gap_end {
                    let gap_text = all_words[gap_start..gap_end].join(" ");
                    // 왼쪽 승자가 흡수 (간단히 왼쪽 우선, Commerce의 양방향 대결 간소화)
                    emit_term(&format!(
                        "   ⚔️ [GAP BATTLE] '{}' → LEFT '{}' 에 흡수",
                        gap_text, winners[i].0.text
                    ));
                    winners[i].0.end = gap_end;
                    winners[i].0.text = format!("{} {}", winners[i].0.text, gap_text);
                }
            }

            // 오른쪽 끝 고아 단어 흡수
            let last_idx = winners.len() - 1;
            if winners[last_idx].0.end < all_words.len() {
                let gap_text = all_words[winners[last_idx].0.end..].join(" ");
                emit_term(&format!(
                    "   🛠️ [RIGHT EDGE] '{}' → '{}' 에 흡수",
                    gap_text, winners[last_idx].0.text
                ));
                winners[last_idx].0.end = all_words.len();
                winners[last_idx].0.text = format!("{} {}", winners[last_idx].0.text, gap_text);
            }
        }

        // winners를 기존 인터페이스에 맞게 분리
        let absorbed_infos: Vec<Vec<AbsorbedInfo>> = winners.iter().map(|(_, a)| a.clone()).collect();
        let winners: Vec<AnalyticSpan> = winners.into_iter().map(|(w, _)| w).collect();

        // =====================================================================
        // STEP 7 : 카테고리별 확정 + 소비 스팬 표시 + 흡수 정보 반영
        // =====================================================================
        let mut vec_time = String::new();
        let mut vec_time_score = f32::MIN;
        let mut vec_time_alts: Vec<(String, f32)> = Vec::new();
        let mut vec_time_text = String::new();
        let mut vec_season = String::new();
        let mut vec_season_score = f32::MIN;
        let mut vec_season_alts: Vec<(String, f32)> = Vec::new();
        let mut vec_season_text = String::new();
        let mut vec_events: Vec<(String, f32)> = Vec::new();
        let mut consumed: Vec<bool> = vec![false; all_words.len()];

        // 🌟 [ABSORBED INFO PROCESSING] NMS에서 흡수된 카테고리 정보도 반영합니다.
        //    이렇게 하면 '최근에 가장 많이'가 이기고 '본게 뭐야?'가 흡수되어도
        //    '본게 뭐야?'가 갖고 있던 이벤트 정보가 유실되지 않습니다.
        for (wi, absorbed_list) in absorbed_infos.iter().enumerate() {
            for ai in absorbed_list {
                match ai.category.as_str() {
                    "time_filters" => {
                        if ai.score > vec_time_score && vec_time.is_empty() {
                            vec_time_score = ai.score;
                            vec_time = ai.key.clone();
                            vec_time_text = winners.get(wi).map(|w| w.text.clone()).unwrap_or_default();
                        }
                    }
                    "season_filters" => {
                        if ai.score > vec_season_score && vec_season.is_empty() {
                            vec_season_score = ai.score;
                            vec_season = ai.key.clone();
                            vec_season_text = winners.get(wi).map(|w| w.text.clone()).unwrap_or_default();
                        }
                    }
                    "event" => {
                        if !vec_events.iter().any(|(k, _)| k == &ai.key) {
                            vec_events.push((ai.key.clone(), ai.score));
                            emit_term(&format!(
                                "   ♻️ [ABSORBED EVENT] 흡수된 '{}' → '{}' 이벤트 타입이 스코프에 포함되었습니다.",
                                winners.get(wi).map(|w| w.text.as_str()).unwrap_or("?"),
                                ai.key
                            ));
                        }
                    }
                    _ => {}
                }
            }
        }

        for w in &winners {
            match w.category.as_str() {
                "time_filters" => {
                    if w.score > vec_time_score {
                        vec_time_score = w.score;
                        vec_time = w.key.clone();
                        vec_time_alts = w.alts.clone();
                        vec_time_text = w.text.clone();
                    }
                }
                "season_filters" => {
                    if w.score > vec_season_score {
                        vec_season_score = w.score;
                        vec_season = w.key.clone();
                        vec_season_alts = w.alts.clone();
                        vec_season_text = w.text.clone();
                    }
                }
                "event" => {
                    if !vec_events.iter().any(|(k, _)| k == &w.key) {
                        vec_events.push((w.key.clone(), w.score));
                    }
                    // 🌟 [EVENT SIBLING RESCUE] 1위와 사실상 동률인 차순위 타입도 함께 살립니다.
                    //    '클릭' 이 hover 와 0.9 이상 근접하면 둘 다 스코프에 넣는 편이
                    //    조건으로 잘라내는 Dexie 구조상 리콜에 유리합니다.
                    for (ak, asc) in w.alts.iter() {
                        if *asc >= w.score * 0.9 && *asc > 0.0 {
                            if !vec_events.iter().any(|(k, _)| k == ak) {
                                vec_events.push((ak.clone(), *asc));
                                emit_term(&format!(
                                    "   🤝 [EVENT SIBLING] \"{}\" 의 차순위 '{}' 이 사실상 동률({:+.4} vs {:+.4})이라 함께 포함합니다.",
                                    w.text, ak, asc, w.score
                                ));
                            }
                        }
                    }
                }
                _ => {}
            }
            for i in w.start..w.end {
                if i < consumed.len() {
                    consumed[i] = true;
                }
            }
        }

        // =====================================================================
        // STEP 8 : 마진 부족 시에만 LLM 1회 재판정
        //   (전체 파싱이 아니라 '이미 좁혀진 후보 중 선택' 이므로 창작 여지가 없습니다)
        // =====================================================================
        let mut time_intent = if !det_time_key.is_empty() {
            det_time_key.clone()
        } else {
            vec_time.clone()
        };
        let mut season_intent = if !det_season_key.is_empty() {
            det_season_key.clone()
        } else {
            vec_season.clone()
        };

        let time_context = format!(
            "- Current UTC time is \"{}\" (epoch ms {}).\n- The user locale language is \"{}\".\n{}",
            current_iso, now_ms, language, deterministic_time
        );

        let need_time_llm = det_time_key.is_empty()
            && !vec_time.is_empty()
            && vec_time_alts
                .first()
                .map_or(false, |(_, s)| *s >= vec_time_score * 0.9);
        let need_season_llm = det_season_key.is_empty()
            && !vec_season.is_empty()
            && vec_season_alts
                .first()
                .map_or(false, |(_, s)| *s >= vec_season_score * 0.9);

        if need_time_llm || need_season_llm {
            emit_term(&format!(
                "   ⚖️ [MARGIN GATE] 1위-2위가 사실상 동률이라 LLM 재판정을 1회 수행합니다. (time: {} | season: {})",
                need_time_llm, need_season_llm
            ));
            self.secure_vram_relay(
                crate::model::ModelSize::Qwen3_5,
                None,
                Some(cancel.clone()),
                false,
                None,
            )
            .await?;

            if need_time_llm {
                let p = crate::parsing::extract_time_intent_prompt(
                    &vec_time_text,
                    &time_context,
                    &vec_time,
                    vec_time_score,
                    &vec_time_alts,
                );
                let params = crate::openai_types::ChatCompletionParameters {
                    messages: vec![crate::openai_types::ChatCompletionRequestMessage::User(
                        crate::openai_types::ChatCompletionRequestUserMessage {
                            content:
                                crate::openai_types::ChatCompletionRequestUserMessageContent::Text(p),
                            name: None,
                        },
                    )],
                    model: "qwen3.5".to_string(),
                    max_tokens: Some(128),
                    temperature: Some(0.0),
                    top_p: Some(0.95),
                    ..Default::default()
                };
                let r = if let Some(gen) = self.qwen3_5_generator.lock().await.as_mut() {
                    gen.generate(
                        params,
                        Some(cancel.clone()),
                        Some(format!("{}_aq_time", task_id)),
                        None,
                        None,
                        None,
                    )
                    .await
                    .unwrap_or_default()
                } else {
                    String::new()
                };
                let picked = crate::parsing::parse_json_from_llm(&r)
                    .get("time_intent")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let allowed = picked == vec_time
                    || vec_time_alts.iter().any(|(k, _)| k == &picked)
                    || picked.is_empty();
                if allowed {
                    emit_term(&format!(
                        "   🤖 [TIME LLM] 후보 중 '{}' 로 확정했습니다.",
                        if picked.is_empty() { "(없음)" } else { &picked }
                    ));
                    time_intent = picked;
                } else {
                    emit_term(&format!(
                        "   🚫 [TIME LLM REJECT] '{}' 는 벡터 후보 목록에 없어 폐기하고 '{}' 를 유지합니다.",
                        picked, vec_time
                    ));
                }
            }

            if need_season_llm {
                let p = crate::parsing::extract_season_intent_prompt(
                    &vec_season_text,
                    &vec_season,
                    vec_season_score,
                    &vec_season_alts,
                );
                let params = crate::openai_types::ChatCompletionParameters {
                    messages: vec![crate::openai_types::ChatCompletionRequestMessage::User(
                        crate::openai_types::ChatCompletionRequestUserMessage {
                            content:
                                crate::openai_types::ChatCompletionRequestUserMessageContent::Text(p),
                            name: None,
                        },
                    )],
                    model: "qwen3.5".to_string(),
                    max_tokens: Some(128),
                    temperature: Some(0.0),
                    top_p: Some(0.95),
                    ..Default::default()
                };
                let r = if let Some(gen) = self.qwen3_5_generator.lock().await.as_mut() {
                    gen.generate(
                        params,
                        Some(cancel.clone()),
                        Some(format!("{}_aq_season", task_id)),
                        None,
                        None,
                        None,
                    )
                    .await
                    .unwrap_or_default()
                } else {
                    String::new()
                };
                let picked = crate::parsing::parse_json_from_llm(&r)
                    .get("season_intent")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .trim()
                    .to_string();
                let allowed = picked == vec_season
                    || vec_season_alts.iter().any(|(k, _)| k == &picked)
                    || picked.is_empty();
                if allowed {
                    emit_term(&format!(
                        "   🤖 [SEASON LLM] 후보 중 '{}' 로 확정했습니다.",
                        if picked.is_empty() { "(없음)" } else { &picked }
                    ));
                    season_intent = picked;
                } else {
                    emit_term(&format!(
                        "   🚫 [SEASON LLM REJECT] '{}' 는 벡터 후보 목록에 없어 폐기하고 '{}' 를 유지합니다.",
                        picked, vec_season
                    ));
                }
            }
        } else if vec_time.is_empty() && vec_season.is_empty() {
            // 🌟 [LOG FIX] 기존에는 후보가 0건이어도 "벡터 마진이 충분하여" 라고 출력해
            //    NMS CANDIDATE 0건이라는 실제 상태를 로그에서 은폐했습니다.
            //    마진 판정은 후보가 존재할 때만 성립합니다.
            emit_term("   ⚪ [NO VECTOR EVIDENCE] 시간/계절 벡터 후보가 0건이라 LLM 재판정 대상이 없습니다.");
        } else {
            emit_term("   ⚡ [DETERMINISTIC] 벡터 마진이 충분하여 LLM 호출을 생략합니다.");
        }

        // =====================================================================
        // STEP 9 : 이벤트 타입 확정
        //   근거가 하나도 없으면 4종 전체를 스코프로 둡니다.
        //   ('없으면 넓게' 가 리콜 우선 원칙이며, 정밀 필터는 Dexie 가 담당합니다)
        // =====================================================================
        let mut event_types: Vec<String> = if !det_event_keys.is_empty() {
            det_event_keys.clone()
        } else {
            vec_events.iter().map(|(k, _)| k.clone()).collect()
        };
        if event_types.is_empty() {
            event_types = crate::analytic::ANALYTIC_SEARCH_TYPES
                .iter()
                .map(|s| s.to_string())
                .collect();
            emit_term(
                "   🛟 [EVENT FALLBACK] 이벤트 종류를 특정할 벡터 근거가 없어 4종 전체를 스코프로 둡니다.",
            );
        } else {
            emit_term(&format!(
                "   ✅ [EVENT CONFIRMED] event_types = {:?}",
                event_types
            ));
        }

        // 🌟 [REPORT ALWAYS-IN] report 는 여러 이벤트를 합성한 문서라
        //    '가장 많이', '패턴', '흐름' 류 질의의 정답을 담고 있습니다.
        //    특정 이벤트가 확정되어도 report 는 후보에 남겨야 정답이 잘리지 않습니다.
        if !event_types.iter().any(|t| t == "report") {
            event_types.push("report".to_string());
            emit_term("   📊 [REPORT ALWAYS-IN] 합성 리포트를 스코프에 함께 포함합니다.");
        }

        // =====================================================================
        // STEP 10 : 확정 스팬을 제외한 나머지가 검색 키워드
        //   🌟 갭 브리징으로 승자 스팬이 확장되었으므로 consumed 범위가 넓어져
        //      키워드가 비는 경우가 줄어듭니다.
        //      그래도 비면 전체 질의를 target으로 사용합니다.
        // =====================================================================
        let mut keywords: Vec<String> = Vec::new();
        for (i, w) in all_words.iter().enumerate() {
            if consumed.get(i).copied().unwrap_or(false) {
                continue;
            }
            if !content_flags.get(i).copied().unwrap_or(true) {
                continue;
            }
            if !keywords.iter().any(|k| k == w) {
                keywords.push(w.clone());
            }
        }
        if keywords.is_empty() {
            // 🌟 [KEYWORD FALLBACK] consumed가 전부를 커버하면
            //    승자 스팬의 텍스트에서 조사/동사를 제외한 명사구를 키워드로 사용합니다.
            //    Commerce의 unassigned_chunks 구조와 동일합니다.
            for w in &winners {
                for word in w.text.split_whitespace() {
                    let word_str = word.trim();
                    if word_str.is_empty() { continue; }
                    // Stanza POS에서 동사/조사로 판정된 단어는 제외
                    let is_func = all_words.iter().position(|aw| aw == word_str)
                        .map(|idx| !content_flags.get(idx).copied().unwrap_or(true))
                        .unwrap_or(false);
                    if is_func { continue; }
                    if !keywords.iter().any(|k| k == word_str) {
                        keywords.push(word_str.to_string());
                    }
                }
            }
        }
        if keywords.is_empty() {
            for (i, w) in all_words.iter().enumerate() {
                if consumed.get(i).copied().unwrap_or(false) {
                    continue;
                }
                if !keywords.iter().any(|k| k == w) {
                    keywords.push(w.clone());
                }
            }
        }
        let target = if keywords.is_empty() {
            query.clone()
        } else {
            keywords.join(" ")
        };
        emit_term(&format!(
            "   🧷 [KEYWORDS] {:?} | target=\"{}\"",
            keywords, target
        ));

        // =====================================================================
        // STEP 11 : 기간을 Rust 가 epoch 로 재확정
        // =====================================================================
        let mut started_at: i64 = 0;
        let mut expired_at: i64 = 0;

        if !season_intent.is_empty() {
            let (y, _, _) = crate::analytic::ymd_of(now_ms);
            let year = if time_intent == "last_year" { y - 1 } else { y };
            if let Some((s, e)) = crate::analytic::season_range(&season_intent, year) {
                started_at = s;
                expired_at = e;
            }
        }
        if started_at == 0 {
            if let Some((s, e)) = crate::analytic::time_intent_range(&time_intent, now_ms) {
                started_at = s;
                expired_at = e;
            }
        }

        if started_at > 0 {
            emit_term(&format!(
                "   🗓️ [PERIOD CONFIRMED] time='{}' | season='{}' → {} ~ {} (epoch ms)",
                if time_intent.is_empty() { "-" } else { &time_intent },
                if season_intent.is_empty() { "-" } else { &season_intent },
                started_at,
                expired_at
            ));
        } else {
            emit_term(
                "   🗓️ [PERIOD] 벡터가 확정한 기간 표현이 없어 전체 구간을 검색합니다. (근거 없는 기간 조건을 만들지 않습니다)",
            );
        }

        // =====================================================================
        // STEP 12 : 컨텍스트 조립 (lib.rs STAGE-3 계약과 동일)
        // =====================================================================
        let mut condition = serde_json::Map::new();
        if started_at > 0 {
            condition.insert(
                "created_at".to_string(),
                json!({ "operator": "gte", "value": started_at }),
            );
        }

        let mut contexts: Vec<serde_json::Value> = Vec::new();
        contexts.push(json!({
            "text": target,
            "language": language,
            "type": event_types[0],
            "types": event_types,
            "condition": serde_json::Value::Object(condition.clone()),
            "unassigned": keywords
        }));

        if expired_at > 0 {
            let mut upper = serde_json::Map::new();
            upper.insert(
                "created_at".to_string(),
                json!({ "operator": "lte", "value": expired_at }),
            );
            contexts.push(json!({
                "text": target,
                "language": language,
                "type": event_types[0],
                "types": event_types,
                "condition": serde_json::Value::Object(upper),
                "unassigned": keywords
            }));
        }

        let out = json!({
            "original_text": query,
            "time_intent": time_intent,
            "season_intent": season_intent,
            "started_at": started_at,
            "expired_at": expired_at,
            "event_types": event_types,
            "keywords": keywords,
            "target": target,
            "context": contexts
        });

        emit_term(&format!(
            "[ANALYTIC-QUERY] ✅ 파싱 결과: {}",
            serde_json::to_string(&out).unwrap_or_default()
        ));

        Ok(out)
    }

}
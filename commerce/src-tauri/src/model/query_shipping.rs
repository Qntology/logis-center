use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use serde_json::{json, Value};
use tauri::Emitter;
use crate::model::merge::{trade_resolve_condition_value, trade_resolve_condition_operator};

impl crate::model::LogisModel {

    // 🌟 [SHIPPING QUERY v3 / VECTOR-FIRST NMS]
    //  ── v3 구조 (STEP A 와 동일 계보) ──
    //   ① 접두어 완전일치      : 'CI-2026-08001' → reference_invoice. 벡터·LLM 없이 확정
    //   ② Stanza POS 토큰화     : 무의미 품사 사전 제거 (NLP 모델)
    //   ③ 슬라이딩 윈도우       : 1~6단어 청크 생성
    //   ④ Depth 1 SURPRISAL     : 7개 조건 카테고리 채점 (편견 = 다른 카테고리 bias)
    //   ⑤ NMS 배틀 + 흡수       : 겹치는 스팬 중 최고 점수만 생존
    //   ⑥ Depth 2 배타 배정     : 승리 카테고리의 필드만 경쟁, 1청크 1필드
    //   ⑦ Depth 3 값 확정       : Rust 결정론. 실패 시에만 LLM 1회
    //  마진이 충분하면 LLM 호출이 0회로 끝납니다.
    pub async fn parse_shipping_query(&self, task_id: &str, app_handle: &tauri::AppHandle, query: String, language: &str, cancel_token: Arc<AtomicBool>) -> anyhow::Result<Value> {
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
        emit_term("[ENGINE] 🚀 Starting Shipping Search Pipeline (v3 / Vector-First NMS)...");
        emit_term(&format!("   질의: \"{}\"", query));

        let payload = json!({ "task_id": task_id, "category": "Shipping", "summary": "Segmenting trade conditions...", "spinner": "⠋" });
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::utils::logger::log_task_progress(app_handle, task_id, &payload);

        if cancel_token.load(std::sync::atomic::Ordering::Relaxed) {
            emit_term("[ENGINE] 🛑 Task cancelled by user. Terminating safely.");
            return Ok(json!({ "context": [], "cancelled": true }));
        }

        // =====================================================================
        // STEP 1 : 문서번호 접두어 완전일치 (벡터·LLM 없이 확정)
        // =====================================================================
        let mut deterministic_refs: Vec<(String, String)> = Vec::new(); // (field, value)
        let mut consumed_words: std::collections::HashSet<String> = std::collections::HashSet::new();

        for raw_word in query.split_whitespace() {
            // 조사/따옴표를 떼어낸 코어 토큰
            let core: String = raw_word
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
                .collect();
            if core.chars().count() < 4 { continue; }
            if !core.contains('-') && !core.contains('_') { continue; }
            if !core.chars().any(|c| c.is_ascii_digit()) { continue; }

            let prefix: String = core
                .chars()
                .take_while(|c| c.is_ascii_alphabetic())
                .collect::<String>()
                .to_uppercase();
            if prefix.is_empty() { continue; }

            if let Some(field) = crate::logic::trade_reference_field_of(&prefix) {
                if deterministic_refs.iter().any(|(f, _)| f == field) { continue; }
                emit_term(&format!(
                    "   ⚡ [PREFIX EXACT MATCH] '{}' → 접두어 '{}' 로 '{}' 축 확정 (벡터·LLM 생략)",
                    core, prefix, field
                ));
                deterministic_refs.push((field.to_string(), core.clone()));
                consumed_words.insert(raw_word.to_string());
            }
        }

        // =====================================================================
        // STEP 2 : Stanza 형태소 토큰화
        // =====================================================================
        let stanza_code = crate::analytic::stanza_lang_code(language);
        let tokens = crate::analytic::tokenize_query_with_morphology(&query, stanza_code).await;
        if tokens.is_empty() {
            emit_term("   ⚠️ [TOKENIZE] 분석 가능한 토큰이 없습니다.");
        } else {
            emit_term(&format!(
                "   🧠 [STANZA POS] {:?}",
                tokens.iter().map(|(w, t, l)| {
                    let tag = if t.is_empty() { "-".to_string() } else { t.clone() };
                    let lem = if l.is_empty() { "-".to_string() } else { l.clone() };
                    format!("{}(tag:{}, lemma:{})", w, tag, lem)
                }).collect::<Vec<_>>()
            ));
        }

        // 🌟 [DUAL AXIS] 드롭 대상 품사도 '청크 후보' 에는 남깁니다.
        //    '선적된' 이 VERB 로 판정되어도 그것이 transport 판정의 유일한 근거일 수 있습니다.
        const DROP_TAGS: [&str; 7] = ["VERB", "ADP", "PUNCT", "PART", "SCONJ", "CCONJ", "PRON"];
        let all_words: Vec<String> = tokens.iter().map(|(w, _, _)| w.clone()).collect();
        let content_flags: Vec<bool> = tokens
            .iter()
            .map(|(_, t, _)| !DROP_TAGS.iter().any(|d| d == t))
            .collect();

        let morph_alts: Vec<Vec<String>> = tokens
            .iter()
            .map(|(w, _, l)| crate::analytic::morphological_variants(w, l))
            .collect();
        let morph_depth: usize = morph_alts.iter().map(|v| v.len()).max().unwrap_or(0);

        // =====================================================================
        // STEP 3 : 슬라이딩 윈도우 청크 (1~6단어) + 형태소 변형
        // =====================================================================
        let mut chunk_texts: Vec<String> = Vec::new();
        let mut chunk_spans: Vec<(usize, usize)> = Vec::new();
        let mut seen_chunk: std::collections::HashSet<String> = std::collections::HashSet::new();

        for s in 0..all_words.len() {
            // 접두어로 이미 확정된 단어는 청크 생성에서 제외합니다.
            if consumed_words.contains(&all_words[s]) { continue; }
            let max_e = all_words.len().min(s + 6);
            for e in (s + 1)..=max_e {
                if (s..e).any(|i| consumed_words.contains(&all_words[i])) { continue; }

                let surface = all_words[s..e].join(" ");
                if !surface.trim().is_empty() {
                    let key = format!("{}|{}|{}", s, e, surface);
                    if seen_chunk.insert(key) {
                        chunk_texts.push(surface);
                        chunk_spans.push((s, e));
                    }
                }

                for d in 0..morph_depth {
                    let mut changed = false;
                    let mut parts: Vec<String> = Vec::with_capacity(e - s);
                    for i in s..e {
                        match morph_alts[i].get(d) {
                            Some(m) => { changed = true; parts.push(m.clone()); },
                            None => parts.push(all_words[i].clone()),
                        }
                    }
                    if !changed { continue; }
                    let mt = parts.join(" ");
                    if mt.trim().is_empty() { continue; }
                    let key = format!("{}|{}|{}", s, e, mt);
                    if seen_chunk.insert(key) {
                        chunk_texts.push(mt);
                        chunk_spans.push((s, e));
                    }
                }
            }
        }

        // =====================================================================
        // STEP 4 : Depth 1 뱅크 구축 + 임베딩
        //   편견은 별도 사전을 만들지 않고 '다른 카테고리의 bias' 를 씁니다.
        //   (get_detail_schema_fields 가 다른 필드의 bias 를 편견으로 쓰는 것과 동일 원리)
        // =====================================================================
        self.check_embedding_downloaded().await?;
        self.ensure_embedding().await?;

        let mut d1_bias: Vec<(String, String, String)> = Vec::new();
        let mut d1_prej: Vec<(String, String, String)> = Vec::new();
        for (cat, raw) in crate::logic::TRADE_CONDITION_CATEGORIES.iter() {
            for p in crate::utils::ai_utils::split_bias_phrases_full(raw) {
                d1_bias.push(("cond".to_string(), cat.to_string(), p));
            }
            for (other, other_raw) in crate::logic::TRADE_CONDITION_CATEGORIES.iter() {
                if other == cat { continue; }
                for p in crate::utils::ai_utils::split_bias_phrases_full(other_raw) {
                    d1_prej.push(("cond".to_string(), cat.to_string(), p));
                }
            }
        }

        // 유일 구만 1회 임베딩하고 재사용합니다.
        let mut uniq_d1: Vec<String> = Vec::new();
        for (_, _, p) in d1_bias.iter().chain(d1_prej.iter()) {
            if !uniq_d1.iter().any(|e| e == p) { uniq_d1.push(p.clone()); }
        }
        let mut uniq_d1_embs: Vec<Vec<f32>> = Vec::with_capacity(uniq_d1.len());
        for part in uniq_d1.chunks(200) {
            let e = self.get_embedding_batch(part.to_vec()).await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; part.len()]);
            uniq_d1_embs.extend(e);
        }
        let d1_emb_of = |p: &str| -> Vec<f32> {
            match uniq_d1.iter().position(|e| e == p) {
                Some(i) => uniq_d1_embs[i].clone(),
                None => vec![0.0f32; 384],
            }
        };
        let d1_bias_bank: Vec<(String, String, Vec<f32>)> = d1_bias.iter()
            .map(|(c, k, p)| (c.clone(), k.clone(), d1_emb_of(p))).collect();
        let d1_prej_bank: Vec<(String, String, Vec<f32>)> = d1_prej.iter()
            .map(|(c, k, p)| (c.clone(), k.clone(), d1_emb_of(p))).collect();

        emit_term(&format!(
            "   📐 [DEPTH-1 BANK] 카테고리 {}개 | 판정 구 {}개 | 편견 구 {}개 | 청크 후보 {}개",
            crate::logic::TRADE_CONDITION_CATEGORIES.len(),
            d1_bias_bank.len(), d1_prej_bank.len(), chunk_texts.len()
        ));

        let chunk_embs: Vec<Vec<f32>> = if chunk_texts.is_empty() {
            Vec::new()
        } else {
            let mut acc: Vec<Vec<f32>> = Vec::with_capacity(chunk_texts.len());
            for part in chunk_texts.chunks(200) {
                let e = self.get_embedding_batch(part.to_vec()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; part.len()]);
                acc.extend(e);
            }
            acc
        };

        // =====================================================================
        // STEP 5 : Depth 1 SURPRISAL 채점
        //   surprisal = (max - μ_global)/σ_global - √(2 ln N)
        //   뱅크 크기 편향(reference 44구 vs parties 3구)이 제거됩니다.
        // =====================================================================
        struct TradeSpan {
            start: usize,
            end: usize,
            text: String,
            category: String,
            score: f32,
            max_cos: f32,
            alts: Vec<(String, f32)>,
        }

        let mut candidates: Vec<TradeSpan> = Vec::new();
        let mut rescue_pool: Vec<TradeSpan> = Vec::new();
        // 🌟 [BANK-NEUTRAL D1] 저장(역방향)과 같은 채점기를 씁니다.
        //  ── 왜 필요한가 ──
        //   TRADE_CONDITION_CATEGORIES 는 카테고리별 구 수가 크게 다릅니다.
        //   (reference 계열은 참조 필드 44개 기준으로 뱅크가 크고 parties 는 3~5구)
        //   √(2 ln N) 차감은 큰 뱅크를 과잉 처벌하므로,
        //   '참조번호 질의' 가 구조적으로 parties 로 흘러가는 편향이 생깁니다.
        //   행/열 이중 센터링은 뱅크 크기에 무관하므로 이 편향이 사라집니다.
        //  ── 다국어 ──
        //   입력은 다국어 임베딩 벡터뿐이라 앵커가 영어 한 벌이어도
        //   한국어/일본어/중국어 질의가 동일한 척도로 채점됩니다.
        let (d1_keys, d1_net, d1_cos) = crate::utils::ai_utils::bank_neutral_key_matrix(
            &chunk_embs, &d1_bias_bank, &d1_prej_bank,
        );
        for (ci, (s, e)) in chunk_spans.iter().enumerate() {
            if cancel_token.load(std::sync::atomic::Ordering::Relaxed) {
                emit_term("[ENGINE] 🛑 Task cancelled by user. Terminating safely.");
                return Ok(json!({ "context": [], "cancelled": true }));
            }
            let q = match chunk_embs.get(ci) { Some(v) => v, None => continue };
            if q.iter().all(|&v| v == 0.0) { continue; }
            if d1_keys.is_empty() { continue; }
            // 이 청크(열 ci)에 대한 카테고리 점수를 내림차순으로 정리합니다.
            let mut col: Vec<(String, f32, f32)> = Vec::with_capacity(d1_keys.len());
            for (ki, k) in d1_keys.iter().enumerate() {
                let v = d1_net[ki][ci];
                if v == f32::MIN { continue; }
                let c = if d1_cos[ki][ci] == f32::MIN { 0.0 } else { d1_cos[ki][ci] };
                col.push((k.clone(), v, c));
            }
            if col.is_empty() { continue; }
            col.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let alts: Vec<(String, f32)> = col.iter().skip(1).take(3)
                .map(|x| (x.0.clone(), x.1)).collect();
            let span = TradeSpan {
                start: *s,
                end: *e,
                text: chunk_texts[ci].clone(),
                category: col[0].0.clone(),
                score: col[0].1,
                max_cos: col[0].2,
                alts,
            };
            if col[0].1 > 0.0 {
                emit_term(&format!(
                    "   🎯 [D1 CANDIDATE] \"{}\" → {} | Score(bank-neutral): {:+.4} | MaxCos: {:.4}",
                    chunk_texts[ci], col[0].0, col[0].1, col[0].2
                ));
                candidates.push(span);
            } else {
                rescue_pool.push(span);
            }
        }

        // 🌟 [COVERAGE RESCUE] 게이트를 넘은 후보가 0건이면 최상위 후보를 승격합니다.
        if candidates.is_empty() && !rescue_pool.is_empty() {
            rescue_pool.sort_by(|a, b| b.max_cos.partial_cmp(&a.max_cos).unwrap_or(std::cmp::Ordering::Equal));
            emit_term(&format!(
                "   🛟 [COVERAGE RESCUE] 게이트 통과 후보가 0건이라 상위 후보 {}건을 승격합니다.",
                rescue_pool.len().min(4)
            ));
            for r in rescue_pool.into_iter().take(4) {
                emit_term(&format!(
                    "      ↳ \"{}\" → {} | Surprisal: {:+.4} | MaxCos: {:.4}",
                    r.text, r.category, r.score, r.max_cos
                ));
                candidates.push(r);
            }
        }

        // =====================================================================
        // STEP 6 : NMS 배틀 + 흡수 + 갭 브리징
        // =====================================================================
        candidates.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                .then((b.end - b.start).cmp(&(a.end - a.start)))
        });

        let mut winners: Vec<TradeSpan> = Vec::new();
        for c in candidates.into_iter() {
            let mut overlapped = false;
            let mut winner_text = String::new();
            for w in winners.iter_mut() {
                if c.start < w.end && c.end > w.start {
                    overlapped = true;
                    winner_text = w.text.clone();
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
                        w.category = c.category.clone();
                    }
                    break;
                }
            }
            if !overlapped {
                emit_term(&format!(
                    "   👑 [NMS WINNER] \"{}\" → {} | Surprisal: {:+.4}",
                    c.text, c.category, c.score
                ));
                winners.push(c);
            } else {
                emit_term(&format!(
                    "   💀 [NMS DEFEAT] \"{}\" ({}) 는 상위 스팬 '{}' 에 흡수되었습니다.",
                    c.text, c.category, winner_text
                ));
            }
        }

        winners.sort_by(|a, b| a.start.cmp(&b.start));

        // 🌟 [GAP BRIDGING] NMS 에서 커버되지 않은 고아 단어를 인접 승자에 흡수시킵니다.
        if !winners.is_empty() {
            if winners[0].start > 0 {
                let gap = all_words[0..winners[0].start].join(" ");
                emit_term(&format!("   🛠️ [LEFT EDGE] '{}' → '{}' 에 흡수", gap, winners[0].text));
                winners[0].start = 0;
                winners[0].text = format!("{} {}", gap, winners[0].text);
            }
            for i in 0..(winners.len().saturating_sub(1)) {
                let gs = winners[i].end;
                let ge = winners[i + 1].start;
                if gs < ge {
                    let gap = all_words[gs..ge].join(" ");
                    emit_term(&format!("   ⚔️ [GAP BATTLE] '{}' → LEFT '{}' 에 흡수", gap, winners[i].text));
                    winners[i].end = ge;
                    winners[i].text = format!("{} {}", winners[i].text, gap);
                }
            }
            let last = winners.len() - 1;
            if winners[last].end < all_words.len() {
                let gap = all_words[winners[last].end..].join(" ");
                emit_term(&format!("   🛠️ [RIGHT EDGE] '{}' → '{}' 에 흡수", gap, winners[last].text));
                winners[last].end = all_words.len();
                winners[last].text = format!("{} {}", winners[last].text, gap);
            }
        }

        // =====================================================================
        // STEP 7 : Depth 1 마진 부족 시에만 LLM 재판정
        // =====================================================================
        let mut need_d1_llm: Vec<usize> = Vec::new();
        for (wi, w) in winners.iter().enumerate() {
            if w.alts.first().map_or(false, |(_, s)| *s >= w.score * 0.9) {
                need_d1_llm.push(wi);
            }
        }

        if !need_d1_llm.is_empty() {
            emit_term(&format!(
                "   ⚖️ [D1 MARGIN GATE] 1위-2위가 사실상 동률인 스팬 {}개에 대해 LLM 재판정을 수행합니다.",
                need_d1_llm.len()
            ));
            self.secure_vram_relay(crate::model::ModelSize::Qwen3_5, None, Some(cancel_token.clone()), false, None).await?;

            for wi in need_d1_llm {
                if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { break; }
                let mut scored: Vec<(String, f32)> = vec![(winners[wi].category.clone(), winners[wi].score)];
                for (k, s) in winners[wi].alts.iter() { scored.push((k.clone(), *s)); }

                let p = crate::parsing::trade_condition_category_prompt(&winners[wi].text, &query, &scored);
                let params = crate::openai_types::ChatCompletionParameters {
                    messages: vec![crate::openai_types::ChatCompletionRequestMessage::User(
                        crate::openai_types::ChatCompletionRequestUserMessage {
                            content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(p),
                            name: None,
                        })],
                    model: "qwen3.5".to_string(),
                    max_tokens: Some(96),
                    temperature: Some(0.0),
                    top_p: Some(0.95),
                    ..Default::default()
                };
                let r = if let Some(gen) = self.qwen3_5_generator.lock().await.as_mut() {
                    gen.generate(params, Some(cancel_token.clone()), Some(format!("{}_tq_d1_{}", task_id, wi)), None, None, None)
                        .await.unwrap_or_default()
                } else { String::new() };

                let picked = crate::parsing::parse_json_from_llm(&r)
                    .get("category").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
                let allowed = scored.iter().any(|(k, _)| k == &picked);
                if allowed && picked != winners[wi].category {
                    emit_term(&format!(
                        "   🤖 [D1 LLM] \"{}\" 의 카테고리를 '{}' → '{}' 로 교정했습니다.",
                        winners[wi].text, winners[wi].category, picked
                    ));
                    winners[wi].category = picked;
                } else if !picked.is_empty() && !allowed {
                    emit_term(&format!(
                        "   🚫 [D1 LLM REJECT] '{}' 는 후보 목록에 없어 폐기하고 '{}' 를 유지합니다.",
                        picked, winners[wi].category
                    ));
                }
            }
        } else {
            emit_term("   ⚡ [D1 DETERMINISTIC] 벡터 마진이 충분하여 카테고리 LLM 호출을 생략합니다.");
        }

        // =====================================================================
        // STEP 8 : Depth 2 — 카테고리 내부 파라미터 배타 배정
        // =====================================================================
        let mut conditions = serde_json::Map::new();
        let mut hub_values: Vec<String> = Vec::new();
        let mut consumed_span: Vec<bool> = vec![false; all_words.len()];
        let mut claimed_fields: std::collections::HashSet<String> = std::collections::HashSet::new();

        // ① 접두어로 확정된 축을 먼저 잠급니다.
        for (field, value) in deterministic_refs.iter() {
            conditions.insert(field.clone(), json!({
                "operator": crate::logic::trade_default_operator(field),
                "value": value
            }));
            claimed_fields.insert(field.clone());
            emit_term(&format!(
                "   🔒 [D2 LOCKED] '{}' = '{}' (접두어 확정, 벡터 경쟁 대상 아님)",
                field, value
            ));
        }

        // ② 카테고리별로 승리 스팬을 묶어 배타 배정합니다.
        let mut by_cat: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
        for (wi, w) in winners.iter().enumerate() {
            by_cat.entry(w.category.clone()).or_default().push(wi);
        }

        // 카테고리 순회 순서를 고정해 로그 재현성을 확보합니다.
        let cat_order: Vec<String> = crate::logic::TRADE_CONDITION_CATEGORIES
            .iter().map(|(c, _)| c.to_string()).collect();

        let mut d2_llm_pending: Vec<(usize, String, Vec<(String, String, f32)>)> = Vec::new();

        for cat in cat_order.iter() {
            let span_idxs = match by_cat.get(cat) { Some(v) => v.clone(), None => continue };
            if span_idxs.is_empty() { continue; }

            // 🌟 hub 는 필드가 하나뿐이라 경쟁이 없습니다.
            if cat == "hub" {
                for wi in span_idxs.iter() {
                    let v = crate::utils::ai_utils::deterministic_condition_value(
                        &vec![winners[*wi].text.clone()], false,
                    );
                    if v.trim().is_empty() { continue; }
                    if !hub_values.iter().any(|x| x == &v) { hub_values.push(v.clone()); }
                    for i in winners[*wi].start..winners[*wi].end {
                        if i < consumed_span.len() { consumed_span[i] = true; }
                    }
                    emit_term(&format!("   🧲 [D2 HUB] \"{}\" → hub_reference", v));
                }
                continue;
            }

            let fields = crate::logic::trade_condition_fields(cat);
            if fields.is_empty() { continue; }

            // 필드 앵커 임베딩
            let mut f_names: Vec<String> = Vec::new();
            let mut f_descs: Vec<String> = Vec::new();
            let mut f_banks: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut f_weights: Vec<Vec<f32>> = Vec::new();
            for (fname, fdesc, anchor) in fields.iter() {
                if claimed_fields.contains(*fname) { continue; }
                let (ph, wt) = crate::utils::ai_utils::split_bias_phrases_weighted_full(anchor);
                if ph.is_empty() { continue; }
                let embs = self.get_embedding_batch(ph.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; ph.len()]);
                f_names.push(fname.to_string());
                f_descs.push(fdesc.to_string());
                f_banks.push(embs);
                f_weights.push(wt);
            }
            if f_names.is_empty() { continue; }

            emit_term(&format!(
                "   📐 [D2 BANK] 카테고리 '{}' | 후보 필드 {}개 | 대상 스팬 {}개",
                cat, f_names.len(), span_idxs.len()
            ));

            // (필드 × 스팬) 행렬
            let mut matrix: Vec<Vec<f32>> = vec![vec![-1.0f32; span_idxs.len()]; f_names.len()];
            let mut span_embs: Vec<Vec<f32>> = Vec::with_capacity(span_idxs.len());
            for wi in span_idxs.iter() {
                let e = self.get_embedding(winners[*wi].text.clone()).await.unwrap_or(vec![0.0; 384]);
                span_embs.push(e);
            }

            for fi in 0..f_names.len() {
                let fmt = crate::utils::ai_utils::detect_field_format(&f_names[fi]);
                for (si, wi) in span_idxs.iter().enumerate() {
                    let e = &span_embs[si];
                    if e.iter().all(|&v| v == 0.0) { continue; }

                    // 🌟 [FORMAT GATE] 배정 '전' 에 값 생김새부터 검증합니다.
                    let raw_val = winners[*wi].text.trim();
                    let ok = match fmt {
                        crate::utils::ai_utils::FieldFormat::Numeric =>
                            raw_val.chars().any(|c| c.is_ascii_digit()),
                        crate::utils::ai_utils::FieldFormat::Date =>
                            raw_val.chars().any(|c| c.is_ascii_digit()),
                        _ => true,
                    };
                    if !ok { continue; }

                    matrix[fi][si] = crate::utils::ai_utils::weighted_max_pool_sim(
                        e, &f_banks[fi], &f_weights[fi],
                    );
                }
            }

            let centered = crate::utils::ai_utils::double_center_matrix(&matrix);
            let assign = crate::utils::ai_utils::exclusive_assign_by_score(&centered, 0.0, 0.0);

            for (fi, a) in assign.iter().enumerate() {
                let (si, own, margin) = match a { Some(v) => *v, None => continue };
                let wi = span_idxs[si];

                // 마진이 사실상 0 이면 LLM 재판정 대기열에 넣습니다.
                if margin.abs() < 0.005 {
                    let mut scored: Vec<(String, String, f32)> = Vec::new();
                    for k in 0..f_names.len() {
                        let s = matrix[k][si];
                        if s < 0.0 { continue; }
                        scored.push((f_names[k].clone(), f_descs[k].clone(), s));
                    }
                    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                    scored.truncate(6);
                    d2_llm_pending.push((wi, cat.clone(), scored));
                    emit_term(&format!(
                        "   ⚖️ [D2 MARGIN GATE] \"{}\" 의 1위-2위 마진 {:+.4} 로 LLM 재판정 대기열에 넣습니다.",
                        winners[wi].text, margin
                    ));
                    continue;
                }

                let field = f_names[fi].clone();
                let value = trade_resolve_condition_value(&field, &winners[wi].text);
                if value.trim().is_empty() {
                    emit_term(&format!(
                        "   ⚪ [D2 NO VALUE] '{}' 에 배정된 \"{}\" 에서 값을 뽑지 못해 조건을 만들지 않습니다.",
                        field, winners[wi].text
                    ));
                    continue;
                }

                let op = trade_resolve_condition_operator(&field, &winners[wi].text);
                conditions.insert(field.clone(), json!({ "operator": op, "value": value }));
                claimed_fields.insert(field.clone());
                for i in winners[wi].start..winners[wi].end {
                    if i < consumed_span.len() { consumed_span[i] = true; }
                }
                emit_term(&format!(
                    "   🔗 [D2 ASSIGN] \"{}\" → {}.{} {} '{}' | Score: {:+.4} | Margin: {:+.4}",
                    winners[wi].text, cat, field, op, value, own, margin
                ));
            }
        }

        // =====================================================================
        // STEP 9 : Depth 2 마진 부족분만 LLM 1회씩
        // =====================================================================
        if !d2_llm_pending.is_empty() {
            self.secure_vram_relay(crate::model::ModelSize::Qwen3_5, None, Some(cancel_token.clone()), false, None).await?;
            for (idx, (wi, cat, scored)) in d2_llm_pending.into_iter().enumerate() {
                if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { break; }

                let p = crate::parsing::trade_condition_field_prompt(&winners[wi].text, &query, &cat, &scored);
                let params = crate::openai_types::ChatCompletionParameters {
                    messages: vec![crate::openai_types::ChatCompletionRequestMessage::User(
                        crate::openai_types::ChatCompletionRequestUserMessage {
                            content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(p),
                            name: None,
                        })],
                    model: "qwen3.5".to_string(),
                    max_tokens: Some(96),
                    temperature: Some(0.0),
                    top_p: Some(0.95),
                    ..Default::default()
                };
                let r = if let Some(gen) = self.qwen3_5_generator.lock().await.as_mut() {
                    gen.generate(params, Some(cancel_token.clone()), Some(format!("{}_tq_d2_{}", task_id, idx)), None, None, None)
                        .await.unwrap_or_default()
                } else { String::new() };

                let picked = crate::parsing::parse_json_from_llm(&r)
                    .get("field").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();

                if picked.is_empty() {
                    emit_term(&format!("   ⚪ [D2 LLM] \"{}\" 는 어느 필드에도 맞지 않아 조건에서 제외합니다.", winners[wi].text));
                    continue;
                }
                if !scored.iter().any(|(f, _, _)| f == &picked) {
                    emit_term(&format!("   🚫 [D2 LLM REJECT] '{}' 는 후보 목록에 없어 폐기합니다.", picked));
                    continue;
                }
                if claimed_fields.contains(&picked) {
                    emit_term(&format!("   🚫 [D2 LLM REJECT] '{}' 는 이미 다른 청크가 선점했습니다.", picked));
                    continue;
                }

                let value = trade_resolve_condition_value(&picked, &winners[wi].text);
                if value.trim().is_empty() {
                    emit_term(&format!("   ⚪ [D2 NO VALUE] '{}' 에서 값을 뽑지 못했습니다.", picked));
                    continue;
                }
                let op = trade_resolve_condition_operator(&picked, &winners[wi].text);
                conditions.insert(picked.clone(), json!({ "operator": op, "value": value }));
                claimed_fields.insert(picked.clone());
                for i in winners[wi].start..winners[wi].end {
                    if i < consumed_span.len() { consumed_span[i] = true; }
                }
                emit_term(&format!(
                    "   🤖 [D2 LLM] \"{}\" → {} {} '{}'",
                    winners[wi].text, picked, op, value
                ));
            }
        }

        // =====================================================================
        // STEP 10 : 벡터 근거가 전무하면 레거시 폴백 1회
        // =====================================================================
        if conditions.is_empty() && hub_values.is_empty() {
            emit_term("   🛟 [FALLBACK] 벡터 근거가 전무하여 레거시 단일 프롬프트를 1회 호출합니다.");
            self.secure_vram_relay(crate::model::ModelSize::Qwen3, None, Some(cancel_token.clone()), false, None).await?;

            let prompt = crate::parsing::extract_shipping_conditions(&query, language);
            let gen_arc = self.qwen3_generator.clone();
            let cancel_clone = cancel_token.clone();
            let res = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
                let mut gen_guard = gen_arc.blocking_lock();
                if let Some(gen) = gen_guard.as_mut() {
                    let params = crate::openai_types::ChatCompletionParameters {
                        messages: vec![
                            crate::openai_types::ChatCompletionRequestMessage::User(crate::openai_types::ChatCompletionRequestUserMessage {
                                content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(prompt),
                                name: None,
                            })
                        ],
                        model: "qwen3".to_string(), max_tokens: Some(256), temperature: Some(0.0), top_p: Some(0.95),
                        ..Default::default()
                    };
                    gen.generate(params, Some(cancel_clone), None, None).map_err(|e| anyhow::anyhow!("Qwen3 Inference failed: {}", e))
                } else {
                    Err(anyhow::anyhow!("Qwen3 Generator is missing"))
                }
            }).await??;

            emit_term(&format!("   [FALLBACK RESULT]\n{}", res));

            let parsed = crate::parsing::parse_json_from_llm(&res);
            if let Some(obj) = parsed.as_object() {
                for (k, v) in obj {
                    let op = v.get("operator").and_then(|x| x.as_str()).unwrap_or("").trim().to_lowercase();
                    let val = match v.get("value") { Some(x) => x.clone(), None => continue };
                    let is_empty = match &val {
                        Value::Null => true,
                        Value::String(s) => s.trim().is_empty() || s == "null" || s == "N/A",
                        _ => false,
                    };
                    if is_empty { continue; }

                    if k == "hub_reference" {
                        if let Some(s) = val.as_str() {
                            if !hub_values.iter().any(|x| x == s) { hub_values.push(s.to_string()); }
                        }
                        continue;
                    }

                    let final_op = if !op.is_empty() {
                        op
                    } else {
                        crate::logic::trade_default_operator(k).to_string()
                    };
                    conditions.insert(k.clone(), json!({ "operator": final_op, "value": val }));
                }
            }
        }

        // =====================================================================
        // STEP 11 : 스코프 확정 (45종 전체) + 허브 확장
        // =====================================================================
        // 🌟 [SCOPE v4]
        //  ── v3 의 결함 ──
        //   27개 코드만 나열해, 45종 데이터셋의
        //   HBL / FCR / POD / SWB / LLC / LG / TR / CDR / ICF / SOA / TI / CSI /
        //   EL / CCC / CM / CP / FI / FC / PC / COA / CNM / IP / DN / CN / BK
        //   가 type IN (...) 에서 전량 탈락했습니다.
        //  ── v4 ──
        //   logic.rs 가 소유한 참조 필드 사전과 같은 계보의 코드 목록을 씁니다.
        //   저장 시 type 컬럼에 doc_type 이 그대로 들어가므로 대소문자를 함께 넣습니다.
        let mut trade_types: Vec<String> = vec![
            "tracking".to_string(), "TRACKING".to_string(),
            "receiving".to_string(), "Receiving".to_string(),
            "shipping".to_string(), "Shipping".to_string(),
            "shipping_doc".to_string(),
        ];
        for t in [
            // 계약 · 결제
            "PO", "PI", "SC", "LC", "LLC", "CP",
            // 상거래 · 선적
            "CI", "CINV", "CSI", "PL", "BL", "HBL", "SWB", "AWB",
            "BC", "BK", "SA", "DO", "AN", "FCR", "POD", "CM", "FI",
            // 통관 · 신고
            "ED", "ID", "CO", "EL", "CCC",
            // 검사 · 증명
            "IC", "WC", "CA", "COA", "PHYTO", "PC", "HC", "BEN_CERT", "FC", "CNM",
            // 특수 · 법무 · 금융
            "DGD", "MSDS", "POA", "BIZ_LIC", "INS", "IP",
            "LG", "TR", "CDR", "ICF", "SOA", "DN", "CN", "TI",
        ] {
            let up = t.to_string();
            let lo = t.to_lowercase();
            if !trade_types.iter().any(|x| x == &up) { trade_types.push(up); }
            if !trade_types.iter().any(|x| x == &lo) { trade_types.push(lo); }
        }

        // 🌟 [HUB EXPANSION] 허브 번호는 어느 참조 축에 들어 있을지 알 수 없습니다.
        //    그래서 '모든 참조 축 + doc_number' 에 대한 OR 조건으로 펼쳐 내려보냅니다.
        //    Dexie(executeDexiePlan)가 alternates 를 읽어 재질의하므로,
        //    LanceDB 스코프를 좁히지 않고도 정밀 필터가 성립합니다.
        let mut alternates = serde_json::Map::new();
        if !hub_values.is_empty() {
            let hub_val = hub_values.join(" ");
            let mut axes: Vec<String> = vec!["doc_number".to_string(), "no".to_string()];
            for f in crate::logic::TRADE_REFERENCE_FIELDS.iter() {
                axes.push(f.to_string());
            }
            conditions.insert("hub_reference".to_string(), json!({
                "operator": "contains",
                "value": hub_val.clone()
            }));
            alternates.insert("hub_reference".to_string(), json!(axes.clone()));
            emit_term(&format!(
                "   🧲 [HUB EXPANSION] '{}' 를 doc_number + 참조 축 {}개로 확장했습니다.",
                hub_val, axes.len()
            ));
        }

        // 🌟 [ALTERNATE AXIS] 참조 축은 서로 오배정될 수 있으므로,
        //    같은 값이 갈 수 있었던 다른 참조 축을 대안으로 함께 실어 보냅니다.
        for (k, v) in conditions.iter() {
            if !k.starts_with("reference_") { continue; }
            let val = v.get("value").and_then(|x| x.as_str()).unwrap_or("");
            if val.is_empty() { continue; }
            let mut axes: Vec<String> = Vec::new();
            for f in crate::logic::TRADE_REFERENCE_FIELDS.iter() {
                if *f == k.as_str() { continue; }
                axes.push(f.to_string());
            }
            axes.push("doc_number".to_string());
            alternates.insert(k.clone(), json!(axes));
        }

        // =====================================================================
        // STEP 12 : FTS 검색어 (조건으로 소비되지 않은 단어 + 확정 값)
        // =====================================================================
        let mut keywords: Vec<String> = Vec::new();
        for (i, w) in all_words.iter().enumerate() {
            if consumed_span.get(i).copied().unwrap_or(false) { continue; }
            if !content_flags.get(i).copied().unwrap_or(true) { continue; }
            if !keywords.iter().any(|k| k == w) { keywords.push(w.clone()); }
        }
        for (_, v) in conditions.iter() {
            if let Some(s) = v.get("value").and_then(|x| x.as_str()) {
                for w in s.split_whitespace() {
                    if !keywords.iter().any(|k| k == w) { keywords.push(w.to_string()); }
                }
            }
        }
        if keywords.is_empty() {
            keywords = query.split_whitespace().map(|s| s.to_string()).collect();
        }

        let target_text = {
            let mut w: Vec<String> = Vec::new();
            for x in query.split_whitespace() {
                if !w.iter().any(|e| e == x) { w.push(x.to_string()); }
            }
            for x in keywords.iter() {
                if !w.iter().any(|e| e == x) { w.push(x.clone()); }
            }
            w.join(" ")
        };

        emit_term(&format!(
            "   🧷 [KEYWORDS] {:?}",
            keywords.iter().take(16).collect::<Vec<_>>()
        ));
        emit_term(&format!(
            "[STAGE-2 CONDITIONS] 확정 조건 {}개: {}",
            conditions.len(),
            serde_json::to_string(&Value::Object(conditions.clone())).unwrap_or_default()
        ));
        emit_term(&format!("[STAGE-2] Trade document types in scope: {} 종", trade_types.len()));

        let payload = json!({ "task_id": task_id, "category": "Done", "summary": "Filter extraction complete.", "spinner": "✅" });
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::utils::logger::log_task_progress(app_handle, task_id, &payload);

        let ctx = json!([{
            "type": "tracking",
            "types": trade_types,
            "text": target_text,
            "condition": Value::Object(conditions),
            "alternates": Value::Object(alternates),
            "unassigned": keywords,
            "substantial": "",
            "find": "",
            "tier": "TRADING"
        }]);

        emit_term("[SUCCESS] Shipping Search Pipeline Completed.");
        Ok(json!({ "context": ctx }))
    }

}
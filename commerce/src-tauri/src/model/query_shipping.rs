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
        crate::utils::score_dynamics::enter_scope(
            "",
            crate::utils::score_dynamics::Track::Search,
            "all",
            "",
        );

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
        let mut deterministic_refs: Vec<(String, String)> = Vec::new();
        let mut prefix_hits: Vec<(String, String, String)> = Vec::new();
        let mut consumed_words: std::collections::HashSet<String> = std::collections::HashSet::new();

        for raw_word in query.split_whitespace() {
            let mut runs: Vec<String> = Vec::new();
            {
                let mut cur = String::new();
                for ch in raw_word.chars() {
                    if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                        cur.push(ch);
                    } else {
                        if !cur.is_empty() { runs.push(std::mem::take(&mut cur)); }
                    }
                }
                if !cur.is_empty() { runs.push(cur); }
            }
            for core in runs {
                let core = core.trim_matches(|c| c == '-' || c == '_').to_string();
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
                    prefix_hits.push((prefix.clone(), core.clone(), field.to_string()));
                    consumed_words.insert(raw_word.to_string());
                }
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
        const DROP_TAGS: [&str; 7] = crate::utils::ai_utils::STANZA_DROP_TAGS;
        let all_words: Vec<String> = tokens.iter().map(|(w, _, _)| w.clone()).collect();
        let content_flags: Vec<bool> = tokens
            .iter()
            .map(|(_, t, _)| !DROP_TAGS.iter().any(|d| d == t))
            .collect();

        let lemmas: Vec<String> = tokens.iter().map(|(_, _, l)| l.clone()).collect();
        self.check_embedding_downloaded().await?;
        self.ensure_embedding().await?;
        let layers = self.build_shipping_query_layers(&all_words, &lemmas, &consumed_words).await;
        for line in layers.logs.iter() {
            emit_term(line);
        }
        let morph_alts: Vec<Vec<String>> = layers.variants.clone();
        let morph_depth: usize = morph_alts.iter().map(|v| v.len()).max().unwrap_or(0);

        // =====================================================================
        // STEP 3 : 슬라이딩 윈도우 청크 (1~6단어) + 형태소 변형
        // =====================================================================
        let mut chunk_texts: Vec<String> = Vec::new();
        let mut chunk_spans: Vec<(usize, usize)> = Vec::new();
        let mut seen_chunk: std::collections::HashSet<String> = std::collections::HashSet::new();

        let relation_exact: Vec<bool> = (0..all_words.len())
            .map(|i| layers.relation_marks.contains(&i) && ship_relation_exact(&all_words[i]))
            .collect();
        if relation_exact.iter().any(|&r| r) {
            emit_term(&format!(
                "   🔗 [RELATION MARK / D1 EXCLUDE] 관계 표지 {:?} 는 D1 카테고리와 독립된 축으로 이미 확정했으므로 청크 후보에서 뺍니다. 후보로 두면 '연결' 이 검사·증명 카테고리의 승자가 되는 식으로 조건과 무관한 D1 승자가 생기고, 그 승자가 겹치는 이웃 스팬을 NMS 로 누릅니다.",
                (0..all_words.len()).filter(|&i| relation_exact[i]).map(|i| all_words[i].clone()).collect::<Vec<_>>()
            ));
        }
        for s in 0..all_words.len() {
            if layers.roles.get(s) != Some(&ShipTokenRole::Content) || relation_exact[s] { continue; }
            let max_e = all_words.len().min(s + 6);
            for e in (s + 1)..=max_e {
                if (s..e).any(|i| layers.roles.get(i) != Some(&ShipTokenRole::Content) || relation_exact[i]) { break; }

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

        let cat_phrases: Vec<(String, Vec<String>)> = crate::logic::TRADE_CONDITION_CATEGORIES
            .iter()
            .map(|(cat, _)| (cat.to_string(), crate::logic::trade_condition_category_phrases(cat)))
            .collect();
        let mut d1_bias: Vec<(String, String, String)> = Vec::new();
        let mut d1_prej: Vec<(String, String, String)> = Vec::new();
        let mut shared_skipped = 0usize;
        for (cat, phrases) in cat_phrases.iter() {
            for p in phrases.iter() {
                d1_bias.push(("cond".to_string(), cat.clone(), p.clone()));
            }
            for (other, other_phrases) in cat_phrases.iter() {
                if other == cat { continue; }
                for p in other_phrases.iter() {
                    if phrases.iter().any(|x| x.eq_ignore_ascii_case(p)) {
                        shared_skipped += 1;
                        continue;
                    }
                    d1_prej.push(("cond".to_string(), cat.clone(), p.clone()));
                }
            }
        }
        if shared_skipped > 0 {
            emit_term(&format!(
                "   🧹 [D1 SELF-PREJUDICE DROP] 두 카테고리가 같은 구를 공유해 자기 판정 구가 자기 편견으로도 들어가려 한 자리 {}건을 제외했습니다. 영어 한 벌일 때는 카테고리 간 표기가 거의 겹치지 않아 드러나지 않던 결함인데, 12개 언어를 합치면 '증명서 번호'·'계약번호' 처럼 같은 번역어가 여러 카테고리에 동시에 존재해 자기 점수가 자기 편견에 상쇄됩니다.",
                shared_skipped
            ));
        }

        // 유일 구만 1회 임베딩하고 재사용합니다.
        let mut uniq_d1: Vec<String> = Vec::new();
        let mut d1_index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for (_, _, p) in d1_bias.iter().chain(d1_prej.iter()) {
            if d1_index.contains_key(p) { continue; }
            d1_index.insert(p.clone(), uniq_d1.len());
            uniq_d1.push(p.clone());
        }
        let mut uniq_d1_embs: Vec<Vec<f32>> = Vec::with_capacity(uniq_d1.len());
        for part in uniq_d1.chunks(200) {
            let e = self.get_embedding_batch(part.to_vec()).await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; part.len()]);
            uniq_d1_embs.extend(e);
        }
        let zero_d1 = vec![0.0f32; 384];
        let d1_emb_of = |p: &str| -> Vec<f32> {
            match d1_index.get(p) {
                Some(&i) => uniq_d1_embs[i].clone(),
                None => zero_d1.clone(),
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

        candidates.sort_by(|a, b| {
            b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                .then((b.end - b.start).cmp(&(a.end - a.start)))
        });

        let mut winners: Vec<TradeSpan> = Vec::new();
        for c in candidates.into_iter() {
            let blocker = winners
                .iter()
                .find(|w| c.start < w.end && c.end > w.start)
                .map(|w| w.text.clone());
            match blocker {
                None => {
                    emit_term(&format!(
                        "   👑 [NMS WINNER] \"{}\" → {} | Surprisal: {:+.4}",
                        c.text, c.category, c.score
                    ));
                    winners.push(c);
                }
                Some(bt) => {
                    emit_term(&format!(
                        "   💀 [NMS SUPPRESS] \"{}\" ({}) 는 겹치는 상위 스팬 '{}' 에 밀려 억제됩니다. 승자 스팬은 확장하지 않습니다.",
                        c.text, c.category, bt
                    ));
                }
            }
        }

        winners.sort_by(|a, b| a.start.cmp(&b.start));

        {
            let is_content = |i: usize| layers.roles.get(i) == Some(&ShipTokenRole::Content);
            let covered = |i: usize, ws: &Vec<TradeSpan>| ws.iter().any(|w| i >= w.start && i < w.end);
            let mut i = 0usize;
            while i < all_words.len() {
                if !is_content(i) || covered(i, &winners) {
                    i += 1;
                    continue;
                }
                let run_start = i;
                while i < all_words.len() && is_content(i) && !covered(i, &winners) {
                    i += 1;
                }
                let run_end = i;
                let gap = all_words[run_start..run_end].join(" ");
                if let Some(li) = winners.iter().position(|w| w.end == run_start) {
                    emit_term(&format!("   🧵 [CONTENT BRIDGE] '{}' → LEFT '{}' 에 연결", gap, winners[li].text));
                    winners[li].end = run_end;
                    winners[li].text = format!("{} {}", winners[li].text, gap);
                } else if let Some(ri) = winners.iter().position(|w| w.start == run_end) {
                    emit_term(&format!("   🧵 [CONTENT BRIDGE] '{}' → RIGHT '{}' 에 연결", gap, winners[ri].text));
                    winners[ri].start = run_start;
                    winners[ri].text = format!("{} {}", gap, winners[ri].text);
                } else {
                    emit_term(&format!(
                        "   ⚪ [CONTENT ORPHAN] '{}' 는 붙을 승자 스팬이 없어 조건을 만들지 않습니다.",
                        gap
                    ));
                }
            }
        }

        {
            let scores: Vec<f32> = winners.iter().map(|w| w.score).collect();
            crate::utils::score_dynamics::record_decay("search.d1.winners", &scores);
            let mut cat_best: Vec<(String, f32)> = Vec::new();
            for w in winners.iter() {
                match cat_best.iter_mut().find(|(c, _)| *c == w.category) {
                    Some(slot) => {
                        if w.score > slot.1 { slot.1 = w.score; }
                    }
                    None => cat_best.push((w.category.clone(), w.score)),
                }
            }
            for (c, m) in cat_best.iter() {
                crate::utils::score_dynamics::record_category_max(c, crate::logic::trade_condition_fields(c).len(), *m);
            }
        }
        let d1_gate = crate::utils::ai_utils::gumbel_expected_z(crate::logic::TRADE_CONDITION_CATEGORIES.len());
        let mut need_d1_llm: Vec<usize> = Vec::new();
        for (wi, w) in winners.iter().enumerate() {
            let value_near = ship_value_near(&layers.roles, w.start, w.end, 2);
            if w.score < d1_gate && !value_near {
                continue;
            }
            let tied = w.alts.first().map_or(false, |(_, s)| *s >= w.score * 0.9);
            if !tied {
                continue;
            }
            if value_near {
                let accepts_value = |c: &str| -> bool {
                    ship_category_accepts(c, crate::utils::ai_utils::FieldFormat::Numeric)
                        || ship_category_accepts(c, crate::utils::ai_utils::FieldFormat::Identifier)
                };
                let (alt, alt_score) = w
                    .alts
                    .first()
                    .map(|(c, s)| (c.clone(), *s))
                    .unwrap_or((String::new(), 0.0));
                if accepts_value(&w.category) && accepts_value(&alt) {
                    emit_term(&format!(
                        "   ⚡ [D1 TIE / VALUE-FIRST DEFER] \"{}\" 의 1·2위 카테고리 '{}'({:+.4}) 와 '{}'({:+.4}) 가 동률이지만, 이 스팬에는 값이 붙어 있고 두 카테고리 모두 그 값 형식의 축을 갖습니다. 값이 붙은 스팬의 필드는 D2 VALUE-FIRST 가 카테고리 경계 없이 정하므로 카테고리 재판정은 결과를 바꾸지 못합니다. LLM 을 부르지 않습니다.",
                        w.text, w.category, w.score, alt, alt_score
                    ));
                    crate::utils::score_dynamics::record_baseline("search.d1_tie_deferred", 1.0);
                    continue;
                }
            }
            need_d1_llm.push(wi);
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

        let d1_view: Vec<ShipWinnerView> = winners
            .iter()
            .map(|w| ShipWinnerView { start: w.start, end: w.end, category: w.category.clone(), score: w.score })
            .collect();

        let mut conditions = serde_json::Map::new();
        let mut hints = serde_json::Map::new();
        let mut identity_alternates = serde_json::Map::new();
        let mut hub_values: Vec<String> = Vec::new();
        let mut consumed_span: Vec<bool> = vec![false; all_words.len()];
        let mut claimed_fields: std::collections::HashSet<String> = std::collections::HashSet::new();
        claimed_fields.insert("doc_type".to_string());

        let has_code_values = !prefix_hits.is_empty()
            || !layers.identifiers.is_empty()
            || layers.numerics.iter().any(|n| ship_is_long_code(n));
        let hub_intent = has_code_values
            && d1_view.iter().any(|w| w.category == "hub" && w.score >= d1_gate);

        let (doc_scope, projection, scope_logs) =
            ship_resolve_doc_scope(&layers.doc_mentions, &d1_view, &layers.relation_marks);
        for line in scope_logs.iter() { emit_term(line); }

        let mut identity_codes: Vec<String> = Vec::new();
        for (code, value, ref_field) in prefix_hits.iter() {
            if hub_intent {
                if !hub_values.iter().any(|x| x == value) { hub_values.push(value.clone()); }
                emit_term(&format!(
                    "   🧲 [PREFIX → HUB] '{}' 는 전체 연결 추적 의도와 함께 등장해 doc_number + 전 참조 축 탐색으로 보냅니다.",
                    value
                ));
                continue;
            }
            let own_type = doc_scope.is_empty() || doc_scope.iter().any(|c| c == code);
            if own_type && !claimed_fields.contains("doc_number") {
                conditions.insert("doc_number".to_string(), json!({ "operator": "eq", "value": value }));
                claimed_fields.insert("doc_number".to_string());
                identity_alternates.insert("doc_number".to_string(), json!([ref_field, "no"]));
                if !identity_codes.iter().any(|c| c == code) { identity_codes.push(code.clone()); }
                emit_term(&format!(
                    "   🔒 [IDENTITY LOCK] '{}' 는 '{}' 서식 자신의 번호입니다 → doc_number eq (대안 축: {}, no)",
                    value, code, ref_field
                ));
            } else if !claimed_fields.contains(ref_field) {
                conditions.insert(ref_field.clone(), json!({
                    "operator": crate::logic::trade_default_operator(ref_field),
                    "value": value
                }));
                claimed_fields.insert(ref_field.clone());
                emit_term(&format!(
                    "   🔒 [D2 LOCKED] '{}' = '{}' (조회 대상 서식 {:?} 가 '{}' 서식을 참조하는 방향)",
                    ref_field, value, doc_scope, code
                ));
            }
        }
        let final_scope: Vec<String> = if hub_intent {
            Vec::new()
        } else if !identity_codes.is_empty() {
            identity_codes.clone()
        } else {
            doc_scope.clone()
        };
        {
            let scope_key = crate::utils::score_dynamics::search_scope_key(&final_scope);
            if scope_key != "all" {
                crate::utils::score_dynamics::enter_scope(
                    "",
                    crate::utils::score_dynamics::Track::Search,
                    &scope_key,
                    "",
                );
                emit_term(&format!("   📈 [SDS] 검색 스코프를 'search|{}|' 로 좁혀 기록합니다.", scope_key));
            }
            crate::utils::score_dynamics::record_baseline("search.doc_scope", if final_scope.is_empty() { 0.0 } else { 1.0 });
            crate::utils::score_dynamics::record_baseline("search.temporal", if layers.temporal.is_some() { 1.0 } else { 0.0 });
        }

        if let Some(t) = layers.temporal.as_ref() {
            if !claimed_fields.contains(&t.field) {
                let cond = match t.operator.as_str() {
                    "gte" => json!({ "operator": "gte", "value": t.start }),
                    "lte" => json!({ "operator": "lte", "value": t.end }),
                    _ => json!({ "operator": "between", "value": t.start, "value_to": t.end }),
                };
                conditions.insert(t.field.clone(), cond);
                claimed_fields.insert(t.field.clone());
                for &i in t.tokens.iter() {
                    if i < consumed_span.len() { consumed_span[i] = true; }
                }
                emit_term(&format!(
                    "   📅 [TEMPORAL LOCK] {} {} | {} ~ {}",
                    t.field, t.operator, t.start, t.end
                ));
            }
        }

        let mut bound_nums: Vec<Vec<ShipNumeric>> = vec![Vec::new(); winners.len()];
        let mut bound_ids: Vec<Vec<String>> = vec![Vec::new(); winners.len()];
        for num in layers.numerics.iter() {
            let long_code = ship_is_long_code(num);
            let pick = ship_bind_values(&layers.roles, &d1_view, num.token, &|cat: &str| {
                ship_category_accepts(cat, crate::utils::ai_utils::FieldFormat::Numeric)
                    || (long_code && ship_category_accepts(cat, crate::utils::ai_utils::FieldFormat::Identifier))
            });
            match pick {
                Some(wi) => {
                    emit_term(&format!(
                        "   🔗 [VALUE BIND] 수치 {} (연산자 {}, 통화 {}) → \"{}\" ({})",
                        num.value,
                        if num.operator.is_empty() { "-" } else { num.operator.as_str() },
                        if num.currency.is_empty() { "-" } else { num.currency.as_str() },
                        winners[wi].text, winners[wi].category
                    ));
                    bound_nums[wi].push(num.clone());
                }
                None => {
                    if long_code {
                        if !hub_values.iter().any(|x| x == &num.value) { hub_values.push(num.value.clone()); }
                        emit_term(&format!(
                            "   🧲 [UNBOUND CODE → HUB] '{}' 는 붙을 라벨 스팬이 없어 doc_number + 전 참조 축 탐색으로 보냅니다.",
                            num.value
                        ));
                    } else {
                        emit_term(&format!(
                            "   ⚪ [VALUE UNBOUND] 수치 {} 는 {}토큰 안에 수치 필드를 가진 라벨 스팬이 없어 조건을 만들지 않습니다.",
                            num.value, SHIP_BIND_RADIUS
                        ));
                    }
                }
            }
        }
        for (tok, id) in layers.identifiers.iter() {
            let pick = ship_bind_values(&layers.roles, &d1_view, *tok, &|cat: &str| {
                ship_category_accepts(cat, crate::utils::ai_utils::FieldFormat::Identifier)
            });
            match pick {
                Some(wi) => {
                    emit_term(&format!(
                        "   🔗 [VALUE BIND] 식별자 {} → \"{}\" ({})",
                        id, winners[wi].text, winners[wi].category
                    ));
                    bound_ids[wi].push(id.clone());
                }
                None => {
                    if !hub_values.iter().any(|x| x == id) { hub_values.push(id.clone()); }
                    emit_term(&format!(
                        "   🧲 [UNBOUND CODE → HUB] '{}' 는 붙을 라벨 스팬이 없어 doc_number + 전 참조 축 탐색으로 보냅니다.",
                        id
                    ));
                }
            }
        }

        let enum_code_for = |field: &str, s: usize, e: usize| -> Option<String> {
            for k in s..e {
                if let Some(hits) = layers.enum_hits.get(k) {
                    if let Some((_, code)) = hits.iter().find(|(f, _)| f == field) { return Some(code.clone()); }
                }
            }
            None
        };

        let span_surface = |wi: usize| -> String { all_words[winners[wi].start..winners[wi].end].join(" ") };

        let mut by_cat: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
        for (wi, w) in winners.iter().enumerate() { by_cat.entry(w.category.clone()).or_default().push(wi); }

        let cat_order: Vec<String> = crate::logic::TRADE_CONDITION_CATEGORIES
            .iter().map(|(c, _)| c.to_string()).collect();

        let mut d2_llm_pending: Vec<(usize, String, Vec<(String, String, f32)>)> = Vec::new();
        let mut done_span: Vec<bool> = vec![false; winners.len()];

        let value_spans: Vec<usize> = (0..winners.len())
            .filter(|&wi| winners[wi].category != "hub" && (!bound_nums[wi].is_empty() || !bound_ids[wi].is_empty()))
            .collect();
        if !value_spans.is_empty() {
            let mut g_fields: Vec<(String, String, String)> = Vec::new();
            let mut g_weights: Vec<Vec<f32>> = Vec::new();
            let mut g_phr: Vec<Vec<String>> = Vec::new();
            let mut g_prej_phr: Vec<Vec<String>> = Vec::new();
            let mut out_of_scope: Vec<String> = Vec::new();
            let mut supplemented = 0usize;
            for gcat in cat_order.iter() {
                if gcat == "hub" { continue; }
                for (fname, fdesc, anchor) in crate::logic::trade_condition_fields(gcat).iter() {
                    if claimed_fields.contains(*fname) { continue; }
                    if g_fields.iter().any(|(f, _, _)| f.as_str() == *fname) { continue; }
                    let fmt = crate::utils::ai_utils::query_value_format(fname);
                    if !matches!(
                        fmt,
                        crate::utils::ai_utils::FieldFormat::Numeric
                            | crate::utils::ai_utils::FieldFormat::Identifier
                            | crate::utils::ai_utils::FieldFormat::TrackingCode
                    ) {
                        continue;
                    }
                    if !ship_field_in_scope(fname, &final_scope) {
                        out_of_scope.push(fname.to_string());
                        continue;
                    }
                    let (mut ph, mut wt) = crate::utils::ai_utils::split_bias_phrases_weighted_full(anchor);
                    if let Some((_, aliases)) = crate::parsing::TRADE_COLUMN_ALIASES.iter().find(|(f, _)| *f == *fname) {
                        for a in aliases.iter() {
                            let a = a.trim();
                            if a.is_empty() || ph.iter().any(|p| p == a) { continue; }
                            ph.push(a.to_string());
                            wt.push(1.0);
                        }
                    }
                    let sup = crate::logic::trade_label_supplement(fname);
                    if !sup.is_empty() {
                        supplemented += crate::logic::merge_phrase_bank(&mut ph, &mut wt, &sup, 1.0);
                    }
                    if ph.is_empty() { continue; }
                    let pp = crate::utils::ai_utils::prejudice_phrase_bank_multilingual(language, "shipping_doc", fname);
                    g_fields.push((fname.to_string(), gcat.clone(), fdesc.to_string()));
                    g_weights.push(wt);
                    g_phr.push(ph);
                    g_prej_phr.push(pp);
                }
            }
            let g_banks = self.ship_embed_phrase_groups(&g_phr).await;
            let g_prejs = self.ship_embed_phrase_groups(&g_prej_phr).await;
            if !out_of_scope.is_empty() {
                emit_term(&format!(
                    "   🎯 [D2 FIELD SCOPE] 질의가 지목한 서식 {:?} 의 저장 스키마에 존재하지 않는 축 {}개를 값 경쟁에서 제외했습니다: {:?} — 저장될 수 없는 축이 1위가 되면 어떤 값을 넣어도 통과하지 못하는 하드 조건이 만들어집니다. 날짜 축에는 이미 같은 기준(DATE FIELD SCOPE)이 적용되어 있었는데 값 축에만 빠져 있었습니다.",
                    final_scope, out_of_scope.len(),
                    out_of_scope.iter().take(12).collect::<Vec<_>>()
                ));
            }
            emit_term(&format!(
                "   📐 [D2 VALUE-FIRST] 값이 결속된 스팬 {}개를 카테고리 경계 없이 수치·식별자 필드 {}개와 경쟁시킵니다. 라벨 뱅크 {}구 + 편견 뱅크 {}구를 배치 2회로 임베딩했습니다 (다국어 보강 라벨 {}구 포함, 필드마다 따로 부르면 {}회).",
                value_spans.len(), g_fields.len(),
                g_phr.iter().map(|v| v.len()).sum::<usize>(),
                g_prej_phr.iter().map(|v| v.len()).sum::<usize>(),
                supplemented,
                g_fields.len() * 2
            ));
            let g_span_texts: Vec<String> = value_spans.iter().map(|wi| winners[*wi].text.clone()).collect();
            let g_span_embs: Vec<Vec<f32>> = self
                .get_embedding_batch(g_span_texts.clone())
                .await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; g_span_texts.len()]);
            let mut g_matrix: Vec<Vec<f32>> = vec![vec![-1.0f32; value_spans.len()]; g_fields.len()];
            for (si, wi) in value_spans.iter().enumerate() {
                let e = &g_span_embs[si];
                if e.iter().all(|&v| v == 0.0) { continue; }
                for fi in 0..g_fields.len() {
                    let probe = ship_make_assignment(
                        &g_fields[fi].0, &g_fields[fi].1, &span_surface(*wi),
                        &bound_nums[*wi], &bound_ids[*wi], None, winners[*wi].score >= d1_gate,
                    );
                    if probe.is_none() { continue; }
                    let own = crate::utils::ai_utils::weighted_max_pool_sim(e, &g_banks[fi], &g_weights[fi]);
                    if !g_prejs[fi].is_empty() {
                        let prej = crate::utils::ai_utils::max_pool_sim(e, &g_prejs[fi]);
                        let coh = crate::utils::ai_utils::bank_internal_cohesion(&g_banks[fi]);
                        if crate::utils::ai_utils::prejudice_dominates(own, prej, coh) { continue; }
                    }
                    g_matrix[fi][si] = own;
                }
            }
            let g_assign = crate::utils::ai_utils::exclusive_assign_by_score(&g_matrix, 0.0, 0.0);
            for (si, wi) in value_spans.iter().enumerate() {
                let mut ranked: Vec<(usize, f32)> = (0..g_fields.len())
                    .filter(|&k| g_matrix[k][si] >= 0.0)
                    .map(|k| (k, g_matrix[k][si]))
                    .collect();
                ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                if ranked.len() < 2 { continue; }
                let (top, second) = (ranked[0], ranked[1]);
                let gap = top.1 - second.1;
                let feas: Vec<f32> = ranked.iter().map(|(_, s)| *s).collect();
                let cnt = feas.len() as f32;
                let mean = feas.iter().sum::<f32>() / cnt;
                let sd = (feas.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / cnt).sqrt();
                if gap < 0.005 {
                    emit_term(&format!(
                        "   🤝 [D2 VALUE-FIRST TIE] \"{}\" | {} ({:.4}) vs {} ({:.4}) | 격차 {:.4} — 동률 게이트 안이라 혼동 사전에는 원점수 승자가 아니라 아래 재판정의 승자를 기록합니다. 원점수 승자를 먼저 기록하면 Phase 1 에서 틀린 승자가 결정론이 됩니다.",
                        winners[*wi].text, g_fields[top.0].0, top.1, g_fields[second.0].0, second.1, gap
                    ));
                } else if sd > 1e-6 && gap < sd {
                    crate::utils::score_dynamics::record_confusion(&g_fields[top.0].0, &g_fields[second.0].0, gap);
                    emit_term(&format!(
                        "   🤝 [D2 VALUE-FIRST CLOSE CALL] \"{}\" | {} ({:.4}) vs {} ({:.4}) | 격차 {:.4} < 후보 {}개의 표준편차 {:.4} — 원점수로 판정하되 다음 회차를 위해 혼동 쌍으로 기록합니다.",
                        winners[*wi].text, g_fields[top.0].0, top.1, g_fields[second.0].0, second.1, gap, feas.len(), sd
                    ));
                } else {
                    emit_term(&format!(
                        "   ✅ [D2 VALUE-FIRST DECISIVE] \"{}\" | {} ({:.4}) 가 {} ({:.4}) 를 격차 {:.4} (후보 {}개 표준편차 {:.4}) 로 앞섭니다. 혼동 사전에 넣지 않습니다 — 동률이 아닌 쌍을 기록하면 그 쌍이 나중에 결정론적 타이브레이커로 읽혀, 경쟁한 적 없는 축이 고정 승자가 됩니다.",
                        winners[*wi].text, g_fields[top.0].0, top.1, g_fields[second.0].0, second.1, gap, feas.len(), sd
                    ));
                }
            }
            for (fi, a) in g_assign.iter().enumerate() {
                let (si, own, margin) = match a { Some(v) => *v, None => continue };
                let wi = value_spans[si];
                let feasible_n = (0..g_fields.len()).filter(|&k| g_matrix[k][si] >= 0.0).count();
                if feasible_n > 1 && margin.abs() < 0.005 {
                    let mut scored: Vec<(String, String, f32)> = Vec::new();
                    for k in 0..g_fields.len() {
                        let sc = g_matrix[k][si];
                        if sc < 0.0 { continue; }
                        scored.push((g_fields[k].0.clone(), g_fields[k].2.clone(), sc));
                    }
                    scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
                    scored.truncate(6);
                    let currency_bound = bound_nums[wi].iter().any(|n| !n.currency.is_empty());
                    let tie_floor = scored.first().map(|s| s.2 - 0.005).unwrap_or(f32::MAX);
                    let tied_total = scored.iter().filter(|(_, _, sc)| *sc >= tie_floor).count();
                    let span_is_row = crate::logic::is_trade_array_category(&winners[wi].category);
                    let tied_monetary_all: Vec<String> = scored
                        .iter()
                        .filter(|(f, _, sc)| *sc >= tie_floor && ship_is_monetary(f))
                        .map(|(f, _, _)| f.clone())
                        .collect();
                    let tied_monetary: Vec<String> = tied_monetary_all
                        .iter()
                        .filter(|f| {
                            span_is_row
                                || !ship_field_category(f.as_str())
                                    .map(|c| crate::logic::is_trade_array_category(&c))
                                    .unwrap_or(false)
                        })
                        .cloned()
                        .collect();
                    if tied_monetary_all.len() != tied_monetary.len() {
                        emit_term(&format!(
                            "   🧾 [D2 CURRENCY TIE-BREAK / ROW AXIS] \"{}\" 는 표 행 카테고리가 아닌 '{}' 스팬이라, 동률 금전 축 {:?} 중 표 행 축을 제외한 {:?} 만 문서 단위 후보로 둡니다. 행 축은 ship_make_assignment 에서도 힌트로만 내려가므로 문서 총액 조건의 후보가 될 수 없습니다.",
                            winners[wi].text, winners[wi].category, tied_monetary_all, tied_monetary
                        ));
                    }
                    let taken_elsewhere = tied_monetary
                        .first()
                        .map(|f| {
                            g_fields
                                .iter()
                                .enumerate()
                                .any(|(k, gf)| gf.0 == *f && k != fi && g_assign[k].is_some())
                        })
                        .unwrap_or(false);
                    if currency_bound && tied_total > 1 && tied_monetary.len() == 1 && !taken_elsewhere {
                        let field = tied_monetary[0].clone();
                        let fcat = ship_field_category(&field).unwrap_or_else(|| g_fields[fi].1.clone());
                        let own_sc = scored
                            .iter()
                            .find(|(f, _, _)| *f == field)
                            .map(|(_, _, s)| *s)
                            .unwrap_or(own);
                        emit_term(&format!(
                            "   💱 [D2 CURRENCY TIE-BREAK] \"{}\" | 동률 후보 {}개 {:?} 중 통화가 결속된 수치를 받을 수 있는 금전 축은 '{}' 하나뿐입니다. 통화 단위는 질의가 이미 확정한 근거이므로 LLM 재판정 없이 확정합니다.",
                            winners[wi].text, tied_total,
                            scored.iter().filter(|(_, _, sc)| *sc >= tie_floor).map(|(f, _, s)| format!("{}({:.4})", f, s)).collect::<Vec<_>>(),
                            field
                        ));
                        match ship_make_assignment(
                            &field, &fcat, &span_surface(wi),
                            &bound_nums[wi], &bound_ids[wi], None, winners[wi].score >= d1_gate,
                        ) {
                            Some(assignment) => {
                                for (f, _, sc) in scored.iter() {
                                    if *f == field || *sc < tie_floor { continue; }
                                    crate::utils::score_dynamics::record_confusion(&field, f, own_sc - *sc);
                                }
                                let (kind, op, value) = ship_apply_assignment(assignment, &mut conditions, &mut hints, &mut claimed_fields);
                                if let Some(code) = ship_companion_currency(&field, &bound_nums[wi]) {
                                    if !claimed_fields.contains("currency") {
                                        conditions.insert("currency".to_string(), json!({ "operator": "contains", "value": code }));
                                        claimed_fields.insert("currency".to_string());
                                        emit_term(&format!(
                                            "   💱 [CURRENCY COMPANION] {} 의 통화 단위 → currency contains '{}'",
                                            field, code
                                        ));
                                    }
                                }
                                for i in winners[wi].start..winners[wi].end {
                                    if i < consumed_span.len() { consumed_span[i] = true; }
                                }
                                for num in bound_nums[wi].iter() {
                                    if num.token < consumed_span.len() { consumed_span[num.token] = true; }
                                }
                                done_span[wi] = true;
                                crate::utils::score_dynamics::record_baseline("search.currency_tie_break", 1.0);
                                emit_term(&format!(
                                    "   🔗 [D2 ASSIGN / {} / VALUE-FIRST] \"{}\" ({}) → {}.{} {} '{}' | Score: {:+.4} | 근거: 통화 결속 동률 해소",
                                    kind, winners[wi].text, winners[wi].category, fcat, field, op, value, own_sc
                                ));
                                continue;
                            }
                            None => {}
                        }
                    }
                    crate::utils::score_dynamics::record_baseline("search.currency_tie_break", 0.0);
                    d2_llm_pending.push((wi, g_fields[fi].1.clone(), scored));
                    done_span[wi] = true;
                    emit_term(&format!(
                        "   ⚖️ [D2 MARGIN GATE] \"{}\" 의 값 필드 1위-2위 마진 {:+.4} 로 LLM 재판정 대기열에 넣습니다.",
                        winners[wi].text, margin
                    ));
                    continue;
                }
                let (field, fcat) = (g_fields[fi].0.clone(), g_fields[fi].1.clone());
                let assignment = match ship_make_assignment(
                    &field, &fcat, &span_surface(wi),
                    &bound_nums[wi], &bound_ids[wi], None, winners[wi].score >= d1_gate,
                ) {
                    Some(v) => v,
                    None => continue,
                };
                let (kind, op, value) = ship_apply_assignment(assignment, &mut conditions, &mut hints, &mut claimed_fields);
                if let Some(code) = ship_companion_currency(&field, &bound_nums[wi]) {
                    if !claimed_fields.contains("currency") {
                        conditions.insert("currency".to_string(), json!({ "operator": "contains", "value": code }));
                        claimed_fields.insert("currency".to_string());
                        emit_term(&format!(
                            "   💱 [CURRENCY COMPANION] {} 의 통화 단위 → currency contains '{}'",
                            field, code
                        ));
                    }
                }
                for i in winners[wi].start..winners[wi].end {
                    if i < consumed_span.len() { consumed_span[i] = true; }
                }
                for num in bound_nums[wi].iter() {
                    if num.token < consumed_span.len() { consumed_span[num.token] = true; }
                }
                done_span[wi] = true;
                emit_term(&format!(
                    "   🔗 [D2 ASSIGN / {} / VALUE-FIRST] \"{}\" ({}) → {}.{} {} '{}' | Score: {:+.4} | Margin: {:+.4}",
                    kind, winners[wi].text, winners[wi].category, fcat, field, op, value, own, margin
                ));
            }
        }

        for cat in cat_order.iter() {
            let span_idxs: Vec<usize> = match by_cat.get(cat) {
                Some(v) => v.iter().copied().filter(|wi| !done_span[*wi]).collect(),
                None => continue,
            };
            if span_idxs.is_empty() { continue; }

            if cat == "hub" {
                for wi in span_idxs.iter() {
                    emit_term(&format!(
                        "   🧲 [D2 HUB] \"{}\" | 추적 의도 {} | 추적 번호 {}건",
                        winners[*wi].text, hub_intent, hub_values.len()
                    ));
                }
                continue;
            }

            let fields = crate::logic::trade_condition_fields(cat);
            if fields.is_empty() { continue; }

            let mut f_names: Vec<String> = Vec::new();
            let mut f_descs: Vec<String> = Vec::new();
            let mut f_weights: Vec<Vec<f32>> = Vec::new();
            let mut f_phr: Vec<Vec<String>> = Vec::new();
            let mut f_prej_phr: Vec<Vec<String>> = Vec::new();
            let mut f_out_of_scope: Vec<String> = Vec::new();
            let mut f_supplemented = 0usize;
            for (fname, fdesc, anchor) in fields.iter() {
                if claimed_fields.contains(*fname) { continue; }
                if !ship_field_in_scope(fname, &final_scope) {
                    f_out_of_scope.push(fname.to_string());
                    continue;
                }
                let (mut ph, mut wt) = crate::utils::ai_utils::split_bias_phrases_weighted_full(anchor);
                if let Some((_, aliases)) = crate::parsing::TRADE_COLUMN_ALIASES.iter().find(|(f, _)| *f == *fname) {
                    for a in aliases.iter() {
                        let a = a.trim();
                        if a.is_empty() || ph.iter().any(|p| p == a) { continue; }
                        ph.push(a.to_string());
                        wt.push(1.0);
                    }
                }
                let sup = crate::logic::trade_label_supplement(fname);
                if !sup.is_empty() {
                    f_supplemented += crate::logic::merge_phrase_bank(&mut ph, &mut wt, &sup, 1.0);
                }
                if ph.is_empty() { continue; }
                let pp = crate::utils::ai_utils::prejudice_phrase_bank_multilingual(language, "shipping_doc", fname);
                f_names.push(fname.to_string());
                f_descs.push(fdesc.to_string());
                f_weights.push(wt);
                f_phr.push(ph);
                f_prej_phr.push(pp);
            }
            if f_names.is_empty() {
                if !f_out_of_scope.is_empty() {
                    emit_term(&format!(
                        "   ⚪ [D2 BANK SKIP] 카테고리 '{}' 의 후보 필드가 전부 서식 {:?} 의 저장 스키마 밖입니다: {:?}",
                        cat, final_scope, f_out_of_scope
                    ));
                }
                continue;
            }
            let f_banks = self.ship_embed_phrase_groups(&f_phr).await;
            let f_prejs = self.ship_embed_phrase_groups(&f_prej_phr).await;

            emit_term(&format!(
                "   📐 [D2 BANK] 카테고리 '{}' | 후보 필드 {}개 (스키마 밖 {}개 제외) | 대상 스팬 {}개 | 라벨 {}구 + 편견 {}구를 배치 2회로 임베딩 (다국어 보강 {}구 포함)",
                cat, f_names.len(), f_out_of_scope.len(), span_idxs.len(),
                f_phr.iter().map(|v| v.len()).sum::<usize>(),
                f_prej_phr.iter().map(|v| v.len()).sum::<usize>(),
                f_supplemented
            ));

            let mut matrix: Vec<Vec<f32>> = vec![vec![-1.0f32; span_idxs.len()]; f_names.len()];
            let span_texts: Vec<String> = span_idxs.iter().map(|wi| winners[*wi].text.clone()).collect();
            let span_embs: Vec<Vec<f32>> = self
                .get_embedding_batch(span_texts.clone())
                .await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; span_texts.len()]);

            for (si, wi) in span_idxs.iter().enumerate() {
                let e = &span_embs[si];
                if e.iter().all(|&v| v == 0.0) { continue; }
                let mut feasible = 0usize;
                for fi in 0..f_names.len() {
                    let probe = ship_make_assignment(
                        &f_names[fi], cat, &span_surface(*wi),
                        &bound_nums[*wi], &bound_ids[*wi],
                        enum_code_for(&f_names[fi], winners[*wi].start, winners[*wi].end).as_deref(),
                        winners[*wi].score >= d1_gate,
                    );
                    if probe.is_none() { continue; }
                    let own = crate::utils::ai_utils::weighted_max_pool_sim(e, &f_banks[fi], &f_weights[fi]);
                    if !f_prejs[fi].is_empty() {
                        let prej = crate::utils::ai_utils::max_pool_sim(e, &f_prejs[fi]);
                        let coh = crate::utils::ai_utils::bank_internal_cohesion(&f_banks[fi]);
                        if crate::utils::ai_utils::prejudice_dominates(own, prej, coh) {
                            emit_term(&format!(
                                "      🚫 [D2 PREJUDICE] \"{}\" → {} | Own: {:.4} | Prej: {:.4} | Cohesion: {:.4}",
                                winners[*wi].text, f_names[fi], own, prej, coh
                            ));
                            continue;
                        }
                    }
                    matrix[fi][si] = own;
                    feasible += 1;
                }
                if feasible == 0 {
                    let significant = winners[*wi].score >= d1_gate;
                    emit_term(&format!(
                        "   ⚪ [D2 NO VALUE EVIDENCE] \"{}\" ({}) 에는 이 카테고리의 어떤 필드도 받을 수 있는 값 근거가 없습니다. 결속된 수치 {}건 · 식별자 {}건 · D1 {:+.4} {} 게이트 {:.3}. 자유서술 축(Text/Address)은 D1 이 게이트를 넘어야만 열리고, 수치·식별자 축은 결속된 값이 있어야만 열립니다.",
                        winners[*wi].text, cat,
                        bound_nums[*wi].len(), bound_ids[*wi].len(),
                        winners[*wi].score,
                        if significant { "≥" } else { "<" },
                        d1_gate
                    ));
                }
            }

            let centered = crate::utils::ai_utils::double_center_matrix(&matrix);
            let assign = crate::utils::ai_utils::exclusive_assign_by_score(&centered, 0.0, 0.0);

            for (fi, a) in assign.iter().enumerate() {
                let (si, own, margin) = match a { Some(v) => *v, None => continue };
                let wi = span_idxs[si];
                let feasible_n = (0..f_names.len()).filter(|&k| matrix[k][si] >= 0.0).count();

                if feasible_n > 1 && margin.abs() < 0.005 {
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
                let mut surface = span_surface(wi);
                if matches!(
                    crate::utils::ai_utils::query_value_format(&field),
                    crate::utils::ai_utils::FieldFormat::Text | crate::utils::ai_utils::FieldFormat::Address
                ) {
                    if let Some((span_val, span_lab, picked)) = self
                        .ship_label_span_value(
                            &field,
                            winners[wi].start,
                            winners[wi].end,
                            &all_words,
                            &layers.variants,
                            &layers.roles,
                            &consumed_span,
                            &f_banks[fi],
                            &f_weights[fi],
                        )
                        .await
                    {
                        match picked {
                            Some((k, value, val, lab)) => {
                                emit_term(&format!(
                                    "   🏷️ [D2 LABEL SPAN → VALUE] \"{}\" 는 '{}' 의 라벨입니다 (라벨 뱅크 {:.4} > 값 뱅크 {:.4}). 라벨 단어를 값으로 두면 힌트가 '{} contains {}' 가 되어 저장값과 만날 수 없고, 원장에는 그 축이 매번 만족 0 으로 쌓여 다음 회차의 강등 판정을 오염시킵니다. 맞닿은 내용어 \"{}\" (값 뱅크 {:.4} > 라벨 뱅크 {:.4}) 를 값으로 씁니다.",
                                    span_surface(wi), field, span_lab, span_val, field, span_surface(wi), value, val, lab
                                ));
                                crate::utils::score_dynamics::record_baseline("search.label_span_value", 1.0);
                                if k < consumed_span.len() { consumed_span[k] = true; }
                                surface = value;
                            }
                            None => {
                                emit_term(&format!(
                                    "   ⚪ [D2 LABEL SPAN / NO VALUE] \"{}\" 는 '{}' 의 라벨입니다 (라벨 뱅크 {:.4} > 값 뱅크 {:.4}). 맞닿은 열린 내용어 중 값 뱅크가 라벨 뱅크와 이 스팬 자신을 둘 다 넘는 토큰이 없어 기존 표면형을 그대로 씁니다. 힌트는 필터가 아니므로 결과를 바꾸지 않고, 이 비율은 SDS search.label_span_value 로 관측합니다.",
                                    span_surface(wi), field, span_lab, span_val
                                ));
                                crate::utils::score_dynamics::record_baseline("search.label_span_value", 0.0);
                            }
                        }
                    }
                }
                let assignment = match ship_make_assignment(
                    &field, cat, &surface,
                    &bound_nums[wi], &bound_ids[wi],
                    enum_code_for(&field, winners[wi].start, winners[wi].end).as_deref(),
                    winners[wi].score >= d1_gate,
                ) {
                    Some(v) => v,
                    None => continue,
                };
                let (kind, op, value) = ship_apply_assignment(assignment, &mut conditions, &mut hints, &mut claimed_fields);
                if let Some(code) = ship_companion_currency(&field, &bound_nums[wi]) {
                    if !claimed_fields.contains("currency") {
                        conditions.insert("currency".to_string(), json!({ "operator": "contains", "value": code }));
                        claimed_fields.insert("currency".to_string());
                        emit_term(&format!(
                            "   💱 [CURRENCY COMPANION] {} 의 통화 단위 → currency contains '{}'",
                            field, code
                        ));
                    }
                }
                for i in winners[wi].start..winners[wi].end {
                    if i < consumed_span.len() { consumed_span[i] = true; }
                }
                for num in bound_nums[wi].iter() {
                    if num.token < consumed_span.len() { consumed_span[num.token] = true; }
                }
                emit_term(&format!(
                    "   🔗 [D2 ASSIGN / {}] \"{}\" → {}.{} {} '{}' | Score: {:+.4} | Margin: {:+.4}",
                    kind, winners[wi].text, cat, field, op, value, own, margin
                ));
            }
        }

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

                let picked_cat = ship_field_category(&picked).unwrap_or_else(|| cat.clone());
                let assignment = match ship_make_assignment(
                    &picked, &picked_cat, &span_surface(wi),
                    &bound_nums[wi], &bound_ids[wi],
                    enum_code_for(&picked, winners[wi].start, winners[wi].end).as_deref(),
                    winners[wi].score >= d1_gate,
                ) {
                    Some(v) => v,
                    None => {
                        emit_term(&format!("   ⚪ [D2 NO VALUE] '{}' 에 붙은 값 근거가 없어 조건을 만들지 않습니다.", picked));
                        continue;
                    }
                };
                let (kind, op, value) = ship_apply_assignment(assignment, &mut conditions, &mut hints, &mut claimed_fields);
                if let Some(code) = ship_companion_currency(&picked, &bound_nums[wi]) {
                    if !claimed_fields.contains("currency") {
                        conditions.insert("currency".to_string(), json!({ "operator": "contains", "value": code }));
                        claimed_fields.insert("currency".to_string());
                        emit_term(&format!(
                            "   💱 [CURRENCY COMPANION] {} 의 통화 단위 → currency contains '{}'",
                            picked, code
                        ));
                    }
                }
                for i in winners[wi].start..winners[wi].end {
                    if i < consumed_span.len() { consumed_span[i] = true; }
                }
                for num in bound_nums[wi].iter() {
                    if num.token < consumed_span.len() { consumed_span[num.token] = true; }
                }
                let picked_sc = scored
                    .iter()
                    .find(|(f, _, _)| *f == picked)
                    .map(|(_, _, s)| *s)
                    .unwrap_or(0.0);
                for (f, _, sc) in scored.iter() {
                    if *f == picked { continue; }
                    crate::utils::score_dynamics::record_confusion(&picked, f, picked_sc - *sc);
                }
                emit_term(&format!(
                    "   🤖 [D2 LLM / {}] \"{}\" → {} {} '{}' | 혼동 사전에 LLM 판정 승자로 기록 (후보 {}개)",
                    kind, winners[wi].text, picked, op, value, scored.len()
                ));
            }
        }

        {
            let orphan: Vec<usize> = (0..winners.len())
                .filter(|&wi| {
                    let w = &winners[wi];
                    w.category != "hub"
                        && bound_nums[wi].is_empty()
                        && bound_ids[wi].is_empty()
                        && w.score >= d1_gate
                        && (w.start..w.end).all(|i| !consumed_span.get(i).copied().unwrap_or(false))
                })
                .collect();
            if !orphan.is_empty() {
                let mut lab_names: Vec<String> = Vec::new();
                let mut lab_phr: Vec<Vec<String>> = Vec::new();
                let mut lab_wt: Vec<Vec<f32>> = Vec::new();
                let mut t_fields: Vec<(String, String, usize)> = Vec::new();
                let mut t_value_phr: Vec<Vec<String>> = Vec::new();
                let mut t_prej_phr: Vec<Vec<String>> = Vec::new();
                let mut t_out_of_scope: Vec<String> = Vec::new();
                for gcat in cat_order.iter() {
                    if gcat == "hub" { continue; }
                    for (fname, _, anchor) in crate::logic::trade_condition_fields(gcat).iter() {
                        if lab_names.iter().any(|f| f.as_str() == *fname) { continue; }
                        if !ship_field_in_scope(fname, &final_scope) {
                            t_out_of_scope.push(fname.to_string());
                            continue;
                        }
                        let (mut ph, mut wt) = crate::utils::ai_utils::split_bias_phrases_weighted_full(anchor);
                        if let Some((_, aliases)) = crate::parsing::TRADE_COLUMN_ALIASES.iter().find(|(f, _)| *f == *fname) {
                            for a in aliases.iter() {
                                let a = a.trim();
                                if a.is_empty() || ph.iter().any(|p| p == a) { continue; }
                                ph.push(a.to_string());
                                wt.push(1.0);
                            }
                        }
                        let sup = crate::logic::trade_label_supplement(fname);
                        if !sup.is_empty() {
                            crate::logic::merge_phrase_bank(&mut ph, &mut wt, &sup, 1.0);
                        }
                        if ph.is_empty() { continue; }
                        let gi = lab_names.len();
                        lab_names.push(fname.to_string());
                        lab_phr.push(ph);
                        lab_wt.push(wt);
                        if claimed_fields.contains(*fname) { continue; }
                        if !matches!(
                            crate::utils::ai_utils::query_value_format(fname),
                            crate::utils::ai_utils::FieldFormat::Text | crate::utils::ai_utils::FieldFormat::Address
                        ) {
                            continue;
                        }
                        let vp = crate::utils::ai_utils::multilingual_value_anchor_phrases_scoped("shipping_doc", fname);
                        if vp.is_empty() { continue; }
                        t_fields.push((fname.to_string(), gcat.clone(), gi));
                        t_value_phr.push(vp);
                        t_prej_phr.push(crate::utils::ai_utils::prejudice_phrase_bank_multilingual(language, "shipping_doc", fname));
                    }
                }
                let lab_banks = self.ship_embed_phrase_groups(&lab_phr).await;
                let t_value = self.ship_embed_phrase_groups(&t_value_phr).await;
                let t_prej = self.ship_embed_phrase_groups(&t_prej_phr).await;
                let g_label: Vec<(String, Vec<Vec<f32>>, Vec<f32>)> = lab_names
                    .into_iter()
                    .zip(lab_banks.into_iter())
                    .zip(lab_wt.into_iter())
                    .map(|((f, b), w)| (f, b, w))
                    .collect();
                if !t_out_of_scope.is_empty() {
                    emit_term(&format!(
                        "   🎯 [TEXT-FIRST LABEL SCOPE] 라벨 최고점을 겨루는 스키마 축에서, 서식 {:?} 의 저장 스키마 밖인 {}개를 뺐습니다: {:?} — 검사·증명 서식 전용 축(계량일 등)이 상용송장 질의의 라벨 최고점을 가져가면, 값 축이 아무리 잘 맞아도 '라벨을 말한 스팬' 으로 판정되어 필터가 만들어지지 않습니다.",
                        final_scope, t_out_of_scope.len(),
                        t_out_of_scope.iter().take(12).collect::<Vec<_>>()
                    ));
                }
                if t_fields.is_empty() {
                    emit_term(&format!(
                        "   ⚪ [D2 TEXT-FIRST SKIP] 값 뱅크(multilingual_value_anchor)를 가진 자유서술 축이 없어, D1 게이트를 넘고도 필드를 얻지 못한 스팬 {}개를 그대로 둡니다: {:?}",
                        orphan.len(),
                        orphan.iter().map(|wi| winners[*wi].text.clone()).collect::<Vec<_>>()
                    ));
                } else {
                    emit_term(&format!(
                        "   📐 [D2 TEXT-FIRST] D1 게이트를 넘었지만 자기 카테고리에서 필드를 얻지 못한 자유서술 스팬 {}개를 카테고리 경계 없이 값 뱅크를 가진 Text/Address 필드 {}개와 경쟁시킵니다. 라벨 단어인지의 판정은 후보 축 하나가 아니라 스키마 전체 라벨 뱅크 {}개의 최고점과 비교합니다. 값 뱅크가 그 최고점을 넘지 못하는 스팬은 어느 축의 라벨을 말한 것이므로 필터로 만들지 않습니다.",
                        orphan.len(), t_fields.len(), g_label.len()
                    ));
                    let o_texts: Vec<String> = orphan.iter().map(|wi| winners[*wi].text.clone()).collect();
                    let o_embs: Vec<Vec<f32>> = self
                        .get_embedding_batch(o_texts.clone())
                        .await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; o_texts.len()]);
                    let mut matrix: Vec<Vec<f32>> = vec![vec![-1.0f32; orphan.len()]; t_fields.len()];
                    let mut span_lab: Vec<(String, f32)> = vec![(String::new(), f32::MIN); orphan.len()];
                    let mut best_val: Vec<(String, f32)> = vec![(String::new(), f32::MIN); orphan.len()];
                    let mut lab_gate_of: Vec<f32> = vec![f32::MIN; orphan.len()];
                    let mut val_z_of: Vec<f32> = vec![f32::MIN; orphan.len()];
                    let lab_draw = crate::utils::ai_utils::gumbel_expected_z(g_label.len().max(1));
                    let val_draw = crate::utils::ai_utils::gumbel_expected_z(t_fields.len().max(1));
                    emit_term(&format!(
                        "   📏 [TEXT-FIRST SAMPLE CORRECTION] 라벨 축은 뱅크 {}개에서 뽑은 최댓값이고 값 축은 뱅크 {}개에서 뽑은 최댓값입니다. 보정 없이 비교하면 뱅크 수가 많은 라벨 쪽이 구조적으로 이겨 모든 스팬이 '라벨을 말한 것' 으로 판정됩니다. 두 최댓값을 같은 분포로 표준화한 뒤 각자의 √(2 ln N) 기대 최댓값({:.3} vs {:.3})을 차감해 비교합니다.",
                        g_label.len(), t_fields.len(), lab_draw, val_draw
                    ));
                    for si in 0..orphan.len() {
                        let e = &o_embs[si];
                        if e.iter().all(|&v| v == 0.0) { continue; }
                        let lab_scores: Vec<f32> = g_label
                            .iter()
                            .map(|(_, le, wt)| crate::utils::ai_utils::weighted_max_pool_sim(e, le, wt))
                            .collect();
                        let val_scores: Vec<f32> = (0..t_fields.len())
                            .map(|fi| crate::utils::ai_utils::max_pool_sim(e, &t_value[fi]))
                            .collect();
                        let (mut lab_global, mut lab_field) = (f32::MIN, String::new());
                        for (gi, s) in lab_scores.iter().enumerate() {
                            if *s > lab_global {
                                lab_global = *s;
                                lab_field = g_label[gi].0.clone();
                            }
                        }
                        span_lab[si] = (lab_field, lab_global);
                        let mut pool: Vec<f32> = lab_scores.clone();
                        pool.extend(val_scores.iter().copied());
                        let cnt = pool.len() as f32;
                        let mean = pool.iter().sum::<f32>() / cnt;
                        let sd = (pool.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / cnt).sqrt();
                        let corrected = sd > 1e-6;
                        let lab_gate = if corrected {
                            (lab_global - mean) / sd - lab_draw
                        } else {
                            f32::MIN
                        };
                        lab_gate_of[si] = lab_gate;
                        for fi in 0..t_fields.len() {
                            let val = val_scores[fi];
                            if val > best_val[si].1 {
                                best_val[si] = (t_fields[fi].0.clone(), val);
                            }
                            let val_z = if corrected { (val - mean) / sd - val_draw } else { val };
                            if val > best_val[si].1 || val_z > val_z_of[si] {
                                val_z_of[si] = val_z_of[si].max(val_z);
                            }
                            let pass = if corrected { val_z > lab_gate } else { val > lab_global };
                            if !pass { continue; }
                            if !t_prej[fi].is_empty() {
                                let prej = crate::utils::ai_utils::max_pool_sim(e, &t_prej[fi]);
                                let coh = crate::utils::ai_utils::bank_internal_cohesion(&t_value[fi]);
                                if crate::utils::ai_utils::prejudice_dominates(val, prej, coh) { continue; }
                            }
                            matrix[fi][si] = val;
                        }
                    }
                    let assign = crate::utils::ai_utils::exclusive_assign_by_score(&matrix, 0.0, 0.0);
                    let mut routed: Vec<bool> = vec![false; orphan.len()];
                    for (fi, a) in assign.iter().enumerate() {
                        let (si, own, margin) = match a { Some(v) => *v, None => continue };
                        let wi = orphan[si];
                        let (field, fcat, gi) = (t_fields[fi].0.clone(), t_fields[fi].1.clone(), t_fields[fi].2);
                        let assignment = match ship_make_assignment(&field, &fcat, &span_surface(wi), &[], &[], None, true) {
                            Some(v) => v,
                            None => continue,
                        };
                        let lab_own = crate::utils::ai_utils::weighted_max_pool_sim(&o_embs[si], &g_label[gi].1, &g_label[gi].2);
                        let (kind, op, value) = ship_apply_assignment(assignment, &mut conditions, &mut hints, &mut claimed_fields);
                        for i in winners[wi].start..winners[wi].end {
                            if i < consumed_span.len() { consumed_span[i] = true; }
                        }
                        routed[si] = true;
                        crate::utils::score_dynamics::record_baseline("search.text_first_route", 1.0);
                        emit_term(&format!(
                            "   🔗 [D2 ASSIGN / {} / TEXT-FIRST] \"{}\" ({}) → {}.{} {} '{}' | 값 뱅크 {:.4} > 전 스키마 라벨 최고 '{}' {:.4} (자기 라벨 {:.4}) | Margin: {:+.4}",
                            kind, winners[wi].text, winners[wi].category, fcat, field, op, value, own,
                            span_lab[si].0, span_lab[si].1, lab_own, margin
                        ));
                    }
                    for (si, wi) in orphan.iter().enumerate() {
                        if routed[si] { continue; }
                        crate::utils::score_dynamics::record_baseline("search.text_first_route", 0.0);
                        let (lf, lg) = &span_lab[si];
                        let (bf, bv) = &best_val[si];
                        let lab_shown = if *lg == f32::MIN { 0.0 } else { *lg };
                        let val_shown = if *bv == f32::MIN { 0.0 } else { *bv };
                        let why = if val_z_of[si] != f32::MIN && lab_gate_of[si] != f32::MIN {
                            format!(
                                "표본 수 보정 후 값 축 z {:+.3} 가 라벨 축 z {:+.3} 를 넘지 못해 라벨을 말한 스팬으로 봅니다. (원점수: 라벨 최고 '{}' {:.4} vs 값 최고 '{}' {:.4})",
                                val_z_of[si], lab_gate_of[si], lf, lab_shown, bf, val_shown
                            )
                        } else if *bv != f32::MIN && *bv <= *lg {
                            format!("전 스키마 라벨 뱅크 최고 '{}' {:.4} 가 값 뱅크 최고 '{}' {:.4} 를 이겨 라벨을 말한 스팬으로 봅니다. (분산이 0 이라 표본 수 보정을 적용하지 못했습니다)", lf, lab_shown, bf, val_shown)
                        } else {
                            "값 뱅크 우세 축이 없거나 편견에 밀려 배정하지 않습니다.".to_string()
                        };
                        emit_term(&format!(
                            "   ⚪ [D2 TEXT-FIRST ORPHAN] \"{}\" ({}) | 최고 값 뱅크 '{}' {:.4} | 전 스키마 라벨 최고 '{}' {:.4} — {}",
                            winners[*wi].text, winners[*wi].category,
                            if bf.is_empty() { "-" } else { bf.as_str() },
                            val_shown,
                            if lf.is_empty() { "-" } else { lf.as_str() },
                            lab_shown,
                            why
                        ));
                    }
                }
            }
        }

        if !hints.is_empty() {
            let hint_fields: Vec<String> = hints.keys().cloned().collect();
            for field in hint_fields.into_iter() {
                if cancel_token.load(std::sync::atomic::Ordering::Relaxed) { break; }
                let raw = match hints.get(&field).and_then(|v| v.get("value")).and_then(|v| v.as_str()) {
                    Some(s) => s.trim().to_string(),
                    None => continue,
                };
                if raw.is_empty() { continue; }
                let words: Vec<String> = raw
                    .split_whitespace()
                    .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
                    .filter(|w| !w.is_empty())
                    .collect();
                // 단어가 하나뿐이면 걷어낼 라벨이 없습니다.
                if words.len() < 2 { continue; }

                let (mut lp, _) = crate::utils::ai_utils::label_phrase_bank_multilingual(language, "shipping_doc", &field);
                if let Some(cat) = ship_field_category(&field) {
                    if let Some((_, _, anchor)) = crate::logic::trade_condition_fields(&cat)
                        .iter()
                        .find(|(f, _, _)| *f == field.as_str())
                    {
                        for p in crate::utils::ai_utils::split_bias_phrases_full(anchor) {
                            if crate::utils::ai_utils::is_value_example_phrase(&p) { continue; }
                            if !lp.iter().any(|e| e == &p) { lp.push(p); }
                        }
                    }
                }
                {
                    let before = lp.len();
                    for p in crate::logic::trade_label_supplement(&field).into_iter() {
                        if crate::utils::ai_utils::is_value_example_phrase(&p) { continue; }
                        if lp.iter().any(|e| e.eq_ignore_ascii_case(&p)) { continue; }
                        lp.push(p);
                    }
                    if lp.len() > before {
                        emit_term(&format!(
                            "   🏷️ [HINT LABEL BANK / ML] {} 의 라벨 뱅크를 {}구 → {}구 로 넓혔습니다. 잔차화는 '이 단어가 라벨인가 값인가' 를 라벨 뱅크와의 코사인으로 가르는데, 뱅크가 영어뿐이면 '제조된'·'원산지' 같은 비영어 라벨어가 값 쪽으로 읽혀 값 문자열에 그대로 남고, 그 상태로 청크 검색에 들어가 신호가 희석됩니다.",
                            field, before, lp.len()
                        ));
                    }
                }
                if lp.is_empty() { continue; }
                let vp = crate::utils::ai_utils::multilingual_value_anchor_phrases_scoped("shipping_doc", &field);

                let lb = self.get_embedding_batch(lp).await.unwrap_or_default();
                let vb = if vp.is_empty() {
                    Vec::new()
                } else {
                    self.get_embedding_batch(vp).await.unwrap_or_default()
                };
                let we = self.get_embedding_batch(words.clone()).await.unwrap_or_default();
                if lb.is_empty() || we.len() != words.len() { continue; }

                let mut kept: Vec<String> = Vec::new();
                let mut dropped: Vec<String> = Vec::new();

                if !vb.is_empty() {
                    for (w, e) in words.iter().zip(we.iter()) {
                        if e.iter().all(|&x| x == 0.0) { kept.push(w.clone()); continue; }
                        let lab = crate::utils::ai_utils::max_pool_sim(e, &lb);
                        let val = crate::utils::ai_utils::max_pool_sim(e, &vb);
                        if val > lab {
                            kept.push(w.clone());
                        } else {
                            dropped.push(format!("{}(라벨 {:.4} ≥ 값 {:.4})", w, lab, val));
                        }
                    }
                    if kept.is_empty() {
                        emit_term(&format!(
                            "   ⚪ [HINT RESIDUAL KEEP] {} 의 값 뱅크가 모든 단어를 라벨로 판정했습니다. 잔차가 비면 조건 자체가 사라져 리콜을 잃으므로 원문 \"{}\" 를 그대로 씁니다.",
                            field, raw
                        ));
                        continue;
                    }
                } else {
                    let mut lab: Vec<(usize, f32)> = Vec::new();
                    for (i, e) in we.iter().enumerate() {
                        if e.iter().all(|&x| x == 0.0) { continue; }
                        lab.push((i, crate::utils::ai_utils::max_pool_sim(e, &lb)));
                    }
                    if lab.len() < 2 { continue; }
                    lab.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                    // 🌟 [SMALL POOL FALLBACK] 단어가 둘뿐이면 꼬리가 1개라 표준편차가 0 이고,
                    //    '평균 + 표준편차' 게이트는 수학적으로 절대 통과하지 못합니다.
                    //    실측: "중국에서 제조된" → 최고 0.8295, 나머지 평균 0.6680, 표준편차 0.0000 → 기각.
                    //    라벨 토큰이 값에 붙은 채 청크 검색에 들어가 STAGE-4C 가 0건이 되었습니다.
                    //    이 코드베이스는 RECOVERY BUDGET 과 OPERATOR SPLIT 에서 이미 같은 판단을 했습니다:
                    //    원소가 둘뿐인 풀에서 자기 분포 이상치 판정은 정의되지 않으므로,
                    //    '엄격한 argmax' 라는 더 약하지만 성립하는 근거로 대체합니다.
                    //    잔차가 비면 어차피 아래에서 원문을 유지하므로 값을 잃을 위험이 없습니다.
                    let (decisive, why) = if lab.len() >= 3 {
                        let tail: Vec<f32> = lab[1..].iter().map(|(_, s)| *s).collect();
                        let n = tail.len() as f32;
                        let mean = tail.iter().sum::<f32>() / n;
                        let sd = (tail.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n).sqrt();
                        if sd <= 1e-6 {
                            (lab[0].1 > lab[1].1,
                             format!("꼬리 표준편차가 0 이라 엄격 argmax 로 판정 (1위 {:.4} vs 2위 {:.4})", lab[0].1, lab[1].1))
                        } else {
                            (lab[0].1 - mean >= sd,
                             format!("자기 분포 이상치 (1위 {:.4} vs 나머지 평균 {:.4} + 표준편차 {:.4})", lab[0].1, mean, sd))
                        }
                    } else {
                        (lab[0].1 > lab[1].1,
                         format!("후보가 {}개뿐이라 분포 판정 불가 → 엄격 argmax (1위 {:.4} vs 2위 {:.4})", lab.len(), lab[0].1, lab[1].1))
                    };

                    if !decisive {
                        emit_term(&format!(
                            "   ⚪ [HINT RESIDUAL SKIP] {} 의 값 뱅크가 bias.json 에 없어 라벨 뱅크만으로 판정했으나 근거가 서지 않습니다. {} — 어느 단어가 라벨인지 단정할 수 없으므로 원문을 그대로 씁니다.",
                            field, why
                        ));
                        continue;
                    }
                    let li = lab[0].0;
                    for (i, w) in words.iter().enumerate() {
                        if i == li {
                            dropped.push(format!("{}({})", w, why));
                        } else {
                            kept.push(w.clone());
                        }
                    }
                    emit_term(&format!(
                        "   🧪 [HINT RESIDUAL / LABEL OUTLIER] {} 는 bias.json 의 multilingual_value_anchor 에 이 서식용 값 축이 없어 값 뱅크를 세울 수 없습니다. 대신 라벨 뱅크와의 유사도가 가장 높은 단어 하나만 라벨로 보고 걷어냅니다. 근거: {}. 값 뱅크가 없다는 이유로 잔차화를 통째로 건너뛰면, 값과 라벨이 섞인 문자열이 그대로 청크 검색에 들어가 신호가 희석됩니다.",
                        field, why
                    ));
                }
                if dropped.is_empty() {
                    emit_term(&format!(
                        "   ⚪ [HINT RESIDUAL PASS] {} 힌트 값 \"{}\" 의 단어 {}개가 전부 값 뱅크 쪽으로 읽혀 걷어낼 라벨 토큰이 없습니다. 값 뱅크의 값 예시 문장 안에 라벨성 서술어가 함께 들어 있으면 그 단어가 값으로 읽혀 여기서 살아남습니다. 원문을 그대로 씁니다.",
                        field, raw, words.len()
                    ));
                    continue;
                }
                if kept.is_empty() { continue; }

                let residual = kept.join(" ");
                emit_term(&format!(
                    "   🧪 [HINT RESIDUAL] {} 힌트 값 \"{}\" → \"{}\" | 걷어낸 라벨 토큰 {:?} — 값 토큰과 라벨 토큰이 한 문자열에 섞이면 다국어 임베딩이 저장값과 연결되어야 할 신호가 라벨 쪽으로 희석되어, 청크 property 타겟 검색이 0건으로 떨어집니다.",
                    field, raw, residual, dropped
                ));
                crate::utils::score_dynamics::record_baseline(
                    "search.hint_residual_dropped",
                    dropped.len() as f32,
                );
                if let Some(h) = hints.get_mut(&field).and_then(|v| v.as_object_mut()) {
                    h.insert("value".to_string(), json!(residual));
                }
            }
        }

        // =====================================================================
        // STEP 10 : 벡터 근거가 전무하면 레거시 폴백 1회
        // =====================================================================
        {
            let hard_fields: Vec<String> = conditions.keys().cloned().collect();
            for f in hard_fields.into_iter() {
                if let Some(reason) = ship_sds_demotion_reason(&f, &final_scope) {
                    if let Some(v) = conditions.remove(&f) {
                        hints.insert(f.clone(), v);
                    }
                    crate::utils::score_dynamics::record_search_demotion(&f);
                    emit_term(&format!("   🧮 [SDS DEMOTE] '{}' 하드 조건을 힌트로 내립니다 | {}", f, reason));
                }
            }
            for f in conditions.keys() {
                crate::utils::score_dynamics::record_search_proposal(f, true);
            }
            for f in hints.keys() {
                crate::utils::score_dynamics::record_search_proposal(f, false);
            }
        }
        if conditions.is_empty() && hub_values.is_empty() && hints.is_empty() && final_scope.is_empty() && projection.is_empty() {
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

        if !final_scope.is_empty() {
            let mut narrowed: Vec<String> = Vec::new();
            for c in final_scope.iter() {
                let up = c.to_uppercase();
                let lo = c.to_lowercase();
                if !narrowed.iter().any(|x| x == &up) { narrowed.push(up); }
                if !narrowed.iter().any(|x| x == &lo) { narrowed.push(lo); }
            }
            emit_term(&format!(
                "   🎯 [DOC SCOPE NARROW] 질의가 지목한 서식 {:?} 로 스코프를 좁힙니다. ({}종 → {}종)",
                final_scope, trade_types.len(), narrowed.len()
            ));
            trade_types = narrowed;
        }

        // 🌟 [HUB EXPANSION] 허브 번호는 어느 참조 축에 들어 있을지 알 수 없습니다.
        //    그래서 '모든 참조 축 + doc_number' 에 대한 OR 조건으로 펼쳐 내려보냅니다.
        //    Dexie(executeDexiePlan)가 alternates 를 읽어 재질의하므로,
        //    LanceDB 스코프를 좁히지 않고도 정밀 필터가 성립합니다.
        let mut alternates = serde_json::Map::new();
        for (k, v) in identity_alternates.iter() {
            alternates.insert(k.clone(), v.clone());
        }
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
            let identifier_axis = k.starts_with("reference_") || k == "doc_number" || k == "no";
            if !identifier_axis { continue; }
            if alternates.contains_key(k) { continue; }
            let val = v.get("value").and_then(|x| x.as_str()).unwrap_or("");
            if val.is_empty() { continue; }
            let mut axes: Vec<String> = Vec::new();
            for f in crate::logic::TRADE_REFERENCE_FIELDS.iter() {
                if *f == k.as_str() { continue; }
                axes.push(f.to_string());
            }
            for f in ["doc_number", "no"] {
                if f != k.as_str() && !axes.iter().any(|a| a == f) {
                    axes.push(f.to_string());
                }
            }
            alternates.insert(k.clone(), json!(axes));
        }

        let structured = !conditions.is_empty()
            || !hints.is_empty()
            || !hub_values.is_empty()
            || !final_scope.is_empty()
            || !projection.is_empty();
        let mut keywords: Vec<String> = Vec::new();
        for (k, v) in conditions.iter() {
            let fmt = crate::utils::ai_utils::query_value_format(k);
            if !matches!(
                fmt,
                crate::utils::ai_utils::FieldFormat::Identifier | crate::utils::ai_utils::FieldFormat::TrackingCode
            ) {
                continue;
            }
            if let Some(s) = v.get("value").and_then(|x| x.as_str()) {
                let t = s.trim();
                if !t.is_empty() && !keywords.iter().any(|x| x == t) {
                    keywords.push(t.to_string());
                }
            }
        }
        for h in hub_values.iter() {
            for w in h.split_whitespace() {
                if !keywords.iter().any(|x| x == w) {
                    keywords.push(w.to_string());
                }
            }
        }
        if !structured {
            for (i, w) in all_words.iter().enumerate() {
                if consumed_span.get(i).copied().unwrap_or(false) { continue; }
                if !content_flags.get(i).copied().unwrap_or(true) { continue; }
                if layers.roles.get(i) != Some(&ShipTokenRole::Content) { continue; }
                if !keywords.iter().any(|k| k == w) { keywords.push(w.clone()); }
            }
            if keywords.is_empty() {
                keywords = query.split_whitespace().map(|s| s.to_string()).collect();
            }
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
        if !hints.is_empty() {
            emit_term(&format!(
                "[STAGE-2 HINTS] 속성 힌트 {}개 (Dexie 필터 아님, 청크 property 타겟 검색 전용): {}",
                hints.len(),
                serde_json::to_string(&Value::Object(hints.clone())).unwrap_or_default()
            ));
        }
        if !projection.is_empty() {
            emit_term(&format!("[STAGE-2 PROJECTION] 함께 보여줄 연결 축: {:?}", projection));
        }
        emit_term(&format!("[STAGE-2] Trade document types in scope: {} 종", trade_types.len()));

        let payload = json!({ "task_id": task_id, "category": "Done", "summary": "Filter extraction complete.", "spinner": "✅" });
        let _ = app_handle.emit("extraction-progress", &payload);
        crate::utils::logger::log_task_progress(app_handle, task_id, &payload);

        let ctx = json!([{
            "type": "tracking",
            "types": trade_types,
            "text": target_text,
            "condition": Value::Object(conditions),
            "hint": Value::Object(hints),
            "projection": projection,
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

const SHIP_BIND_RADIUS: usize = 6;

fn ship_normalize_token(s: &str) -> String {
    crate::utils::ai_utils::lower_alnum(s)
}

fn ship_bank_centroid(bank: &[Vec<f32>]) -> Vec<f32> {
    let dim = match bank.iter().map(|e| e.len()).max() {
        Some(d) if d > 0 => d,
        _ => return Vec::new(),
    };
    let mut c = vec![0.0f32; dim];
    let mut cnt = 0usize;
    for e in bank.iter() {
        if e.len() != dim || e.iter().all(|&v| v == 0.0) { continue; }
        for (k, v) in e.iter().enumerate() { c[k] += v; }
        cnt += 1;
    }
    if cnt == 0 { return Vec::new(); }
    let norm = c.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for v in c.iter_mut() { *v /= norm; }
    }
    c
}

fn ship_currency_name_exact(core: &str) -> Option<&'static str> {
    crate::utils::ai_utils::currency_name_exact(core)
}

fn ship_currency_symbol(raw: &str) -> Option<&'static str> {
    crate::utils::ai_utils::currency_symbol_in(raw)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ShipTokenRole {
    Consumed,
    Function,
    Operator,
    Numeric,
    Identifier,
    Temporal,
    DocType,
    Unit,
    Content,
}

#[derive(Clone, Debug)]
pub struct ShipTemporal {
    pub start: String,
    pub end: String,
    pub operator: String,
    pub tokens: Vec<usize>,
    pub field: String,
}

#[derive(Clone, Debug)]
pub struct ShipDocMention {
    pub start: usize,
    pub end: usize,
    pub codes: Vec<String>,
    pub exact: bool,
}

#[derive(Clone, Debug, Default)]
pub struct ShipNumeric {
    pub token: usize,
    pub value: String,
    pub grouped: bool,
    pub operator: String,
    pub currency: String,
}

#[derive(Clone, Debug)]
pub struct ShipWinnerView {
    pub start: usize,
    pub end: usize,
    pub category: String,
    pub score: f32,
}

pub enum ShipAssign {
    Hard { field: String, operator: String, value: String, value_to: Option<String> },
    Soft { field: String, operator: String, value: String, value_to: Option<String> },
}

pub struct ShipQueryLayers {
    pub roles: Vec<ShipTokenRole>,
    pub variants: Vec<Vec<String>>,
    pub numerics: Vec<ShipNumeric>,
    pub identifiers: Vec<(usize, String)>,
    pub doc_mentions: Vec<ShipDocMention>,
    pub relation_marks: Vec<usize>,
    pub enum_hits: Vec<Vec<(String, String)>>,
    pub temporal: Option<ShipTemporal>,
    pub logs: Vec<String>,
}

#[derive(Clone, Debug)]
enum ShipTimePart {
    Year(i32),
    Month(u32),
    Day(u32),
    Iso(String),
    Rel(String),
}

struct ShipEmbTable {
    index: std::collections::HashMap<String, usize>,
    embs: Vec<Vec<f32>>,
    zero: Vec<f32>,
}

impl ShipEmbTable {
    fn get(&self, t: &str) -> &Vec<f32> {
        match self.index.get(t.trim()) {
            Some(&i) => self.embs.get(i).unwrap_or(&self.zero),
            None => &self.zero,
        }
    }

    fn bank(&self, phrases: &[String]) -> Vec<Vec<f32>> {
        phrases.iter().map(|p| self.get(p).clone()).collect()
    }
}

fn ship_add_text(t: &str, texts: &mut Vec<String>, seen: &mut std::collections::HashSet<String>) {
    let s = t.trim();
    if s.is_empty() { return; }
    if seen.insert(s.to_string()) { texts.push(s.to_string()); }
}

pub fn ship_edge_core(word: &str) -> String {
    word.trim_matches(|c: char| !c.is_alphanumeric()).to_string()
}

pub fn ship_identifier_shape(core: &str) -> bool {
    if core.chars().count() < 4 { return false; }
    if crate::utils::ai_utils::has_date_literal(core) { return false; }
    if !core
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '/' || c == '.')
    {
        return false;
    }
    let digits = core.chars().filter(|c| c.is_ascii_digit()).count();
    let letters = core.chars().filter(|c| c.is_ascii_alphabetic()).count();
    let seps = core.chars().filter(|c| !c.is_ascii_alphanumeric()).count();
    if digits == 0 { return false; }
    if letters >= 2 && digits >= 3 { return true; }
    seps > 0 && letters >= 1 && digits >= 2
}

pub fn ship_numeric_value(core: &str) -> (String, bool) {
    let mut out = String::new();
    let mut grouped = false;
    let mut started = false;
    for ch in core.chars() {
        if ch.is_ascii_digit() {
            out.push(ch);
            started = true;
        } else if ch == '.' && started && !out.contains('.') {
            out.push(ch);
        } else if ch == ',' && started {
            grouped = true;
        } else if started && !out.is_empty() { break; }
    }
    (out.trim_end_matches('.').to_string(), grouped)
}

pub fn ship_numeric_residue(core: &str) -> String {
    core.chars()
        .filter(|c| !(c.is_ascii_digit() || *c == '.' || *c == ','))
        .collect::<String>()
        .trim_matches(|c: char| !c.is_alphanumeric())
        .to_string()
}

pub fn ship_split_alpha_numeric(core: &str) -> Option<(String, String)> {
    let chars: Vec<char> = core.chars().collect();
    if chars.is_empty() { return None; }
    let is_num = |c: char| c.is_ascii_digit() || c == '.' || c == ',';
    let lead_alpha = chars[0].is_ascii_alphabetic();
    let split = chars
        .iter()
        .position(|&c| if lead_alpha { is_num(c) } else { !is_num(c) })?;
    let head: String = chars[..split].iter().collect();
    let tail: String = chars[split..].iter().collect();
    let (letters, number) = if lead_alpha { (head, tail) } else { (tail, head) };
    if letters.is_empty() || number.is_empty() { return None; }
    if !letters.chars().all(|c| c.is_ascii_alphabetic()) { return None; }
    if !number.chars().all(is_num) { return None; }
    if letters.chars().count() > 4 { return None; }
    Some((letters, number))
}

pub fn ship_is_long_code(n: &ShipNumeric) -> bool {
    !n.grouped && n.value.chars().filter(|c| c.is_ascii_digit()).count() >= 6
}

fn ship_fmt_range(a: chrono::NaiveDate, b: chrono::NaiveDate) -> (String, String) {
    crate::utils::time_guide::iso_bounds(a, b)
}

pub fn ship_day_range(y: i32, m: u32, d: u32) -> Option<(String, String)> {
    let day = chrono::NaiveDate::from_ymd_opt(y, m, d)?;
    Some(ship_fmt_range(day, day))
}

pub fn ship_month_range(y: i32, m: u32) -> Option<(String, String)> {
    let start = chrono::NaiveDate::from_ymd_opt(y, m, 1)?;
    let next = if m == 12 {
        chrono::NaiveDate::from_ymd_opt(y + 1, 1, 1)?
    } else {
        chrono::NaiveDate::from_ymd_opt(y, m + 1, 1)?
    };
    Some(ship_fmt_range(start, next.pred_opt()?))
}

pub fn ship_year_range(y: i32) -> Option<(String, String)> {
    let start = chrono::NaiveDate::from_ymd_opt(y, 1, 1)?;
    let end = chrono::NaiveDate::from_ymd_opt(y, 12, 31)?;
    Some(ship_fmt_range(start, end))
}

pub fn ship_iso_day_range(literal: &str) -> Option<(String, String)> {
    let groups: Vec<&str> = literal
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| !s.is_empty())
        .collect();
    if groups.len() < 3 { return None; }
    if groups[0].len() == 4 {
        let y: i32 = groups[0].parse().ok()?;
        let m: u32 = groups[1].parse().ok()?;
        let d: u32 = groups[2].parse().ok()?;
        return ship_day_range(y, m, d);
    }
    if groups[2].len() == 4 && groups[0].len() <= 2 && groups[1].len() <= 2 {
        let a: u32 = groups[0].parse().ok()?;
        let b: u32 = groups[1].parse().ok()?;
        let y: i32 = groups[2].parse().ok()?;
        if a > 12 && b <= 12 { return ship_day_range(y, b, a); }
        if b > 12 && a <= 12 { return ship_day_range(y, a, b); }
        return ship_day_range(y, b, a);
    }
    None
}

pub fn ship_numeric_date_literal(core: &str) -> Option<String> {
    let groups: Vec<&str> = core.split(|c: char| c == '-' || c == '/' || c == '.').collect();
    if groups.len() != 3 { return None; }
    if !groups.iter().all(|g| !g.is_empty() && g.len() <= 4 && g.chars().all(|c| c.is_ascii_digit())) {
        return None;
    }
    if groups[0].len() == 4 && groups[1].len() <= 2 && groups[2].len() <= 2 {
        return Some(core.to_string());
    }
    if groups[2].len() == 4 && groups[0].len() <= 2 && groups[1].len() <= 2 {
        return Some(core.to_string());
    }
    None
}

pub fn ship_year_month_literal(core: &str) -> Option<(i32, u32)> {
    let groups: Vec<&str> = core.split(|c: char| c == '-' || c == '/').collect();
    if groups.len() != 2 { return None; }
    if groups[0].len() != 4 || groups[1].is_empty() || groups[1].len() > 2 { return None; }
    if !groups.iter().all(|g| g.chars().all(|c| c.is_ascii_digit())) { return None; }
    let y: i32 = groups[0].parse().ok()?;
    let m: u32 = groups[1].parse().ok()?;
    if !(1..=12).contains(&m) || !(1900..=2100).contains(&y) { return None; }
    Some((y, m))
}

pub fn ship_relative_range(key: &str, today: chrono::NaiveDate) -> Option<(String, String)> {
    crate::utils::time_guide::relative_period(key, today).map(|(a, b)| ship_fmt_range(a, b))
}

fn ship_compose_range(parts: &[ShipTimePart], today: chrono::NaiveDate) -> Option<(String, String)> {
    use chrono::Datelike;
    let mut y: Option<i32> = None;
    let mut m: Option<u32> = None;
    let mut d: Option<u32> = None;
    for p in parts.iter() {
        match p {
            ShipTimePart::Iso(s) => return ship_iso_day_range(s),
            ShipTimePart::Rel(k) => return ship_relative_range(k, today),
            ShipTimePart::Year(v) => {
                if y.is_none() { y = Some(*v); }
            }
            ShipTimePart::Month(v) => {
                if m.is_none() { m = Some(*v); }
            }
            ShipTimePart::Day(v) => {
                if d.is_none() { d = Some(*v); }
            }
        }
    }
    let yy = y.unwrap_or(today.year());
    match (y.is_some(), m, d) {
        (_, Some(mm), Some(dd)) => ship_day_range(yy, mm, dd),
        (_, Some(mm), None) => ship_month_range(yy, mm),
        (true, None, None) => ship_year_range(yy),
        (true, None, Some(dd)) => {
            println!(
                "   🧯 [TEMPORAL REDUCE] 연 {} 과 일 {} 만 확정되고 월이 없습니다. 조각을 통째로 버리면 확정된 연도까지 함께 사라지므로, 확정 가능한 가장 넓은 구간인 연 범위로 축약합니다.",
                yy, dd
            );
            ship_year_range(yy)
        }
        _ => None,
    }
}

fn ship_op_key(
    q: &[f32],
    op_keys: &[String],
    op_bias: &[Vec<Vec<f32>>],
    op_prej: &[Vec<Vec<f32>>],
) -> Option<String> {
    let mut scored: Vec<(usize, f32)> = Vec::new();
    for ki in 0..op_keys.len() {
        let b = crate::utils::ai_utils::max_pool_sim(q, &op_bias[ki]);
        let p = crate::utils::ai_utils::max_pool_sim(q, &op_prej[ki]);
        scored.push((ki, b - (p - b).max(0.0)));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top = scored.first()?;
    let second = scored.get(1).map(|x| x.1).unwrap_or(f32::MIN);
    if top.1 > second {
        Some(op_keys[top.0].clone())
    } else {
        None
    }
}

fn ship_enum_resolve(
    q: &[f32],
    core: &str,
    codes: &[(String, Vec<Vec<f32>>)],
    fun: f32,
    opb: f32,
    lab: f32,
) -> Option<(String, f32, f32)> {
    let upper = core.to_uppercase();
    if let Some((code, _)) = codes.iter().find(|(c, _)| *c == upper) { return Some((code.clone(), 1.0, 1.0)); }
    if codes.len() < 2 { return None; }
    let mut sims: Vec<(usize, f32)> = codes
        .iter()
        .enumerate()
        .map(|(i, (_, bank))| (i, crate::utils::ai_utils::max_pool_sim(q, bank)))
        .collect();
    sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let n = sims.len() as f32;
    let mean = sims.iter().map(|x| x.1).sum::<f32>() / n;
    let sd = (sims.iter().map(|x| (x.1 - mean) * (x.1 - mean)).sum::<f32>() / n).sqrt();
    let top = sims[0];
    let gap = top.1 - sims[1].1;
    if gap > sd && top.1 > fun && top.1 > opb && top.1 >= lab {
        Some((codes[top.0].0.clone(), top.1, gap))
    } else {
        None
    }
}

pub fn ship_category_accepts(category: &str, want: crate::utils::ai_utils::FieldFormat) -> bool {
    use crate::utils::ai_utils::FieldFormat;
    crate::logic::trade_condition_fields(category).iter().any(|(f, _, _)| {
        let q = crate::utils::ai_utils::query_value_format(f);
        q == want || (want == FieldFormat::Identifier && q == FieldFormat::TrackingCode)
    })
}

/// 이 축이 질의가 지목한 서식의 저장 스키마에 존재하는가.
///
///  ── 왜 필요한가 ──
///   D2 는 카테고리의 전 필드로 뱅크를 세웁니다. reference 카테고리는 45종 서식의
///   참조 축을 모두 들고 있어 후보가 53개인데, CI 스키마에 실제로 존재하는 축은 그중 일부입니다.
///   존재하지 않는 축을 뱅크에 넣으면 ① 임베딩과 코사인을 헛돌리고
///   ② 그 축이 1위가 되면 저장될 수 없는 조건이 만들어져 결과가 확정적으로 0건이 됩니다.
///   (날짜 축은 DATE FIELD SCOPE 가 이미 같은 기준으로 좁히고 있습니다)
///
///  ── 모르면 통과 ──
///   trade_schema_owner_of 가 그 서식을 모른다고 답하면 판정 근거가 없는 것이므로
///   좁히지 않습니다. 스코프가 비어 있을 때도 전부 통과입니다.
pub fn ship_field_in_scope(field: &str, scope: &[String]) -> bool {
    if scope.is_empty() { return true; }
    let mut checked = 0usize;
    for code in scope.iter() {
        let (known, cat) = crate::model::merge::trade_schema_owner_of(&code.to_uppercase(), field);
        if !known { continue; }
        checked += 1;
        if !cat.is_empty() { return true; }
    }
    checked == 0
}

pub fn ship_field_category(field: &str) -> Option<String> {
    crate::logic::TRADE_CONDITION_CATEGORIES
        .iter()
        .find(|(c, _)| crate::logic::trade_condition_fields(c).iter().any(|(f, _, _)| *f == field))
        .map(|(c, _)| c.to_string())
}

pub const SHIP_OPERATOR_BIND_RADIUS: usize = 2;

/// 이 연산자 토큰이 실제로 비교할 대상을 가질 수 있는가.
///
///  ── 왜 필요한가 ──
///   OPERATOR SPLIT 은 '연산자 뱅크가 이 토큰을 라벨보다 잘 설명하는가' 만 봅니다.
///   그런데 비교 연산자는 비교 대상이 있어야 연산자입니다. 대상이 없는데 연산자로 남으면
///   아래 numerics_raw 의 연산자 탐색이 거리 2 안에서 그 토큰을 주워,
///   전혀 다른 수치에 엉뚱한 연산자를 붙입니다.
///
///  ── 경로 규칙 ──
///   수치 결속 탐색(ship_bind_values / numerics 연산자 탐색)과 같은 규칙을 씁니다.
///   사이에 Content 와 Unit 만 있을 때 '경로가 열려 있다' 고 봅니다.
///   기능어·다른 연산자·서식명이 끼면 문장 경계로 보고 막습니다.
pub fn ship_operator_bindable(roles: &[ShipTokenRole], i: usize) -> bool {
    let n = roles.len();
    for dist in 1..=SHIP_OPERATOR_BIND_RADIUS {
        for j in [i + dist, i.wrapping_sub(dist)] {
            if j >= n { continue; }
            let (lo, hi) = if j > i { (i + 1, j) } else { (j + 1, i) };
            let clear = (lo..hi).all(|k| {
                matches!(roles[k], ShipTokenRole::Content | ShipTokenRole::Unit)
            });
            if !clear { continue; }
            if matches!(
                roles[j],
                ShipTokenRole::Numeric | ShipTokenRole::Temporal | ShipTokenRole::Identifier
            ) {
                return true;
            }
        }
    }
    false
}

pub fn ship_value_near(roles: &[ShipTokenRole], start: usize, end: usize, radius: usize) -> bool {
    let is_val = |k: usize| {
        matches!(
            roles.get(k),
            Some(ShipTokenRole::Numeric) | Some(ShipTokenRole::Identifier) | Some(ShipTokenRole::Unit)
        )
    };
    let mut k = start;
    let mut steps = 0usize;
    while k > 0 && steps < radius {
        k -= 1;
        steps += 1;
        if roles.get(k) == Some(&ShipTokenRole::Function) { break; }
        if is_val(k) { return true; }
    }
    let mut k = end;
    let mut steps = 0usize;
    while k < roles.len() && steps < radius {
        if roles.get(k) == Some(&ShipTokenRole::Function) { break; }
        if is_val(k) { return true; }
        k += 1;
        steps += 1;
    }
    false
}

pub fn ship_bind_values(
    roles: &[ShipTokenRole],
    winners: &[ShipWinnerView],
    token: usize,
    accepts: &dyn Fn(&str) -> bool,
) -> Option<usize> {
    let mut best: Option<(usize, usize, bool)> = None;
    for (wi, w) in winners.iter().enumerate() {
        if !accepts(&w.category) { continue; }
        let (dist, left, lo, hi) = if token >= w.end {
            (token + 1 - w.end, true, w.end, token)
        } else if token < w.start {
            (w.start - token, false, token + 1, w.start)
        } else {
            (0usize, true, token, token)
        };
        if dist > SHIP_BIND_RADIUS { continue; }
        if (lo..hi).any(|k| roles.get(k) == Some(&ShipTokenRole::Function)) { continue; }
        let better = match best {
            None => true,
            Some((_, bd, bl)) => dist < bd || (dist == bd && left && !bl),
        };
        if better { best = Some((wi, dist, left)); }
    }
    best.map(|(wi, _, _)| wi)
}

pub fn ship_mention_by_mark(m: &ShipDocMention, relation_marks: &[usize]) -> bool {
    relation_marks.iter().any(|&k| {
        (k < m.start && m.start - k <= 2) || (k >= m.end && k - m.end <= 1)
    })
}

pub fn ship_resolve_doc_scope(
    mentions: &[ShipDocMention],
    winners: &[ShipWinnerView],
    relation_marks: &[usize],
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let mut scope: Vec<String> = Vec::new();
    let mut projection: Vec<String> = Vec::new();
    let mut logs: Vec<String> = Vec::new();

    fn relational(
        m: &ShipDocMention,
        winners: &[ShipWinnerView],
        relation_marks: &[usize],
    ) -> (bool, bool) {
        let by_winner = winners.iter().any(|w| {
            (w.category == "reference" || w.category == "hub")
                && (w.end == m.start || w.start == m.end)
        });
        (by_winner, ship_mention_by_mark(m, relation_marks))
    }
    fn absorb(m: &ShipDocMention, scope: &mut Vec<String>, logs: &mut Vec<String>) {
        if m.codes.iter().any(|c| scope.contains(c)) {
            logs.push(format!(
                "   📄 [DOC SCOPE] {:?} 는 이미 확정된 범위 {:?} 와 같은 계열이라 범위를 넓히지 않습니다.",
                m.codes, scope
            ));
            return;
        }
        for c in m.codes.iter() {
            if !scope.contains(c) { scope.push(c.clone()); }
        }
        logs.push(format!("   📄 [DOC SCOPE] 조회 대상 서식 {:?} 추가 → 범위 {:?}", m.codes, scope));
    }

    let mut deferred: Vec<&ShipDocMention> = Vec::new();
    for m in mentions.iter() {
        let (by_winner, by_mark) = relational(m, winners, relation_marks);
        if by_mark && !by_winner {
            logs.push(format!(
                "   🔗 [RELATION ADJACENT] {:?} 옆에 관계 표지가 있어 이 서식 언급을 조회 범위가 아니라 연결 축으로 읽습니다. D1 카테고리가 이 스팬을 reference·hub 로 판정하지 못해도 성립합니다.",
                m.codes
            ));
        }
        if by_winner || by_mark {
            deferred.push(m);
        } else {
            absorb(m, &mut scope, &mut logs);
        }
    }
    for m in deferred.into_iter() {
        let disjoint = !m.codes.iter().any(|c| scope.contains(c));
        if !scope.is_empty() && disjoint {
            if let Some(f) = m.codes.iter().find_map(|c| crate::logic::trade_reference_field_of(c)) {
                if !projection.iter().any(|p| p == f) { projection.push(f.to_string()); }
                logs.push(format!(
                    "   🧭 [DOC PROJECTION] {:?} 는 참조·연결 스팬에 붙은 다른 서식입니다. 필터가 아니라 함께 보여줄 축 '{}' 로 보냅니다.",
                    m.codes, f
                ));
                continue;
            }
        }
        absorb(m, &mut scope, &mut logs);
    }
    (scope, projection, logs)
}

pub fn ship_is_monetary(field: &str) -> bool {
    ["amount", "price", "charge", "value", "debit", "credit", "balance", "premium", "fee"]
        .iter()
        .any(|k| field.contains(k))
}

pub fn ship_companion_currency(field: &str, nums: &[ShipNumeric]) -> Option<String> {
    if !ship_is_monetary(field) { return None; }
    nums.iter()
        .find(|n| !n.currency.is_empty())
        .map(|n| n.currency.clone())
}

pub fn ship_make_assignment(
    field: &str,
    category: &str,
    span_text: &str,
    nums: &[ShipNumeric],
    ids: &[String],
    enum_code: Option<&str>,
    d1_significant: bool,
) -> Option<ShipAssign> {
    use crate::utils::ai_utils::FieldFormat;
    let array_field = crate::logic::is_trade_array_category(category);
    let fmt = crate::utils::ai_utils::query_value_format(field);
    let (operator, value, value_to): (String, String, Option<String>) = match fmt {
        FieldFormat::Identifier | FieldFormat::TrackingCode => {
            let mut v = String::new();
            for id in ids.iter() {
                let r = trade_resolve_condition_value(field, id);
                if !r.is_empty() {
                    v = r;
                    break;
                }
            }
            if v.is_empty() {
                if let Some(n) = nums.iter().find(|n| ship_is_long_code(n)) { v = n.value.clone(); }
            }
            if v.is_empty() { return None; }
            (crate::logic::trade_default_operator(field).to_string(), v, None)
        }
        FieldFormat::Numeric => {
            if nums.is_empty() { return None; }
            let lower = nums.iter().find(|n| n.operator == "gte" || n.operator == "gt");
            let upper = nums.iter().find(|n| n.operator == "lte" || n.operator == "lt");
            match (lower, upper) {
                (Some(lo), Some(hi)) if lo.token != hi.token => {
                    ("between".to_string(), lo.value.clone(), Some(hi.value.clone()))
                }
                _ => {
                    let monetary = ship_is_monetary(field);
                    let eligible: Vec<&ShipNumeric> = nums
                        .iter()
                        .filter(|n| if monetary { true } else { n.currency.is_empty() && !n.grouped })
                        .collect();
                    let pool: Vec<&ShipNumeric> =
                        if eligible.is_empty() { nums.iter().collect() } else { eligible };
                    let rank = |n: &ShipNumeric| -> (u8, u8, u8, usize) {
                        (
                            if n.operator.is_empty() { 0 } else { 1 },
                            if monetary && !n.currency.is_empty() { 1 } else { 0 },
                            if monetary && n.grouped { 1 } else { 0 },
                            n.value.chars().filter(|c| c.is_ascii_digit()).count(),
                        )
                    };
                    let n = pool.iter().copied().max_by_key(|n| rank(n))?;
                    let op = if n.operator.is_empty() {
                        trade_resolve_condition_operator(field, span_text)
                    } else {
                        n.operator.clone()
                    };
                    (op, n.value.clone(), None)
                }
            }
        }
        FieldFormat::Enum => {
            let code = match enum_code {
                Some(c) if !c.trim().is_empty() => c.trim().to_string(),
                _ => return None,
            };
            ("contains".to_string(), code, None)
        }
        FieldFormat::Text | FieldFormat::Address => {
            if !d1_significant { return None; }
            let v = span_text.split_whitespace().collect::<Vec<_>>().join(" ");
            if v.is_empty() { return None; }
            return Some(ShipAssign::Soft {
                field: field.to_string(),
                operator: "contains".to_string(),
                value: v,
                value_to: None,
            });
        }
        _ => return None,
    };
    if array_field {
        Some(ShipAssign::Soft { field: field.to_string(), operator, value, value_to })
    } else {
        Some(ShipAssign::Hard { field: field.to_string(), operator, value, value_to })
    }
}

pub fn ship_apply_assignment(
    assign: ShipAssign,
    conditions: &mut serde_json::Map<String, Value>,
    hints: &mut serde_json::Map<String, Value>,
    claimed: &mut std::collections::HashSet<String>,
) -> (&'static str, String, String) {
    let (kind, field, operator, value, value_to) = match assign {
        ShipAssign::Hard { field, operator, value, value_to } => ("HARD", field, operator, value, value_to),
        ShipAssign::Soft { field, operator, value, value_to } => ("HINT", field, operator, value, value_to),
    };
    let mut obj = json!({ "operator": operator.clone(), "value": value.clone() });
    if let Some(v2) = value_to.as_ref() { obj["value_to"] = json!(v2); }
    if kind == "HARD" {
        conditions.insert(field.clone(), obj);
    } else {
        hints.insert(field.clone(), obj);
    }
    claimed.insert(field);
    let shown = match value_to {
        Some(v2) => format!("{} ~ {}", value, v2),
        None => value,
    };
    (kind, operator, shown)
}

pub fn ship_sds_demotion_reason(field: &str, scope: &[String]) -> Option<String> {
    if field == "doc_number" || field == "no" || field.starts_with("reference_") || field == "hub_reference" {
        return None;
    }
    if crate::logic::is_trade_array_category(crate::logic::trade_field_category(field)) {
        return None;
    }
    if !scope.is_empty() {
        let mut checked = 0usize;
        let mut owned = 0usize;
        for code in scope.iter() {
            let up = code.to_uppercase();
            let (known, cat) = crate::model::merge::trade_schema_owner_of(&up, field);
            if !known { continue; }
            checked += 1;
            if !cat.is_empty() { owned += 1; }
        }
        if checked > 0 && owned == 0 {
            return Some(format!(
                "질의가 지목한 서식 {:?} 의 저장 스키마(base + overlay)에 '{}' 축이 존재하지 않습니다. 저장될 수 없는 축을 하드 조건으로 두면 회수 문서 전부가 비교할 값 없이 탈락해 결과가 확정적으로 0건이 됩니다. 조건을 버리지 않고 힌트로 내려 청크 검색에만 씁니다",
                scope, field
            ));
        }
    }
    if let Some(rate) = crate::utils::score_dynamics::search_kill_rate(field) {
        let ceiling = 1.0 - 1.0 / crate::utils::score_dynamics::Track::Search.ring_len() as f32;
        if rate >= ceiling {
            return Some(format!(
                "이 스코프에서 이 필드 하드 조건이 리콜 문서를 전부 걸러낸 비율 {:.0}% ≥ {:.0}%",
                rate * 100.0, ceiling * 100.0
            ));
        }
    }
    if !scope.is_empty() {
        if let Some((rate, assigned, docs)) = crate::utils::score_dynamics::storage_fill_prior(scope, field) {
            if assigned == 0 {
                return Some(format!(
                    "저장 측 {:?} 문서 {}건 중 이 필드가 채워진 문서 0건 (평활 채움률 {:.3})",
                    scope, docs, rate
                ));
            }
        }
    }
    None
}

impl crate::model::LogisModel {
    /// 여러 필드의 구 묶음을 **한 번의 배치 임베딩**으로 만듭니다.
    ///
    ///  ── 무엇이 문제였나 ──
    ///   D2 뱅크 구축이 필드마다 get_embedding_batch 를 따로 불렀습니다.
    ///   실측 로그에 '요청 2건 | 실연산 1건' 같은 마이크로 배치가 60회 이상 찍힙니다.
    ///   호출마다 락 획득·캐시 조회·텐서 할당이 붙으므로, 구 수가 아니라 호출 수가 비용입니다.
    ///   다국어 뱅크로 구가 12배가 되면 이 구조가 그대로 12배 느려집니다.
    ///
    ///  ── 중복 접기 ──
    ///   필드 간에 같은 구(별칭·보강 라벨)가 겹치므로 유일 구만 실연산합니다.
    pub async fn ship_embed_phrase_groups(&self, groups: &[Vec<String>]) -> Vec<Vec<Vec<f32>>> {
        let mut uniq: Vec<String> = Vec::new();
        let mut index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for g in groups.iter() {
            for p in g.iter() {
                if index.contains_key(p) { continue; }
                index.insert(p.clone(), uniq.len());
                uniq.push(p.clone());
            }
        }
        let mut embs: Vec<Vec<f32>> = Vec::with_capacity(uniq.len());
        for part in uniq.chunks(200) {
            let e = self
                .get_embedding_batch(part.to_vec())
                .await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; part.len()]);
            embs.extend(e);
        }
        let zero = vec![0.0f32; 384];
        groups
            .iter()
            .map(|g| {
                g.iter()
                    .map(|p| match index.get(p) {
                        Some(&i) => embs.get(i).cloned().unwrap_or_else(|| zero.clone()),
                        None => zero.clone(),
                    })
                    .collect()
            })
            .collect()
    }

    pub async fn ship_label_span_value(
        &self,
        field: &str,
        start: usize,
        end: usize,
        words: &[String],
        variants: &[Vec<String>],
        roles: &[ShipTokenRole],
        consumed: &[bool],
        label_bank: &Vec<Vec<f32>>,
        label_wt: &Vec<f32>,
    ) -> Option<(f32, f32, Option<(usize, String, f32, f32)>)> {
        use crate::utils::ai_utils::{max_pool_sim, weighted_max_pool_sim};
        if start >= end || end > words.len() || label_bank.is_empty() { return None; }
        let value_phr = crate::utils::ai_utils::multilingual_value_anchor_phrases_scoped("shipping_doc", field);
        if value_phr.is_empty() { return None; }
        let open = |k: usize| -> bool {
            roles.get(k) == Some(&ShipTokenRole::Content) && !consumed.get(k).copied().unwrap_or(true)
        };
        let mut cand: Vec<usize> = Vec::new();
        let mut k = start;
        while k > 0 && start - k < SHIP_OPERATOR_BIND_RADIUS {
            k -= 1;
            if !open(k) { break; }
            cand.push(k);
        }
        let mut k = end;
        while k < words.len() && k - end < SHIP_OPERATOR_BIND_RADIUS {
            if !open(k) { break; }
            cand.push(k);
            k += 1;
        }
        let mut forms: Vec<(usize, String)> = Vec::new();
        for &c in cand.iter() {
            let base = ship_edge_core(&words[c]);
            if !base.is_empty() { forms.push((c, base)); }
            if let Some(vs) = variants.get(c) {
                for v in vs.iter() {
                    let v = v.trim().to_string();
                    if v.is_empty() || forms.iter().any(|(fc, f)| *fc == c && *f == v) { continue; }
                    forms.push((c, v));
                }
            }
        }
        let mut texts: Vec<String> = vec![words[start..end].join(" ")];
        texts.extend(forms.iter().map(|(_, f)| f.clone()));
        let embs = self.get_embedding_batch(texts).await.ok()?;
        if embs.len() != forms.len() + 1 || embs[0].iter().all(|&v| v == 0.0) { return None; }
        let value_bank = self.ship_embed_phrase_groups(&[value_phr]).await.into_iter().next()?;
        if value_bank.is_empty() { return None; }
        let span_val = max_pool_sim(&embs[0], &value_bank);
        let span_lab = weighted_max_pool_sim(&embs[0], label_bank, label_wt);
        if span_val >= span_lab { return None; }
        let mut best: Option<(usize, String, f32, f32)> = None;
        for (fi, (c, f)) in forms.iter().enumerate() {
            let e = &embs[fi + 1];
            if e.iter().all(|&v| v == 0.0) { continue; }
            let val = max_pool_sim(e, &value_bank);
            let lab = weighted_max_pool_sim(e, label_bank, label_wt);
            if val <= lab || val <= span_val { continue; }
            if best.as_ref().map_or(true, |b| val > b.2) {
                best = Some((*c, f.clone(), val, lab));
            }
        }
        Some((span_val, span_lab, best))
    }

    pub async fn build_shipping_query_layers(
        &self,
        words: &[String],
        lemmas: &[String],
        consumed_words: &std::collections::HashSet<String>,
    ) -> ShipQueryLayers {
        use crate::utils::ai_utils::{cosine_similarity, gumbel_expected_z, max_pool_sim, split_bias_phrases_full};

        let n = words.len();
        let mut logs: Vec<String> = Vec::new();
        let mut roles: Vec<ShipTokenRole> = vec![ShipTokenRole::Content; n];
        let cores: Vec<String> = words
            .iter()
            .map(|w| crate::utils::ai_utils::normalize_digits_ascii(&ship_edge_core(w)))
            .collect();

        for i in 0..n {
            if consumed_words.contains(&words[i]) {
                roles[i] = ShipTokenRole::Consumed;
                continue;
            }
            if cores[i].is_empty() {
                roles[i] = ShipTokenRole::Function;
                continue;
            }
            if cores[i].chars().any(|c| c.is_ascii_digit()) {
                roles[i] = if ship_identifier_shape(&cores[i]) {
                    ShipTokenRole::Identifier
                } else {
                    ShipTokenRole::Numeric
                };
            }
        }
        let pending: Vec<usize> = (0..n).filter(|&i| roles[i] == ShipTokenRole::Content).collect();

        let label_phr: Vec<String> = crate::logic::trade_condition_all_phrases();
        logs.push(format!(
            "   🏷️ [LABEL BANK / ML] 조건 카테고리 {}개의 12개 언어 라벨 구 {}개로 라벨 축을 세웁니다. 영어 한 벌만 두면 '단가가'·'금액이'·'번호도' 같은 비영어 라벨어가 라벨 축에서 설명되지 못해, 기능어·연산자 축이 그 토큰을 대신 가져갑니다.",
            crate::logic::TRADE_CONDITION_CATEGORIES.len(),
            label_phr.len()
        ));
        let mut func_phr: Vec<String> = Vec::new();
        for node in ["verb", "expression"] {
            if let Some(obj) = crate::parsing::BIAS_DICT
                .get(node)
                .and_then(|v| v.get("bias"))
                .and_then(|v| v.as_object())
            {
                for (_, v) in obj.iter() {
                    if let Some(s) = v.as_str() {
                        for p in split_bias_phrases_full(s) {
                            if !func_phr.contains(&p) { func_phr.push(p); }
                        }
                    }
                }
            }
        }
        if let Some(s) = crate::parsing::BIAS_DICT
            .get("ignore")
            .and_then(|v| v.get("bias"))
            .and_then(|v| v.as_str())
        {
            for p in split_bias_phrases_full(s) {
                if !func_phr.contains(&p) { func_phr.push(p); }
            }
        }
        {
            let mut added = 0usize;
            for p in ship_function_pivot_phrases().into_iter() {
                if func_phr.iter().any(|e| e.eq_ignore_ascii_case(&p)) { continue; }
                func_phr.push(p);
                added += 1;
            }
            logs.push(format!(
                "   🗣️ [FUNCTION BANK / ML] bias.json 의 verb·expression·ignore 구 {}개에 12개 언어 요청 동사·담화 표지 {}개를 더했습니다. 요청 동사와 담화 표지는 어느 언어에서나 닫힌 집합인데, 기존 뱅크는 사실상 영어뿐이라 '보여줘'·'알려주고'·'묶어서'·'같이'·'것만' 이 기능어 축에서 설명되지 못하고 라벨·연산자 축으로 흘러갔습니다.",
                func_phr.len() - added, added
            ));
        }
        let relation_phr = ship_relation_pivot_phrases();

        let mut op_keys: Vec<String> = Vec::new();
        let mut op_bias_phr: Vec<Vec<String>> = Vec::new();
        let mut op_prej_phr: Vec<Vec<String>> = Vec::new();
        if let Some(ops) = crate::parsing::BIAS_DICT.get("operators").and_then(|v| v.as_object()) {
            for (k, node) in ops.iter() {
                if k == "top" || k == "bottom" { continue; }
                let mut b: Vec<String> = Vec::new();
                for f in ["semantic", "bias"] {
                    if let Some(s) = node.get(f).and_then(|v| v.as_str()) {
                        for p in split_bias_phrases_full(s) {
                            if !b.contains(&p) { b.push(p); }
                        }
                    }
                }
                for p in crate::utils::ai_utils::temporal_operator_phrases(k) {
                    if !b.iter().any(|e| e.eq_ignore_ascii_case(&p)) { b.push(p); }
                }
                let mut p: Vec<String> = Vec::new();
                if let Some(s) = node.get("prejudice").and_then(|v| v.as_str()) {
                    for x in split_bias_phrases_full(s) {
                        if !p.contains(&x) { p.push(x); }
                    }
                }
                op_keys.push(k.clone());
                op_bias_phr.push(b);
                op_prej_phr.push(p);
            }
        }

        let mut titles: Vec<(String, Vec<String>)> = Vec::new();
        for (code, title) in crate::utils::ai_utils::all_trade_doc_titles().into_iter() {
            match titles.iter_mut().find(|(t, _)| t.eq_ignore_ascii_case(&title)) {
                Some((_, codes)) => {
                    if !codes.iter().any(|c| *c == code) { codes.push(code); }
                }
                None => titles.push((title, vec![code])),
            }
        }

        let time_phr = crate::utils::ai_utils::filter_category_phrases(&["time_filters"]);
        let mut time_keys: Vec<String> = Vec::new();
        for (_, k, _) in time_phr.iter() {
            if !time_keys.contains(k) { time_keys.push(k.clone()); }
        }

        let mut date_fields: Vec<(String, Vec<String>)> = Vec::new();
        let mut nondate_phr: Vec<String> = Vec::new();
        for (cat, _) in crate::logic::TRADE_CONDITION_CATEGORIES.iter() {
            for (fname, _, anchor) in crate::logic::trade_condition_fields(cat).iter() {
                let mut phrases: Vec<String> = split_bias_phrases_full(anchor)
                    .into_iter()
                    .filter(|p| !crate::utils::ai_utils::is_value_example_phrase(p))
                    .collect();
                let (ml, _) = crate::utils::ai_utils::label_phrase_bank_multilingual("en", "shipping_doc", fname);
                for p in ml.into_iter() {
                    if !phrases.contains(&p) { phrases.push(p); }
                }
                if crate::utils::ai_utils::query_value_format(fname) == crate::utils::ai_utils::FieldFormat::Date {
                    if !date_fields.iter().any(|(f, _)| f.as_str() == *fname) {
                        date_fields.push((fname.to_string(), phrases));
                    }
                } else {
                    for p in phrases {
                        if !nondate_phr.contains(&p) { nondate_phr.push(p); }
                    }
                }
            }
        }

        let mut enum_tables: Vec<(String, Vec<(String, Vec<String>)>, Vec<String>)> = Vec::new();
        for (field, codes) in crate::logic::TRADE_ENUM_VALUE_ANCHORS.iter() {
            let own_cat: &str = crate::logic::TRADE_CONDITION_CATEGORIES
                .iter()
                .find(|(c, _)| {
                    crate::logic::trade_condition_fields(c)
                        .iter()
                        .any(|(f, _, _)| *f == *field)
                })
                .map(|(c, _)| *c)
                .unwrap_or("");
            let mut foreign: Vec<String> = Vec::new();
            for (c, raw) in crate::logic::TRADE_CONDITION_CATEGORIES.iter() {
                if *c == own_cat { continue; }
                for p in split_bias_phrases_full(raw) {
                    if !foreign.contains(&p) { foreign.push(p); }
                }
            }
            let mut rows: Vec<(String, Vec<String>)> = Vec::new();
            for (code, anchor) in codes.iter() { rows.push((code.to_string(), split_bias_phrases_full(anchor))); }
            enum_tables.push((field.to_string(), rows, foreign));
        }

        let mut raw_variants: Vec<Vec<String>> = vec![Vec::new(); n];
        for &i in pending.iter() {
            let lemma = lemmas.get(i).map(|s| s.as_str()).unwrap_or("");
            raw_variants[i] = crate::analytic::morphological_variants(&words[i], lemma);
        }

        let mut texts: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for i in 0..n {
            if roles[i] == ShipTokenRole::Consumed || cores[i].is_empty() { continue; }
            ship_add_text(&cores[i], &mut texts, &mut seen);
            if roles[i] == ShipTokenRole::Numeric {
                ship_add_text(&ship_numeric_residue(&cores[i]), &mut texts, &mut seen);
            }
            if roles[i] == ShipTokenRole::Identifier {
                if let Some((letters, _)) = ship_split_alpha_numeric(&cores[i]) {
                    ship_add_text(&letters, &mut texts, &mut seen);
                }
            }
        }
        for width in 2..=3usize {
            for s in 0..n {
                let e = s + width;
                if e > n { break; }
                if (s..e).all(|k| roles[k] == ShipTokenRole::Content) {
                    ship_add_text(&cores[s..e].join(" "), &mut texts, &mut seen);
                }
            }
        }
        for vs in raw_variants.iter() {
            for v in vs.iter() { ship_add_text(v, &mut texts, &mut seen); }
        }
        for p in label_phr
            .iter()
            .chain(func_phr.iter())
            .chain(nondate_phr.iter())
            .chain(relation_phr.iter())
        {
            ship_add_text(p, &mut texts, &mut seen);
        }
        for bank in op_bias_phr.iter().chain(op_prej_phr.iter()) {
            for p in bank.iter() { ship_add_text(p, &mut texts, &mut seen); }
        }
        for (t, _) in titles.iter() { ship_add_text(t, &mut texts, &mut seen); }
        for (_, raw) in crate::utils::ai_utils::TIME_UNIT_PIVOTS_ML.iter() {
            for p in split_bias_phrases_full(raw) { ship_add_text(&p, &mut texts, &mut seen); }
        }
        for raw in crate::utils::ai_utils::MONTH_NAMES_ML.iter() {
            for p in split_bias_phrases_full(raw) { ship_add_text(&p, &mut texts, &mut seen); }
        }
        for (_, _, p) in time_phr.iter() { ship_add_text(p, &mut texts, &mut seen); }
        for (_, rows, _) in enum_tables.iter() {
            for (_, phrases) in rows.iter() {
                for p in phrases.iter() { ship_add_text(p, &mut texts, &mut seen); }
            }
        }
        for (_, phrases) in date_fields.iter() {
            for p in phrases.iter() { ship_add_text(p, &mut texts, &mut seen); }
        }

        let mut embs: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        for part in texts.chunks(200) {
            let e = self
                .get_embedding_batch(part.to_vec())
                .await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; part.len()]);
            embs.extend(e);
        }
        let dim = embs.iter().map(|v| v.len()).max().unwrap_or(384).max(1);
        let table = ShipEmbTable {
            index: texts.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect(),
            embs,
            zero: vec![0.0; dim],
        };

        let label_bank = table.bank(&label_phr);
        let func_bank = table.bank(&func_phr);
        let nondate_bank = table.bank(&nondate_phr);
        let op_bias_banks: Vec<Vec<Vec<f32>>> = op_bias_phr.iter().map(|b| table.bank(b)).collect();
        let op_prej_banks: Vec<Vec<Vec<f32>>> = op_prej_phr.iter().map(|b| table.bank(b)).collect();
        let op_all_bank: Vec<Vec<f32>> = op_bias_banks.iter().flatten().cloned().collect();
        let title_embs: Vec<Vec<f32>> = titles.iter().map(|(t, _)| table.get(t).clone()).collect();
        let unit_banks: Vec<(String, Vec<Vec<f32>>)> = {
            let raw_units: Vec<(String, Vec<String>)> = crate::utils::ai_utils::TIME_UNIT_PIVOTS_ML
                .iter()
                .map(|(k, raw)| (k.to_string(), split_bias_phrases_full(raw)))
                .collect();
            let full_banks: Vec<Vec<Vec<f32>>> = raw_units
                .iter()
                .map(|(_, ph)| ph.iter().map(|p| table.get(p).clone()).collect())
                .collect();
            let heads: Vec<Vec<f32>> = full_banks.iter().map(|b| ship_bank_centroid(b)).collect();
            let cohs: Vec<f32> = full_banks
                .iter()
                .map(|b| crate::utils::ai_utils::bank_internal_cohesion(b))
                .collect();
            let mut out: Vec<(String, Vec<Vec<f32>>)> = Vec::with_capacity(raw_units.len());
            let mut dropped: Vec<String> = Vec::new();
            let mut spared: Vec<String> = Vec::new();
            for (ui, (key, phrases)) in raw_units.iter().enumerate() {
                let mut bank: Vec<Vec<f32>> = Vec::new();
                for (pi, p) in phrases.iter().enumerate() {
                    let e = table.get(p);
                    if e.iter().all(|&v| v == 0.0) { continue; }
                    if pi > 0 && !heads[ui].iter().all(|&v| v == 0.0) {
                        let own = cosine_similarity(e, &heads[ui]);
                        let (rk, rs) = raw_units
                            .iter()
                            .enumerate()
                            .filter(|(k, _)| *k != ui)
                            .map(|(k, (name, _))| (name.clone(), cosine_similarity(e, &heads[k])))
                            .fold((String::new(), f32::MIN), |acc, x| if x.1 > acc.1 { x } else { acc });
                        if crate::utils::ai_utils::prejudice_dominates(own, rs, cohs[ui]) {
                            dropped.push(format!(
                                "{}←\"{}\" (자기 '{}' {:.4} × (1+{:.3}) < '{}' {:.4})",
                                key, p, key, own, cohs[ui].clamp(0.0, 0.5), rk, rs
                            ));
                            continue;
                        }
                        if rs > own {
                            spared.push(format!(
                                "{}←\"{}\" (자기 '{}' {:.4} vs '{}' {:.4}, 차이 {:.4})",
                                key, p, key, own, rk, rs, rs - own
                            ));
                        }
                    }
                    bank.push(e.clone());
                }
                if bank.is_empty() {
                    bank = full_banks[ui].clone();
                }
                out.push((key.clone(), bank));
            }
            if !dropped.is_empty() {
                logs.push(format!(
                    "   🧹 [TIME UNIT SELF-POISON] 자기 단위 뱅크 중심보다 다른 단위 뱅크 중심이 응집도 여유까지 넘어서 설명하는 앵커 구 {}개를 그 단위 뱅크에서 끕니다: {:?} — 대표를 첫 구(영어) 하나로 두면 다른 언어의 구가 교차언어 거리 때문에 잘리므로, 12개 언어 구 전체의 중심으로 판정합니다.",
                    dropped.len(), dropped
                ));
            }
            if !spared.is_empty() {
                logs.push(format!(
                    "   🛟 [TIME UNIT SELF-POISON SPARED] 다른 단위 중심이 근소하게 앞서지만 응집도 여유를 넘지 못해 살려 둔 구 {}개: {:?} — 절대 비교(경쟁 > 자기)로 자르면 0.0004 차이로도 그 언어의 연·월·일 구가 통째로 사라집니다. 뱅크 내부 응집도만큼은 교차언어 잡음으로 보고 허용합니다.",
                    spared.len(), spared
                ));
            }
            out
        };
        let month_banks: Vec<Vec<Vec<f32>>> = crate::utils::ai_utils::MONTH_NAMES_ML
            .iter()
            .map(|raw| table.bank(&split_bias_phrases_full(raw)))
            .collect();
        let time_banks: Vec<(String, Vec<Vec<f32>>)> = time_keys
            .iter()
            .map(|k| {
                let phrases: Vec<String> = time_phr
                    .iter()
                    .filter(|(_, kk, _)| kk == k)
                    .map(|(_, _, p)| p.clone())
                    .collect();
                (k.clone(), table.bank(&phrases))
            })
            .collect();
        let date_banks: Vec<(String, Vec<Vec<f32>>)> = date_fields
            .iter()
            .map(|(f, phrases)| (f.clone(), table.bank(phrases)))
            .collect();
        let enum_banks: Vec<(String, Vec<(String, Vec<Vec<f32>>)>, Vec<Vec<f32>>)> = enum_tables
            .iter()
            .map(|(f, rows, foreign)| {
                (
                    f.clone(),
                    rows.iter().map(|(c, phrases)| (c.clone(), table.bank(phrases))).collect(),
                    table.bank(foreign),
                )
            })
            .collect();

        let mut doc_mentions: Vec<ShipDocMention> = Vec::new();
        for &i in pending.iter() {
            let core = &cores[i];
            let codes = match crate::utils::ai_utils::trade_code_mention(core) {
                Some(c) => c,
                None => continue,
            };
            let next_is_value = matches!(
                roles.get(i + 1),
                Some(ShipTokenRole::Numeric) | Some(ShipTokenRole::Identifier)
            );
            if next_is_value { continue; }
            roles[i] = ShipTokenRole::DocType;
            logs.push(format!(
                "   📄 [DOC TYPE / CODE] \"{}\" → {:?} (서식 코드 완전일치)",
                core, codes
            ));
            doc_mentions.push(ShipDocMention { start: i, end: i + 1, codes, exact: true });
        }

        {
            let max_w = crate::utils::ai_utils::trade_title_max_words();
            let mut hits: Vec<String> = Vec::new();
            let mut s = 0usize;
            while s < n {
                if roles[s] != ShipTokenRole::Content {
                    s += 1;
                    continue;
                }
                let mut found: Option<(usize, Vec<String>)> = None;
                for w in (1..=max_w).rev() {
                    if s + w > n || !(s..s + w).all(|k| roles[k] == ShipTokenRole::Content) { continue; }
                    let joined: String = cores[s..s + w].iter().map(|c| ship_normalize_token(c)).collect();
                    if let Some(codes) = crate::utils::ai_utils::trade_title_exact(&joined) {
                        found = Some((w, codes));
                        break;
                    }
                }
                let (w, codes) = match found {
                    Some(v) => v,
                    None => {
                        s += 1;
                        continue;
                    }
                };
                let next_is_value = matches!(
                    roles.get(s + w),
                    Some(ShipTokenRole::Numeric) | Some(ShipTokenRole::Identifier)
                );
                if next_is_value {
                    s += w;
                    continue;
                }
                for k in s..s + w { roles[k] = ShipTokenRole::DocType; }
                hits.push(format!("{}→{:?}", cores[s..s + w].join(" "), codes));
                doc_mentions.push(ShipDocMention { start: s, end: s + w, codes, exact: true });
                s += w;
            }
            crate::utils::score_dynamics::record_baseline("search.title.exact", hits.len() as f32);
            if !hits.is_empty() {
                logs.push(format!(
                    "   📄 [DOC TYPE / TITLE EXACT] 12개 언어 서식 전문 표와 완전일치한 표현 {}개: {:?} — 서식 이름은 서식 코드와 같은 닫힌 어휘라 코사인 경쟁에 넣지 않습니다. 코사인 경로는 서식 뱅크 전체의 분포에서 z 를 재므로, 여러 서식이 함께 쓰는 단어('invoice'·'인보이스')가 섞인 표현은 전문을 그대로 적어도 게이트를 넘지 못할 수 있습니다. 가장 긴 서식 이름부터 대조하므로 'house bill of lading' 이 'bill of lading' 에 먼저 잡히지 않습니다. 한국어·일본어·중국어는 조사만 붙은 형태('상업송장을')까지 인정하고, 바로 뒤에 수치·식별자가 오면 번호 라벨로 보고 확정하지 않습니다. 라틴 문자 한 단어짜리 서식 이름('Procura'·'Offerte' 처럼 다른 언어에서는 일반 낱말일 수 있는 것)은 이 표로 확정하지 않고 기존 코사인 경로에 맡깁니다.",
                    hits.len(), hits
                ));
            }
        }

        let mut parts: Vec<(usize, ShipTimePart)> = Vec::new();
        let mut i = 0usize;
        while i < n {
            if roles[i] != ShipTokenRole::Content {
                i += 1;
                continue;
            }
            if i + 1 < n && roles[i + 1] == ShipTokenRole::Content {
                let two = format!("{} {}", cores[i], cores[i + 1]);
                if let Some(k) = crate::utils::ai_utils::exact_match_filter_key("time_filters", &two) {
                    roles[i] = ShipTokenRole::Temporal;
                    roles[i + 1] = ShipTokenRole::Temporal;
                    parts.push((i, ShipTimePart::Rel(k.clone())));
                    parts.push((i + 1, ShipTimePart::Rel(k)));
                    i += 2;
                    continue;
                }
            }
            let hit = crate::utils::ai_utils::exact_match_filter_key("time_filters", &cores[i]).or_else(|| {
                crate::utils::ai_utils::prefix_match_filter_stem("time_filters", &cores[i]).map(|(k, _)| k)
            });
            if let Some(k) = hit {
                roles[i] = ShipTokenRole::Temporal;
                parts.push((i, ShipTimePart::Rel(k)));
            }
            i += 1;
        }
        let lab_coh = crate::utils::ai_utils::bank_internal_cohesion(&label_bank);
        let title_gate = crate::utils::ai_utils::gumbel_decision_z(title_embs.len());
        let title_top = |q: &Vec<f32>| -> f32 {
            title_embs.iter().map(|t| cosine_similarity(q, t)).fold(f32::MIN, f32::max)
        };
        let mut title_best: Option<(String, f32, f32, f32)> = None;
        let closed_op: Vec<bool> = (0..n)
            .map(|k| {
                let one = [cores[k].clone()];
                roles[k] == ShipTokenRole::Content
                    && (crate::utils::ai_utils::comparator_exact(&one).is_some()
                        || crate::utils::ai_utils::temporal_operator_exact(&one).is_some()
                        || ship_relation_exact(&cores[k]))
            })
            .collect();
        if closed_op.iter().any(|&c| c) {
            logs.push(format!(
                "   📄 [DOC TYPE / CLOSED CLASS SKIP] 비교 연산자·시간 연산자·관계 표지 표와 완전일치한 토큰 {:?} 은 서식 이름 후보에서 뺍니다. 닫힌 어휘로 역할이 정해진 토큰을 서식 뱅크와 코사인으로 겨루게 두면, 연산자 토큰이 서식 후보 1위가 되어 robust z 게이트를 넘을 수 있고(그때는 라벨·기능어·연산자 대비 우위 검사 하나만 남습니다), 서식 판정의 SDS 기준선(search.title.robust_z · search.title.dominance)도 연산자 토큰의 점수로 채워집니다.",
                (0..n).filter(|&k| closed_op[k]).map(|k| cores[k].clone()).collect::<Vec<_>>()
            ));
        }
        for width in (1..=3usize).rev() {
            let mut s = 0usize;
            while s + width <= n {
                let e = s + width;
                if !(s..e).all(|k| roles[k] == ShipTokenRole::Content && !closed_op[k]) {
                    s += 1;
                    continue;
                }
                let text = cores[s..e].join(" ");
                let q = table.get(&text);
                if q.iter().all(|&v| v == 0.0) || title_embs.is_empty() {
                    s += 1;
                    continue;
                }
                let sims: Vec<f32> = title_embs.iter().map(|t| cosine_similarity(q, t)).collect();
                let cnt = sims.len() as f32;
                let mean = sims.iter().sum::<f32>() / cnt;
                let sd = (sims.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / cnt).sqrt();
                let (top_i, top) = sims
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::MIN), |acc, (i, &v)| if v > acc.1 { (i, v) } else { acc });
                if sd <= 1e-6 {
                    s += 1;
                    continue;
                }
                let z = {
                    let mut sorted = sims.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let med = sorted[sorted.len() / 2];
                    let mut dev: Vec<f32> = sims.iter().map(|x| (x - med).abs()).collect();
                    dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let mad = dev[dev.len() / 2] * 1.4826;
                    if mad > 1e-6 { (top - med) / mad } else { (top - mean) / sd }
                };
                let lab = max_pool_sim(q, &label_bank);
                let fun = max_pool_sim(q, &func_bank);
                let opb = max_pool_sim(q, &op_all_bank);
                let compositional = width == 1
                    || (s..e).all(|k| top > title_top(table.get(&cores[k])));
                let dom = top - lab.max(fun).max(opb);
                if title_best.as_ref().map_or(true, |b| z > b.1) {
                    title_best = Some((text.clone(), z, dom, top));
                }
                if z > title_gate && compositional && top > lab.max(fun).max(opb) {
                    let mut codes: Vec<String> = Vec::new();
                    for (ti, v) in sims.iter().enumerate() {
                        if *v >= top - sd {
                            for c in titles[ti].1.iter() {
                                if !codes.contains(c) { codes.push(c.clone()); }
                            }
                        }
                    }
                    let (ms, me) = if width > 1 {
                        let mut best_k = usize::MAX;
                        let mut best_v = f32::MIN;
                        for k in s..e {
                            let v = title_top(table.get(&cores[k]));
                            if v > best_v {
                                best_v = v;
                                best_k = k;
                            }
                        }
                        if best_k != usize::MAX && best_v >= top - sd {
                            logs.push(format!(
                                "   ✂️ [DOC TYPE / SPAN SHRINK] \"{}\" 의 서식 cos {:.4} 는 단일 토큰 \"{}\" 의 {:.4} 에 서식 뱅크 σ {:.4} 안으로 붙어 있습니다. 나머지 토큰은 서식 판정에 기여하지 않았으므로 내용어로 남기고 그 토큰만 서식으로 잡습니다.",
                                text, top, cores[best_k], best_v, sd
                            ));
                            (best_k, best_k + 1)
                        } else {
                            (s, e)
                        }
                    } else {
                        (s, e)
                    };
                    for k in ms..me { roles[k] = ShipTokenRole::DocType; }
                    logs.push(format!(
                        "   📄 [DOC TYPE / COSINE] \"{}\" → {:?} | 최고 '{}' cos {:.4} | z {:+.3} > √(2lnN) + 최댓값 표준편차 {:.3} | 라벨 cos {:.4}",
                        cores[ms..me].join(" "), codes, titles[top_i].0, top, z, title_gate, lab
                    ));
                    doc_mentions.push(ShipDocMention { start: ms, end: me, codes, exact: false });
                    s = e;
                    continue;
                }
                s += 1;
            }
        }
        if let Some((t, z, dom, top)) = title_best.as_ref() {
            crate::utils::score_dynamics::record_baseline("search.title.robust_z", *z);
            crate::utils::score_dynamics::record_baseline("search.title.dominance", *dom);
            if !doc_mentions.iter().any(|m| !m.exact) {
                logs.push(format!(
                    "   📄 [DOC TYPE MISS] 최고 후보 \"{}\" | 서식 cos {:.4} | robust z {:+.3} (게이트 {:.3}) | 라벨·기능어·연산자 대비 우위 {:+.4}",
                    t, top, z, title_gate, dom
                ));
            }
        }
        doc_mentions.sort_by(|a, b| a.start.cmp(&b.start));

        let relation_bank = table.bank(&relation_phr);
        let mut relation_marks: Vec<usize> = Vec::new();
        {
            let rel_coh = crate::utils::ai_utils::bank_internal_cohesion(&relation_bank);
            for &i in pending.iter() {
                if roles[i] != ShipTokenRole::Content { continue; }
                if ship_relation_exact(&cores[i]) {
                    relation_marks.push(i);
                    logs.push(format!(
                        "   🔗 [RELATION MARK / EXACT] \"{}\" 가 12개 언어 관계 표지 표와 일치합니다. 이 표지는 '어느 서식을 조회할지' 가 아니라 '이 서식이 다른 서식과 이어져 있는지' 를 말합니다.",
                        cores[i]
                    ));
                    continue;
                }
                let q = table.get(&cores[i]);
                if q.iter().all(|&v| v == 0.0) || relation_bank.is_empty() { continue; }
                let rel = max_pool_sim(q, &relation_bank);
                let rival = max_pool_sim(q, &label_bank)
                    .max(max_pool_sim(q, &func_bank))
                    .max(max_pool_sim(q, &op_all_bank));
                if crate::utils::ai_utils::prejudice_dominates(rival, rel, rel_coh) {
                    relation_marks.push(i);
                    logs.push(format!(
                        "   🔗 [RELATION MARK / COSINE] \"{}\" | 관계 뱅크 {:.4} 가 라벨·기능어·연산자 최고 {:.4} 를 응집도 {:.4} 여유까지 넘어섭니다.",
                        cores[i], rel, rival, rel_coh.clamp(0.0, 0.5)
                    ));
                }
            }
            if relation_marks.is_empty() {
                logs.push(format!(
                    "   ⚪ [RELATION MARK NONE] 관계 표지 구 {}개 중 어느 것도 이 질의의 토큰을 설명하지 못했습니다. 서식 언급은 전부 조회 범위로 읽습니다.",
                    relation_phr.len()
                ));
            } else {
                logs.push(format!(
                    "   🔗 [RELATION AXIS] 관계 표지 토큰 {:?} 를 D1 카테고리와 독립된 축으로 확정했습니다. 기존에는 인접 스팬이 reference·hub 카테고리로 판정되어야만 '함께 보여줄 서식' 이 성립했는데, 그 판정은 D1 잡음에 흔들립니다. 관계 표지는 그 자체가 직접 근거입니다.",
                    relation_marks.iter().map(|&i| cores[i].clone()).collect::<Vec<_>>()
                ));
            }
        }

        let mut op_exact: Vec<Option<&'static str>> = vec![None; n];
        let mut op_group: Vec<Option<usize>> = vec![None; n];
        {
            let mut hits: Vec<String> = Vec::new();
            let mut i = 0usize;
            while i < n {
                let open = |k: usize| roles[k] == ShipTokenRole::Content && !relation_marks.contains(&k);
                if !open(i) {
                    i += 1;
                    continue;
                }
                let mut matched: Option<(&'static str, usize)> = None;
                for w in (1..=3usize).rev() {
                    if i + w > n || !(i..i + w).all(|k| open(k)) { continue; }
                    if let Some(key) = crate::utils::ai_utils::comparator_exact(&cores[i..i + w]) {
                        matched = Some((key, w));
                        break;
                    }
                }
                match matched {
                    Some((key, w)) => {
                        for k in i..i + w {
                            roles[k] = ShipTokenRole::Operator;
                            op_exact[k] = Some(key);
                            op_group[k] = Some(i);
                        }
                        hits.push(format!("\"{}\"→{}", cores[i..i + w].join(" "), key));
                        i += w;
                    }
                    None => i += 1,
                }
            }
            crate::utils::score_dynamics::record_baseline("search.role.op_exact", hits.len() as f32);
            if !hits.is_empty() {
                logs.push(format!(
                    "   ⚖️ [OPERATOR / EXACT] 12개 언어 비교 연산자 표와 완전일치한 표현 {}개: {:?} — 비교 연산자는 시간 단위·관계 표지·통화명과 같은 닫힌 어휘라 코사인 경쟁에 넣지 않습니다. 코사인으로 가르면 '이하인' 이 '수하인'·'송하인' 라벨 구에 더 가까워 연산자 자격을 잃고 당사자 축의 값이 됩니다(연산자와 라벨의 코사인 마진은 SDS search.role.op_lab_margin 분포에서 잡음 폭 안입니다). 한국어·일본어·중국어는 조사·어미(인·이고·의·の·的 등)만 붙은 형태까지 인정하고, 결속할 수치가 없으면 아래 UNBINDABLE 이 기능어로 내립니다.",
                    hits.len(), hits
                ));
            }
        }

        {
            let fun_pivots: Vec<Vec<String>> = ship_function_pivot_phrases()
                .into_iter()
                .map(|p| {
                    p.split_whitespace()
                        .map(ship_normalize_token)
                        .filter(|x| !x.is_empty())
                        .collect::<Vec<String>>()
                })
                .filter(|parts| !parts.is_empty() && parts.len() <= 3)
                .collect();
            let title_keys: std::collections::HashSet<String> = titles
                .iter()
                .map(|(t, _)| ship_normalize_token(t))
                .filter(|t| !t.is_empty())
                .collect();
            let mut hits: Vec<String> = Vec::new();
            let mut i = 0usize;
            while i < n {
                let open = |k: usize| roles[k] == ShipTokenRole::Content && !relation_marks.contains(&k);
                if !open(i) {
                    i += 1;
                    continue;
                }
                let mut matched: Option<usize> = None;
                for w in (1..=3usize).rev() {
                    if i + w > n || !(i..i + w).all(|k| open(k)) { continue; }
                    let norm: Vec<String> = cores[i..i + w].iter().map(|c| ship_normalize_token(c)).collect();
                    if !fun_pivots.iter().any(|p| *p == norm) { continue; }
                    let in_title = (0..=2usize).any(|l| {
                        (0..=2usize).any(|r| {
                            if l + r == 0 || l > i || i + w + r > n { return false; }
                            let joined: String = cores[i - l..i + w + r].iter().map(|c| ship_normalize_token(c)).collect();
                            title_keys.contains(&joined)
                        })
                    });
                    if in_title { continue; }
                    if matches!(
                        roles.get(i + w),
                        Some(ShipTokenRole::Numeric) | Some(ShipTokenRole::Identifier) | Some(ShipTokenRole::Temporal)
                    ) {
                        continue;
                    }
                    matched = Some(w);
                    break;
                }
                match matched {
                    Some(w) => {
                        for k in i..i + w { roles[k] = ShipTokenRole::Function; }
                        hits.push(cores[i..i + w].join(" "));
                        i += w;
                    }
                    None => i += 1,
                }
            }
            crate::utils::score_dynamics::record_baseline("search.role.fun_exact", hits.len() as f32);
            if !hits.is_empty() {
                logs.push(format!(
                    "   🗣️ [FUNCTION / EXACT] 12개 언어 요청 동사·담화 표지 표와 완전일치한 표현 {}개를 기능어로 확정합니다: {:?} — 요청 동사와 담화 표지는 닫힌 어휘라 코사인 경쟁에 넣지 않습니다. 역할 채점의 기능어 판정은 라벨 뱅크 응집도만큼의 여유를 요구하는데, 라벨 뱅크가 촘촘하면 표와 글자까지 같은 '보여줘'(cos 1.0)도 그 여유를 넘지 못해 내용어로 남고, 그대로 D1 후보가 되어 '것만'·'중에서' 같은 담화 표지가 식별·정산 카테고리를 차지합니다. 바로 뒤에 수치·식별자·시간이 오는 자리(unter 1000, entre 100 처럼 범위·비교로도 읽히는 표현)와 이웃 토큰과 합쳐 서식 이름이 되는 자리(packing list 의 list)는 이 표로 확정하지 않고 기존 경로에 맡깁니다.",
                    hits.len(), hits
                ));
            }
        }

        {
            let live: Vec<usize> = pending
                .iter()
                .copied()
                .filter(|&i| roles[i] == ShipTokenRole::Content)
                .filter(|&i| !relation_marks.contains(&i))
                .filter(|&i| !table.get(&cores[i]).iter().all(|&v| v == 0.0))
                .collect();
            let queries: Vec<Vec<f32>> = live.iter().map(|&i| table.get(&cores[i]).clone()).collect();
            let mut role_bias: Vec<(String, String, Vec<f32>)> = Vec::new();
            for (key, bank) in [("label", &label_bank), ("function", &func_bank), ("operator", &op_all_bank)] {
                for e in bank.iter() {
                    if e.iter().all(|&v| v == 0.0) { continue; }
                    role_bias.push(("role".to_string(), key.to_string(), e.clone()));
                }
            }
            let no_prej: Vec<(String, String, Vec<f32>)> = Vec::new();
            let (keys, net, _raw) =
                crate::utils::ai_utils::bank_neutral_key_matrix(&queries, &role_bias, &no_prej);
            let slot = |k: &str| keys.iter().position(|x| x == k);
            let (ki_lab, ki_fun, ki_op) = (slot("label"), slot("function"), slot("operator"));
            logs.push(format!(
                "   ⚖️ [ROLE BANK-NEUTRAL] 토큰 {}개를 라벨({}구) · 기능어({}구) · 연산자({}구) 세 축으로 행·열 이중 센터링해 채점합니다. 원시 Max-Pool 절대 비교는 뱅크가 넓을수록 최댓값이 커져 라벨 뱅크가 구조적으로 이깁니다.",
                live.len(), label_bank.len(), func_bank.len(), op_all_bank.len()
            ));
            for (qi, &i) in live.iter().enumerate() {
                let pick = |ki: Option<usize>| -> f32 {
                    match ki {
                        Some(k) if net[k][qi] != f32::MIN => net[k][qi],
                        _ => f32::MIN,
                    }
                };
                let (nlab, nfun, nop) = (pick(ki_lab), pick(ki_fun), pick(ki_op));
                if nlab == f32::MIN && nfun == f32::MIN && nop == f32::MIN { continue; }
                let q = table.get(&cores[i]);
                let lab = max_pool_sim(q, &label_bank);
                let fun = max_pool_sim(q, &func_bank);
                let opb = max_pool_sim(q, &op_all_bank);
                crate::utils::score_dynamics::record_baseline("search.role.fun_margin", nfun - nlab);
                crate::utils::score_dynamics::record_baseline("search.role.op_margin", nop - nlab);
                if nfun >= nop && nfun > nlab && crate::utils::ai_utils::prejudice_dominates(lab, fun, lab_coh) {
                    roles[i] = ShipTokenRole::Function;
                    logs.push(format!(
                        "   🗣️ [FUNCTION] \"{}\" | 중립 기능어 {:+.4} > 라벨 {:+.4} | 원시 cos 기능어 {:.4} vs 라벨 {:.4} × (1 + 응집도 {:.4})",
                        cores[i], nfun, nlab, fun, lab, lab_coh.clamp(0.0, 0.5)
                    ));
                } else if nop > nlab && nop > nfun {
                    roles[i] = ShipTokenRole::Operator;
                    logs.push(format!(
                        "   ⚖️ [OPERATOR WORD] \"{}\" | 중립 연산자 {:+.4} > 라벨 {:+.4} / 기능어 {:+.4} | 원시 cos 연산자 {:.4} vs 라벨 {:.4}",
                        cores[i], nop, nlab, nfun, opb, lab
                    ));
                }
            }
        }

        {
            let mut live: Vec<(usize, f32, f32, bool, String)> = Vec::new();
            for &i in pending.iter() {
                if roles[i] != ShipTokenRole::Operator { continue; }
                if op_exact[i].is_some() { continue; }
                let q = table.get(&cores[i]);
                if q.iter().all(|&v| v == 0.0) || op_keys.len() < 2 { continue; }
                let scored: Vec<f32> = (0..op_keys.len())
                    .map(|ki| {
                        let b = max_pool_sim(q, &op_bias_banks[ki]);
                        let p = max_pool_sim(q, &op_prej_banks[ki]);
                        b - (p - b).max(0.0)
                    })
                    .collect();
                let cnt = scored.len() as f32;
                let mean = scored.iter().sum::<f32>() / cnt;
                let sd = (scored.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / cnt).sqrt();
                if sd <= 1e-6 { continue; }
                let (top_i, top) = scored
                    .iter()
                    .enumerate()
                    .fold((0usize, f32::MIN), |acc, (k, &v)| if v > acc.1 { (k, v) } else { acc });
                let z = (top - mean) / sd;
                let second = scored
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != top_i)
                    .map(|(_, v)| *v)
                    .fold(f32::MIN, f32::max);
                let self_evident = second > f32::MIN && (top - second) > sd;
                let margin = max_pool_sim(q, &op_all_bank) - max_pool_sim(q, &label_bank);
                crate::utils::score_dynamics::record_baseline("search.role.op_selfz", z);
                crate::utils::score_dynamics::record_baseline("search.role.op_lab_margin", margin);
                live.push((i, z, margin, self_evident, op_keys[top_i].clone()));
            }
            if !live.is_empty() {
                let mut sorted: Vec<f32> = live.iter().map(|(_, _, m, _, _)| *m).collect();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let gaps: Vec<f32> = sorted.windows(2).map(|w| w[1] - w[0]).collect();
                let mut widest = 0.0f32;
                let mut widest_at = usize::MAX;
                for (gi, g) in gaps.iter().enumerate() {
                    if *g > widest { widest = *g; widest_at = gi; }
                }
                let edge = if widest_at == usize::MAX { f32::MAX } else { sorted[widest_at + 1] };
                let split_firm = {
                    let rest: Vec<f32> = gaps
                        .iter()
                        .enumerate()
                        .filter(|(gi, _)| *gi != widest_at)
                        .map(|(_, g)| *g)
                        .collect();
                    if rest.len() < 3 {
                        false
                    } else {
                        let n = rest.len() as f32;
                        let mean = rest.iter().sum::<f32>() / n;
                        let sd = (rest.iter().map(|g| (g - mean) * (g - mean)).sum::<f32>() / n).sqrt();
                        sd > 1e-6 && widest - mean >= sd
                    }
                };
                let cut = if sorted.len() >= 2 && widest > 1e-6 {
                    logs.push(format!(
                        "   📐 [OPERATOR SPLIT] 후보 {}개의 '연산자 cos − 라벨 cos' 마진을 정렬해 가장 넓게 벌어진 자리에서 가릅니다. 마진 {:?} | 최대 격차 {:.4} → 기준 {:.4} 이상만 연산자로 남깁니다. 격차 이상치 판정: {} — 최대 격차가 나머지 격차의 평균+표준편차를 넘을 때만 이 분할이 '이 토큰이 연산자인가' 를 답한 것으로 보고, 그때는 뱅크 자체 변별로 아래 무리를 되살리지 않습니다.",
                        sorted.len(),
                        sorted.iter().map(|m| format!("{:.4}", m)).collect::<Vec<_>>(),
                        widest, edge,
                        if split_firm { "성립" } else { "불성립(격차 표본 3개 미만 또는 평탄)" }
                    ));
                    crate::utils::score_dynamics::record_baseline("search.role.op_split_gap", widest);
                    crate::utils::score_dynamics::record_baseline("search.role.op_split_firm", if split_firm { 1.0 } else { 0.0 });
                    edge
                } else {
                    logs.push(
                        "   📐 [OPERATOR SPLIT SKIP] 후보가 하나뿐이거나 마진이 전부 같아 가를 자리가 없습니다. 뱅크 자체 변별만으로 판정합니다.".to_string(),
                    );
                    f32::MAX
                };
                let cut_shown = if cut == f32::MAX { "없음".to_string() } else { format!("{:+.4}", cut) };
                let op_words: std::collections::HashSet<String> = op_bias_phr
                    .iter()
                    .flatten()
                    .flat_map(|p| p.split_whitespace().map(ship_normalize_token).collect::<Vec<String>>())
                    .collect();
                let title_words: std::collections::HashSet<String> = titles
                    .iter()
                    .flat_map(|(t, _)| {
                        let ws: Vec<String> = t
                            .split_whitespace()
                            .map(ship_normalize_token)
                            .filter(|w| !w.is_empty())
                            .collect();
                        let last = ws.len().saturating_sub(1);
                        ws.into_iter()
                            .enumerate()
                            .filter(|(k, _)| *k == 0 || *k == last)
                            .map(|(_, w)| w)
                            .collect::<Vec<String>>()
                    })
                    .filter(|w| w.chars().count() >= 2 && !op_words.contains(w))
                    .collect();
                for (i, z, margin, self_evident, key) in live.into_iter() {
                    let by_split = margin >= cut && margin > 0.0;
                    let by_self = !split_firm && self_evident && margin > 0.0;
                    let survives = by_split || by_self;
                    let q = table.get(&cores[i]);
                    let title_cos = title_top(q);
                    let op_cos = max_pool_sim(q, &op_all_bank);
                    let by_title_cos = !title_embs.is_empty() && title_cos >= op_cos;
                    let title_word = if survives {
                        std::iter::once(cores[i].clone())
                            .chain(raw_variants[i].iter().cloned())
                            .map(|v| ship_normalize_token(&v))
                            .find(|v| title_words.contains(v))
                    } else {
                        None
                    };
                    let revoked = by_title_cos || title_word.is_some();
                    crate::utils::score_dynamics::record_baseline(
                        "search.role.op_title_revoked",
                        if revoked { 1.0 } else { 0.0 },
                    );
                    if revoked {
                        roles[i] = ShipTokenRole::Function;
                        let evidence = match title_word.as_ref() {
                            Some(w) => format!(
                                "서식 이름의 첫 단어나 끝 단어 '{}' 와 같고 연산자 뱅크 어휘에는 없습니다 (서식 이름 cos {:.4} / 연산자 cos {:.4})",
                                w, title_cos, op_cos
                            ),
                            None => format!("서식 이름 cos {:.4} ≥ 연산자 cos {:.4}", title_cos, op_cos),
                        };
                        logs.push(format!(
                            "   ↩️ [OPERATOR REVOKED / TITLE] \"{}\" | {} | 연산자 분할 {} — 서식 이름을 이루는 단어는 비교 연산자가 아닙니다. 서식 판정 게이트를 넘지 못한 서식 일반명('인보이스')은 조회 범위를 말하는 담화 성분이므로 기능어로 둡니다. 연산자로 남기면 옆에 수치가 오는 순간('인보이스 3건') 그 수치에 비교 조건을 붙입니다. 이 판정은 연산자 분할이 끝난 뒤에 합니다. 분할 전에 후보를 빼면 남은 후보의 마진 분포가 바뀌어 분할 기준이 음수 쪽으로 내려가고, 라벨 쪽이 더 가까운 토큰까지 연산자로 남습니다. 서식 이름 단어 대조는 분할을 통과한 토큰에만 합니다. 서식 이름에는 'weight'·'arrival' 같은 라벨 단어도 들어 있어, 분할에서 탈락해 내용어로 돌아갈 토큰까지 기능어로 내리면 라벨을 잃습니다. 서식 이름 가운데 자리의 연결어('of'·'van'·'do'·'za')는 대조에서 뺍니다.",
                            cores[i], evidence, if survives { "통과" } else { "탈락" }
                        ));
                        continue;
                    }
                    if survives {
                        logs.push(format!(
                            "   ⚖️ [OPERATOR SELF-EVIDENCE] \"{}\" → '{}' | 마진 {:+.4} (기준 {}) | 자기 z {:+.3} | 근거: {} — 연산자 뱅크 {}개 중 한 곳만 이 토큰을 배타적으로 설명합니다.",
                            cores[i], key, margin, cut_shown, z,
                            if by_split { "분포 분할" } else { "분할이 이상치로 성립하지 않아 뱅크 자체 변별 + 라벨 대비 우위로 판정" },
                            op_keys.len()
                        ));
                        continue;
                    }
                    roles[i] = ShipTokenRole::Content;
                    if cut != f32::MAX && margin < cut {
                        crate::utils::score_dynamics::record_baseline(
                            "search.role.op_revoked_margin",
                            cut - margin,
                        );
                    }
                    let why = if margin <= 0.0 {
                        "라벨 뱅크 대비 마진이 양수가 아닙니다".to_string()
                    } else if self_evident {
                        format!("연산자 뱅크 안에서는 '{}' 가 1·2위 격차로 앞서지만, 분포 분할이 이상치로 성립한 뒤에는 그 분할이 '이 토큰이 연산자인가' 를 답하고 뱅크 자체 변별은 '연산자 중 어느 것인가' 만 답합니다", key)
                    } else {
                        "뱅크 자체 변별도 없습니다".to_string()
                    };
                    logs.push(format!(
                        "   ↩️ [OPERATOR REVOKED] \"{}\" | 마진 {:+.4} {} 기준 {} | 자기 z {:+.3} | {} — 내용어로 되돌립니다.",
                        cores[i], margin, if margin < cut { "<" } else { "≥" }, cut_shown, z, why
                    ));
                }
            }
        }

        {
            let bindable: Vec<bool> = (0..n)
                .map(|i| roles[i] == ShipTokenRole::Operator && ship_operator_bindable(&roles, i))
                .collect();
            let targets: Vec<usize> = (0..n)
                .filter(|&i| roles[i] == ShipTokenRole::Operator)
                .filter(|&i| !bindable[i])
                .filter(|&i| match op_group[i] {
                    Some(g) => !(0..n).any(|k| op_group[k] == Some(g) && bindable[k]),
                    None => true,
                })
                .collect();
            if !targets.is_empty() {
                let shown: Vec<String> = targets.iter().map(|&i| cores[i].clone()).collect();
                for &i in targets.iter() {
                    roles[i] = ShipTokenRole::Function;
                    op_exact[i] = None;
                    op_group[i] = None;
                    crate::utils::score_dynamics::record_baseline("search.role.op_unbindable", 1.0);
                }
                logs.push(format!(
                    "   ⛓️‍💥 [OPERATOR UNBINDABLE] 연산자로 살아남았지만 {}토큰 안에 결속할 수치·시간·식별자가 없는 토큰 {}개를 기능어로 내립니다: {:?} — 비교 연산자는 비교 대상이 있을 때만 연산자입니다. 대상이 없으면 그 토큰은 '~만', '함께' 같은 담화 표지이고, 연산자로 남겨 두면 뒤의 수치 결속 탐색이 엉뚱한 자리에서 그 연산자를 주워 '이하'를 '미만'으로 바꾸는 식의 조용한 오역이 생깁니다.",
                    SHIP_OPERATOR_BIND_RADIUS, targets.len(), shown
                ));
            } else {
                crate::utils::score_dynamics::record_baseline("search.role.op_unbindable", 0.0);
            }
        }

        let time_gate = gumbel_expected_z(time_banks.len());
        for &i in pending.iter() {
            if roles[i] != ShipTokenRole::Content { continue; }
            let q = table.get(&cores[i]);
            if q.iter().all(|&v| v == 0.0) || time_banks.len() < 2 { continue; }
            let sims: Vec<f32> = time_banks.iter().map(|(_, b)| max_pool_sim(q, b)).collect();
            let cnt = sims.len() as f32;
            let mean = sims.iter().sum::<f32>() / cnt;
            let sd = (sims.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / cnt).sqrt();
            let (top_i, top) = sims
                .iter()
                .enumerate()
                .fold((0usize, f32::MIN), |acc, (k, &v)| if v > acc.1 { (k, v) } else { acc });
            if sd <= 1e-6 { continue; }
            let z = (top - mean) / sd;
            let lab = max_pool_sim(q, &label_bank);
            let fun = max_pool_sim(q, &func_bank);
            let opb = max_pool_sim(q, &op_all_bank);
            if z > time_gate && crate::utils::ai_utils::prejudice_dominates(lab.max(fun).max(opb), top, lab_coh) {
                roles[i] = ShipTokenRole::Temporal;
                logs.push(format!(
                    "   🕒 [RELATIVE TIME / COSINE] \"{}\" → {} | cos {:.4} | z {:+.3} > {:.3}",
                    cores[i], time_banks[top_i].0, top, z, time_gate
                ));
                parts.push((i, ShipTimePart::Rel(time_banks[top_i].0.clone())));
            }
        }

        let month_gate = gumbel_expected_z(month_banks.len());
        for &i in pending.iter() {
            if roles[i] != ShipTokenRole::Content { continue; }
            let adj_num = (i > 0 && roles[i - 1] == ShipTokenRole::Numeric)
                || (i + 1 < n && roles[i + 1] == ShipTokenRole::Numeric);
            if !adj_num { continue; }
            if let Some(m) = crate::utils::ai_utils::month_from_name(&cores[i]) {
                roles[i] = ShipTokenRole::Temporal;
                logs.push(format!(
                    "   🗓️ [MONTH NAME / EXACT] \"{}\" → {}월 | 12개 언어 월 이름 표와 완전일치합니다. 수치 토큰과 맞닿아 있을 때만 승격합니다.",
                    cores[i], m
                ));
                parts.push((i, ShipTimePart::Month(m)));
                continue;
            }
            let q = table.get(&cores[i]);
            if q.iter().all(|&v| v == 0.0) { continue; }
            let sims: Vec<f32> = month_banks.iter().map(|b| max_pool_sim(q, b)).collect();
            let cnt = sims.len() as f32;
            let mean = sims.iter().sum::<f32>() / cnt;
            let sd = (sims.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / cnt).sqrt();
            let (top_i, top) = sims
                .iter()
                .enumerate()
                .fold((0usize, f32::MIN), |acc, (k, &v)| if v > acc.1 { (k, v) } else { acc });
            if sd <= 1e-6 { continue; }
            let z = (top - mean) / sd;
            let lab = max_pool_sim(q, &label_bank);
            let fun = max_pool_sim(q, &func_bank);
            let opb = max_pool_sim(q, &op_all_bank);
            if z > month_gate && crate::utils::ai_utils::prejudice_dominates(lab.max(fun).max(opb), top, lab_coh) {
                roles[i] = ShipTokenRole::Temporal;
                logs.push(format!(
                    "   🗓️ [MONTH NAME / COSINE] \"{}\" → {}월 | cos {:.4} | z {:+.3} > {:.3}",
                    cores[i], top_i + 1, top, z, month_gate
                ));
                parts.push((i, ShipTimePart::Month((top_i + 1) as u32)));
            }
        }

        let mut numerics_raw: Vec<ShipNumeric> = Vec::new();
        for i in 0..n {
            if roles[i] == ShipTokenRole::Identifier {
                if let Some((letters, number)) = ship_split_alpha_numeric(&cores[i]) {
                    let q = table.get(&letters);
                    let fun = max_pool_sim(q, &func_bank);
                    let opb = max_pool_sim(q, &op_all_bank);
                    if let Some((_, rows, foreign)) = enum_banks.iter().find(|(f, _, _)| f == "currency") {
                        let lab = max_pool_sim(q, foreign);
                        let exact = ship_currency_name_exact(&letters)
                            .filter(|c| rows.iter().any(|(r, _)| r.as_str() == *c))
                            .map(|c| (c.to_string(), 1.0f32, 1.0f32));
                        if let Some((code, _, _)) = exact.or_else(|| ship_enum_resolve(q, &letters, rows, fun, opb, lab)) {
                            roles[i] = ShipTokenRole::Numeric;
                            let (value, grouped) = ship_numeric_value(&number);
                            logs.push(format!(
                                "   💱 [UNIT SPLIT] \"{}\" → 수치 {} + 통화 {}",
                                cores[i], value, code
                            ));
                            numerics_raw.push(ShipNumeric { token: i, value, grouped, operator: String::new(), currency: code });
                        }
                    }
                }
            }
        }

        let mut year_like: Vec<(usize, i32)> = Vec::new();
        let mut day_like: Vec<(usize, u32)> = Vec::new();
        let mut conditional_time: Vec<(usize, Vec<ShipTimePart>, f32, f32, bool)> = Vec::new();
        for i in 0..n {
            if roles[i] != ShipTokenRole::Numeric { continue; }
            if numerics_raw.iter().any(|x| x.token == i) { continue; }
            let core = &cores[i];
            if let Some((yy, mm)) = ship_year_month_literal(core) {
                roles[i] = ShipTokenRole::Temporal;
                logs.push(format!(
                    "   🕒 [YEAR-MONTH LITERAL] \"{}\" → Year({}), Month({}) | '연도-월' 두 조각 수치입니다. 마침표 구분은 금액(2000.05)과 겹치므로 '-' 와 '/' 만 인정합니다.",
                    cores[i], yy, mm
                ));
                parts.push((i, ShipTimePart::Year(yy)));
                parts.push((i, ShipTimePart::Month(mm)));
                continue;
            }
            if let Some(lit) = ship_numeric_date_literal(core) {
                roles[i] = ShipTokenRole::Temporal;
                logs.push(format!(
                    "   🕒 [DATE LITERAL] \"{}\" → 세 조각 수치 날짜 | 4자리 연도가 앞이면 연-월-일, 뒤이면 일-월-연으로 읽습니다. 두 조각이 모두 12 이하인 모호한 표기도 일-월 순(de·fr·es·it·pt·nl·cs·ar 관례)으로 둡니다. 정규식 경로는 '19.04.2022' 를 '19.04.20' 으로 잘라 연도를 잃습니다.",
                    cores[i]
                ));
                parts.push((i, ShipTimePart::Iso(lit)));
                continue;
            }
            if let Some(lit) = crate::utils::ai_utils::extract_date_literal(core) {
                roles[i] = ShipTokenRole::Temporal;
                parts.push((i, ShipTimePart::Iso(lit)));
                continue;
            }
            let (value, _) = ship_numeric_value(core);
            let residue = ship_numeric_residue(core);
            let int_val: Option<i64> = if value.contains('.') { None } else { value.parse::<i64>().ok() };
            let digits = value.chars().filter(|c| c.is_ascii_digit()).count();
            if residue.is_empty() {
                if let Some(v) = int_val {
                    if digits == 4 && (1900..=2100).contains(&v) { year_like.push((i, v as i32)); }
                    if digits <= 2 && (1..=31).contains(&v) { day_like.push((i, v as u32)); }
                }
                continue;
            }
            if let (Some(unit_key), Some(v)) = (crate::utils::ai_utils::time_unit_exact(&residue), int_val) {
                let exact = match unit_key {
                    "year" if digits == 4 && (1..=9999).contains(&v) => Some(ShipTimePart::Year(v as i32)),
                    "month" if (1..=12).contains(&v) => Some(ShipTimePart::Month(v as u32)),
                    "day" if (1..=31).contains(&v) => Some(ShipTimePart::Day(v as u32)),
                    _ => None,
                };
                if let Some(p) = exact {
                    roles[i] = ShipTokenRole::Temporal;
                    crate::utils::score_dynamics::record_baseline("search.time.unit_exact", 1.0);
                    logs.push(format!(
                        "   🕒 [TIME UNIT / EXACT] \"{}\" → {:?} | 잔여 \"{}\" 가 12개 언어 단위 표의 '{}' 와 완전일치합니다. 닫힌 어휘는 코사인 경쟁에 넣지 않습니다.",
                        cores[i], p, residue, unit_key
                    ));
                    parts.push((i, p));
                    continue;
                }
            }
            let q = table.get(&residue);
            if q.iter().all(|&v| v == 0.0) { continue; }
            let mut ranked: Vec<(String, f32)> = unit_banks
                .iter()
                .map(|(k, b)| (k.clone(), max_pool_sim(q, b)))
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let (unit_key, unit_top) = match ranked.first() {
                Some((k, s)) => (k.clone(), *s),
                None => continue,
            };
            let lab_r = max_pool_sim(q, &label_bank);
            let fun_r = max_pool_sim(q, &func_bank);
            let op_r = max_pool_sim(q, &op_all_bank);
            crate::utils::score_dynamics::record_baseline("search.time.unit_margin", unit_top - lab_r.max(fun_r).max(op_r));
            let v = match int_val {
                Some(v) => v,
                None => continue,
            };
            let feasible = |k: &str| -> Option<ShipTimePart> {
                match k {
                    "year" if digits == 4 && (1..=9999).contains(&v) => Some(ShipTimePart::Year(v as i32)),
                    "month" if (1..=12).contains(&v) => Some(ShipTimePart::Month(v as u32)),
                    "day" if (1..=31).contains(&v) => Some(ShipTimePart::Day(v as u32)),
                    _ => None,
                }
            };
            let cands: Vec<ShipTimePart> = ranked.iter().filter_map(|(k, _)| feasible(k)).collect();
            let p = match cands.first() {
                Some(p) => p.clone(),
                None => continue,
            };
            let (unit_gap, unit_sd) = {
                let sims: Vec<f32> = ranked.iter().map(|(_, s)| *s).collect();
                if sims.len() < 2 {
                    (0.0f32, 0.0f32)
                } else {
                    let cnt = sims.len() as f32;
                    let mean = sims.iter().sum::<f32>() / cnt;
                    let sd = (sims.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / cnt).sqrt();
                    (sims[0] - sims[1], sd)
                }
            };
            let unit_coh = unit_banks
                .iter()
                .find(|(k, _)| *k == unit_key)
                .map(|(_, b)| crate::utils::ai_utils::bank_internal_cohesion(b))
                .unwrap_or(0.0);
            let rival = lab_r.max(fun_r).max(op_r);
            let lenient = !crate::utils::ai_utils::prejudice_dominates(unit_top, rival, unit_coh);
            let outright = unit_top > rival;
            let self_evident = unit_sd > 1e-6 && unit_gap > unit_sd;
            if !(outright && self_evident) {
                logs.push(format!(
                    "   ⏸️ [TIME UNIT DEFER] \"{}\" | 1위 '{}' cos {:.4} | 경쟁 최고 {:.4} | 단위 뱅크 1·2위 격차 {:.4} vs 표준편차 {:.4} → 단위 축만으로는 확정하지 못했습니다. 가능한 해석 {:?} 를 순서대로 들고 인접 시간 조각과의 연→월→일 연쇄에 맡깁니다. 응집도 여유를 허용하면 경쟁 축이 앞서는데도 단위 판정이 확정되어 월을 일로 읽습니다.",
                    cores[i], unit_key, unit_top, rival, unit_gap, unit_sd, cands
                ));
                conditional_time.push((i, cands, unit_top, rival, lenient));
                continue;
            }
            roles[i] = ShipTokenRole::Temporal;
            logs.push(format!(
                "   🕒 [TIME UNIT / COSINE] \"{}\" → {:?} | 단위 '{}' cos {:.4} > 경쟁 최고 {:.4} | 단위 뱅크 1·2위 격차 {:.4} > 표준편차 {:.4} | 근거: 경쟁 축 완전 우위 + 뱅크 자체 변별",
                cores[i], p, unit_key, unit_top, rival, unit_gap, unit_sd
            ));
            parts.push((i, p));
        }
        for j in 0..n {
            if roles[j] != ShipTokenRole::Content { continue; }
            if crate::utils::ai_utils::time_unit_exact(&cores[j]) != Some("year") { continue; }
            for k in [j.wrapping_sub(1), j + 1] {
                if k >= n || roles[k] != ShipTokenRole::Numeric { continue; }
                let pos = match year_like.iter().position(|(i, _)| *i == k) {
                    Some(pos) => pos,
                    None => continue,
                };
                let (_, y) = year_like.remove(pos);
                roles[k] = ShipTokenRole::Temporal;
                roles[j] = ShipTokenRole::Temporal;
                crate::utils::score_dynamics::record_baseline("search.time.unit_exact", 1.0);
                logs.push(format!(
                    "   🕒 [YEAR UNIT WORD / EXACT] \"{}\" + \"{}\" → Year({}) | 띄어 쓴 연도 단위어가 12개 언어 단위 표의 'year' 와 완전일치하고 이웃 수치가 4자리 연도입니다. 붙여 쓰는 한국어·CJK 와 달리 de·fr·es·it·pt·nl·cs·ar·en 은 단위어가 별도 토큰으로 옵니다. 월·일 단위어는 '3 months' 같은 기간 표현과 겹치므로 연도에만 적용합니다.",
                    cores[j], cores[k], y
                ));
                parts.push((k, ShipTimePart::Year(y)));
                parts.push((j, ShipTimePart::Year(y)));
                break;
            }
        }
        loop {
            let mut promoted = false;
            let mut rest: Vec<(usize, Vec<ShipTimePart>, f32, f32, bool)> = Vec::new();
            for (i, cands, unit_top, rival, lenient) in conditional_time.into_iter() {
                let linked = cands
                    .iter()
                    .find(|p| {
                        [i.wrapping_sub(1), i + 1].iter().any(|k| {
                            parts.iter().any(|(j, q)| {
                                *j == *k
                                    && matches!(
                                        (q, *p),
                                        (ShipTimePart::Year(_), ShipTimePart::Month(_))
                                            | (ShipTimePart::Month(_), ShipTimePart::Day(_))
                                    )
                            })
                        })
                    })
                    .cloned();
                match linked {
                    Some(p) => {
                        roles[i] = ShipTokenRole::Temporal;
                        logs.push(format!(
                            "   🕒 [TIME UNIT / ADJACENT] \"{}\" → {:?} | 단위 cos {:.4} 와 경쟁 cos {:.4} 로는 확정하지 못했지만, 가능한 해석 {:?} 중 인접 시간 조각과 연→월→일 순서로 이어지는 것을 채택합니다. 단위의 순서는 어휘가 아니라 달력 구조가 정합니다.",
                            cores[i], p, unit_top, rival, cands
                        ));
                        parts.push((i, p));
                        promoted = true;
                    }
                    None => rest.push((i, cands, unit_top, rival, lenient)),
                }
            }
            conditional_time = rest;
            let mut rest_years: Vec<(usize, i32)> = Vec::new();
            for (i, y) in year_like.into_iter() {
                let near_time = (i > 0 && parts.iter().any(|(k, p)| *k == i - 1 && matches!(p, ShipTimePart::Month(_) | ShipTimePart::Day(_))))
                    || parts.iter().any(|(k, p)| *k == i + 1 && matches!(p, ShipTimePart::Month(_) | ShipTimePart::Day(_)));
                if near_time {
                    roles[i] = ShipTokenRole::Temporal;
                    parts.push((i, ShipTimePart::Year(y)));
                    promoted = true;
                } else {
                    rest_years.push((i, y));
                }
            }
            year_like = rest_years;
            let mut rest_days: Vec<(usize, u32)> = Vec::new();
            for (i, d) in day_like.into_iter() {
                let near_month = parts
                    .iter()
                    .any(|(k, p)| (*k + 1 == i || *k == i + 1) && matches!(p, ShipTimePart::Month(_)));
                if near_month {
                    roles[i] = ShipTokenRole::Temporal;
                    logs.push(format!(
                        "   🗓️ [DAY / ADJACENT] \"{}\" → Day({}) | 월 조각과 맞닿은 1~31 정수입니다. 일·월·연이 따로 인쇄된 표기를 한 구간으로 묶습니다.",
                        cores[i], d
                    ));
                    parts.push((i, ShipTimePart::Day(d)));
                    promoted = true;
                } else {
                    rest_days.push((i, d));
                }
            }
            day_like = rest_days;
            if !promoted { break; }
        }
        for (i, cands, unit_top, rival, lenient) in conditional_time.into_iter() {
            let p = match cands.first() {
                Some(p) if lenient => p.clone(),
                _ => {
                    logs.push(format!(
                        "   ⚪ [TIME UNIT DROP] \"{}\" | 단위 cos {:.4} 가 경쟁 축 {:.4} 에 응집도 여유까지 내주었고 인접 연쇄도 없습니다. 시간 조각으로 쓰지 않습니다.",
                        cores[i], unit_top, rival
                    ));
                    continue;
                }
            };
            roles[i] = ShipTokenRole::Temporal;
            logs.push(format!(
                "   🕒 [TIME UNIT / ARGMAX FALLBACK] \"{}\" → {:?} | 인접 연쇄가 없어 단위 뱅크 1위 해석을 그대로 씁니다. (단위 cos {:.4} | 경쟁 최고 {:.4}) — 연쇄가 없는 단독 시간 토큰은 종전과 동일하게 처리해 리콜을 잃지 않습니다.",
                cores[i], p, unit_top, rival
            ));
            parts.push((i, p));
        }
        parts.sort_by(|a, b| a.0.cmp(&b.0));

        let mut temporal: Option<ShipTemporal> = None;
        let mut temporal_keep: Vec<usize> = Vec::new();
        if !parts.is_empty() {
            let mut group: Vec<(usize, ShipTimePart)> = vec![parts[0].clone()];
            for p in parts.iter().skip(1) {
                let last = group.last().map(|x| x.0).unwrap_or(0);
                if p.0 == last || p.0 == last + 1 {
                    group.push(p.clone());
                } else {
                    break;
                }
            }
            let today = chrono::Local::now().date_naive();
            let group_parts: Vec<ShipTimePart> = group.iter().map(|(_, p)| p.clone()).collect();
            let mut tokens: Vec<usize> = Vec::new();
            for (k, _) in group.iter() {
                if !tokens.contains(k) { tokens.push(*k); }
            }
            match ship_compose_range(&group_parts, today) {
                Some((mut start, mut end)) => {
                    let first = *tokens.first().unwrap_or(&0);
                    let last = *tokens.last().unwrap_or(&0);
                    let mut operator = "between".to_string();
                    let lo = first.saturating_sub(3);
                    let hi = (last + 4).min(n);
                    let exact_period = crate::utils::ai_utils::exact_absolute_period(&words[lo..hi], today)
                        .map(|p| {
                            let covered: Vec<usize> = p.tokens.iter().map(|t| t + lo).collect();
                            (p, covered)
                        })
                        .filter(|(_, covered)| tokens.iter().all(|t| covered.contains(t)));
                    crate::utils::score_dynamics::record_baseline(
                        "search.time.period_exact",
                        if exact_period.is_some() { 1.0 } else { 0.0 },
                    );
                    let mut exact_range = false;
                    if let Some((p, covered)) = exact_period {
                        let (s, e) = ship_fmt_range(p.start, p.end);
                        let changed = s != start || e != end || p.operator != "between";
                        start = s;
                        end = e;
                        operator = p.operator.to_string();
                        exact_range = p.range;
                        let mut added: Vec<String> = Vec::new();
                        for t in covered.into_iter() {
                            if tokens.contains(&t) { continue; }
                            if !matches!(
                                roles.get(t),
                                Some(ShipTokenRole::Content)
                                    | Some(ShipTokenRole::Function)
                                    | Some(ShipTokenRole::Operator)
                                    | Some(ShipTokenRole::Numeric)
                                    | Some(ShipTokenRole::Temporal)
                            ) {
                                continue;
                            }
                            tokens.push(t);
                            roles[t] = ShipTokenRole::Temporal;
                            added.push(cores[t].clone());
                        }
                        tokens.sort();
                        if changed || !added.is_empty() {
                            logs.push(format!(
                                "   🕒 [TEMPORAL / EXACT PERIOD] {} ~ {} | 연산자 {} | 편입 토큰 {:?} | 근거 {:?} — 연·월·일 단위, 시간 연산자(이후·까지·以降·まで·after·until 등), 같은 단위의 반복(3월부터 5월까지)을 12개 언어 닫힌 어휘 표로 확정했습니다. 코사인으로 모은 시간 조각을 이 표가 전부 설명할 때만 채택하므로, 표가 모르는 조각이 섞인 구간은 기존 경로를 그대로 탑니다. 조각을 하나씩 조립하면 첫 월만 남아 '3월부터 5월까지' 가 3월 한 달로 줄고, 단위에 붙은 '까지' 는 연산자로 읽히지 않습니다.",
                                start, end, operator, added, p.evidence
                            ));
                        }
                    }
                    if operator == "between" && !exact_range {
                        let span_lo = tokens.iter().copied().min().unwrap_or(first);
                        let span_hi = tokens.iter().copied().max().unwrap_or(last);
                        let mut probes: Vec<usize> = vec![span_hi + 1];
                        if span_lo > 0 { probes.push(span_lo - 1); }
                        if span_hi + 2 < n && roles.get(span_hi + 1) == Some(&ShipTokenRole::Content) { probes.push(span_hi + 2); }
                        for j in probes.into_iter() {
                            let open_role = matches!(
                                roles.get(j),
                                Some(ShipTokenRole::Content) | Some(ShipTokenRole::Function) | Some(ShipTokenRole::Operator)
                            );
                            let exact_temporal = if open_role {
                                crate::utils::ai_utils::temporal_operator_exact(&[cores[j].clone()])
                            } else {
                                None
                            };
                            if exact_temporal.is_none() && roles.get(j) != Some(&ShipTokenRole::Operator) { continue; }
                            let q = table.get(&cores[j]);
                            let exact_key = exact_temporal
                                .map(|k| k.to_string())
                                .or_else(|| op_exact.get(j).copied().flatten().map(|k| k.to_string()));
                            if let Some(k) = exact_key.or_else(|| ship_op_key(q, &op_keys, &op_bias_banks, &op_prej_banks)) {
                                if k == "gte" || k == "gt" {
                                    operator = "gte".to_string();
                                } else if k == "lte" || k == "lt" { operator = "lte".to_string(); }
                                if operator != "between" {
                                    tokens.push(j);
                                    roles[j] = ShipTokenRole::Temporal;
                                    break;
                                }
                            }
                        }
                    }
                    let mut label_cands: Vec<usize> = Vec::new();
                    if last + 1 < n { label_cands.push(last + 1); }
                    if first > 0 { label_cands.push(first - 1); }
                    if last + 2 < n && roles.get(last + 1) == Some(&ShipTokenRole::Content) {
                        label_cands.push(last + 2);
                    }
                    if first > 1 && roles.get(first - 1) == Some(&ShipTokenRole::Content) {
                        label_cands.push(first - 2);
                    }
                    let span_lo = tokens.iter().copied().min().unwrap_or(first);
                    let span_hi = tokens.iter().copied().max().unwrap_or(last);
                    let mut outer: Vec<usize> = Vec::new();
                    if span_hi + 1 < n { outer.push(span_hi + 1); }
                    if span_hi + 2 < n && roles.get(span_hi + 1) == Some(&ShipTokenRole::Content) { outer.push(span_hi + 2); }
                    if span_lo > 0 { outer.push(span_lo - 1); }
                    if span_lo > 1 && roles.get(span_lo - 1) == Some(&ShipTokenRole::Content) { outer.push(span_lo - 2); }
                    for j in outer.into_iter() {
                        if !label_cands.contains(&j) { label_cands.push(j); }
                    }
                    let anchored_codes: Vec<String> = doc_mentions
                        .iter()
                        .filter(|m| !ship_mention_by_mark(m, &relation_marks))
                        .flat_map(|m| m.codes.iter().cloned())
                        .collect();
                    let projected = |m: &ShipDocMention| -> bool {
                        ship_mention_by_mark(m, &relation_marks)
                            && !anchored_codes.is_empty()
                            && !m.codes.iter().any(|c| anchored_codes.contains(c))
                            && m.codes.iter().any(|c| crate::logic::trade_reference_field_of(c).is_some())
                    };
                    let projected_codes: Vec<String> = doc_mentions
                        .iter()
                        .filter(|m| projected(m))
                        .flat_map(|m| m.codes.iter().cloned())
                        .collect();
                    crate::utils::score_dynamics::record_baseline(
                        "search.time.scope_relation_drop",
                        projected_codes.len() as f32,
                    );
                    if !projected_codes.is_empty() {
                        logs.push(format!(
                            "   🔗 [DATE FIELD SCOPE / RELATION] 관계 표지 옆에 붙은 서식 {:?} 는 조회 범위가 아니라 함께 보여줄 연결 축이므로 날짜 축 후보 범위에서 뺍니다. 기준은 DOC PROJECTION 의 관계 표지 규칙(범위 서식이 따로 있고, 코드가 겹치지 않으며, 참조 축이 있는 서식)과 같습니다. 값 축(D2 FIELD SCOPE)은 연결 서식을 뺀 범위로 좁히는데 날짜 축만 연결 서식까지 넓히면, 연결 서식에만 있는 날짜 축이 1위가 되어 조회 서식에는 없는 축에 기간이 잠깁니다.",
                            projected_codes
                        ));
                    }
                    let mut scope_codes: Vec<String> = Vec::new();
                    for m in doc_mentions.iter() {
                        if projected(m) { continue; }
                        for c in m.codes.iter() {
                            if !scope_codes.contains(c) { scope_codes.push(c.clone()); }
                        }
                    }
                    let in_scope_schema = |f: &str| -> bool {
                        if scope_codes.is_empty() { return true; }
                        let mut checked = 0usize;
                        for code in scope_codes.iter() {
                            let (known, cat) = crate::model::merge::trade_schema_owner_of(&code.to_uppercase(), f);
                            if !known { continue; }
                            checked += 1;
                            if !cat.is_empty() { return true; }
                        }
                        checked == 0
                    };
                    let scoped_banks: Vec<&(String, Vec<Vec<f32>>)> = date_banks
                        .iter()
                        .filter(|(f, _)| in_scope_schema(f))
                        .collect();
                    if !scope_codes.is_empty() && scoped_banks.len() < date_banks.len() {
                        logs.push(format!(
                            "   🎯 [DATE FIELD SCOPE] 질의가 지목한 서식 {:?} 의 저장 스키마에 존재하는 날짜 축 {}개로 후보를 좁힙니다 (전체 {}개). 날짜 라벨은 어느 서식에서나 비슷하게 읽히므로, 후보를 전 서식 합집합으로 두면 그 서식에 저장될 수 없는 축이 1위가 되어 조건이 확정적으로 0건을 만듭니다.",
                            scope_codes, scoped_banks.len(), date_banks.len()
                        ));
                    }
                    let pool: Vec<&(String, Vec<Vec<f32>>)> = if scoped_banks.is_empty() {
                        date_banks.iter().collect()
                    } else {
                        scoped_banks
                    };
                    // 허브 보정: 'ETA'·'ETD' 같은 짧은 약어 구는 어떤 토큰과도 코사인이 높아 Max-Pool 에서
                    // 자기 축을 대신 이깁니다. 살아 있는 질의 토큰 전체를 배경으로 구마다 평균 코사인을 재고
                    // 그 성분을 뺀 뒤에야 "이 토큰에만 반응한 구" 가 남습니다.
                    let bg: Vec<Vec<f32>> = cores
                        .iter()
                        .map(|c| table.get(c).clone())
                        .filter(|v| !v.iter().all(|&x| x == 0.0))
                        .collect();
                    let hub_of = |e: &Vec<f32>| -> f32 {
                        if bg.is_empty() { return 0.0; }
                        bg.iter().map(|t| cosine_similarity(t, e)).sum::<f32>() / bg.len() as f32
                    };
                    let pool_hub: Vec<Vec<f32>> = pool
                        .iter()
                        .map(|(_, b)| b.iter().map(|e| hub_of(e)).collect())
                        .collect();
                    let nd_hub: Vec<f32> = nondate_bank.iter().map(|e| hub_of(e)).collect();
                    let mut best: Option<(usize, String, f32)> = None;
                    for j in label_cands.into_iter() {
                        if roles.get(j) != Some(&ShipTokenRole::Content) { continue; }
                        let q = table.get(&cores[j]);
                        if q.iter().all(|&v| v == 0.0) { continue; }
                        let mut all: Vec<f32> = Vec::new();
                        let mut per_field: Vec<(String, f32, usize)> = Vec::new();
                        for (fi, (f, b)) in pool.iter().enumerate() {
                            if b.is_empty() { continue; }
                            let mut mx = f32::MIN;
                            for (ei, e) in b.iter().enumerate() {
                                let s = cosine_similarity(q, e) - pool_hub[fi][ei];
                                all.push(s);
                                if s > mx { mx = s; }
                            }
                            per_field.push((f.clone(), mx, b.len()));
                        }
                        let mut nd_mx = f32::MIN;
                        for (ei, e) in nondate_bank.iter().enumerate() {
                            let s = cosine_similarity(q, e) - nd_hub[ei];
                            all.push(s);
                            if s > nd_mx { nd_mx = s; }
                        }
                        if per_field.is_empty() || nondate_bank.is_empty() || all.len() < 2 { continue; }
                        let cnt = all.len() as f32;
                        let mean = all.iter().sum::<f32>() / cnt;
                        let sd = (all.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / cnt).sqrt();
                        if sd <= 1e-6 { continue; }
                        let z_of = |mx: f32, n: usize| -> f32 { (mx - mean) / sd - gumbel_expected_z(n.max(1)) };
                        let mut ranked: Vec<(String, f32)> = per_field
                            .iter()
                            .map(|(f, mx, n)| (f.clone(), z_of(*mx, *n)))
                            .collect();
                        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                        let z_nd = z_of(nd_mx, nondate_bank.len());
                        let (bf, bz) = ranked[0].clone();
                        let second = ranked.get(1).map(|x| x.1).unwrap_or(f32::MIN);
                        // 날짜 축끼리의 z 표준편차. 1·2위 격차가 그 안이면 어느 축인지를 답한 것이 아니라 잡음이
                        // 답한 것이므로 기본 축(issue_date)으로 물러납니다. 틀린 축을 하드 조건으로 잠그면 0건이 확정되고,
                        // 치환으로 살아나더라도 SDS 에는 그 축의 '단독 차단' 이 오기록됩니다.
                        let zs: Vec<f32> = ranked.iter().map(|x| x.1).collect();
                        let zm = zs.iter().sum::<f32>() / zs.len() as f32;
                        let zsd = (zs.iter().map(|x| (x - zm) * (x - zm)).sum::<f32>() / zs.len() as f32).sqrt();
                        let ambiguous = ranked.len() >= 2 && (bz - second) < zsd;
                        logs.push(format!(
                            "   📅 [DATE LABEL PROBE] \"{}\" | 허브 보정 z 순위 [{}] vs 비날짜 라벨 z {:+.3} (구 {}개) | 1·2위 격차 {:+.3} vs 축 간 σ {:.3}{} — 구마다 질의 토큰 전체와의 평균 코사인(허브 성분)을 뺀 뒤 √(2 ln N) 기대 최댓값을 다시 뺐습니다. 약어 한 구가 뱅크 전체를 대신 이기는 자리를 막습니다.",
                            cores[j],
                            ranked.iter().take(3).map(|(f, z)| format!("{}{:+.3}", f, z)).collect::<Vec<_>>().join(", "),
                            z_nd,
                            nondate_bank.len(),
                            bz - second,
                            zsd,
                            if ambiguous { " → 모호, 기본 축으로 후퇴" } else { "" }
                        ));
                        if ambiguous { continue; }
                        if bz >= z_nd && best.as_ref().map_or(true, |b| bz > b.2) {
                            best = Some((j, bf, bz));
                        }
                    }
                    let field = match best {
                        Some((j, f, s)) => {
                            roles[j] = ShipTokenRole::Temporal;
                            tokens.push(j);
                            logs.push(format!(
                                "   📅 [DATE FIELD / COSINE] 날짜 라벨 \"{}\" → {} (기대 최댓값 보정 z {:+.3})",
                                cores[j], f, s
                            ));
                            f
                        }
                        None => {
                            logs.push(
                                "   📅 [DATE FIELD] 인접한 날짜 라벨이 없어 문서 발행일(issue_date) 축으로 둡니다.".to_string(),
                            );
                            "issue_date".to_string()
                        }
                    };
                    logs.push(format!(
                        "   📅 [TEMPORAL] {} ~ {} | 연산자 {} | 토큰 {:?}",
                        start, end, operator, tokens
                    ));
                    temporal_keep = tokens.clone();
                    temporal = Some(ShipTemporal { start, end, operator, tokens, field });
                }
                None => {
                    logs.push(format!(
                        "   ⚪ [TEMPORAL SKIP] 시간 조각 {:?} 로 구간을 만들 수 없습니다 (연·월이 없음).",
                        group_parts
                    ));
                }
            }
        }

        for i in 0..n {
            if roles[i] != ShipTokenRole::Temporal || temporal_keep.contains(&i) { continue; }
            roles[i] = if cores[i].chars().any(|c| c.is_ascii_digit()) {
                ShipTokenRole::Numeric
            } else {
                ShipTokenRole::Content
            };
            logs.push(format!(
                "   ↩️ [TEMPORAL REVERT] \"{}\" 는 채택된 시간 구간 밖이라 원래 역할 {:?} 로 되돌립니다.",
                cores[i], roles[i]
            ));
        }

        let mut enum_hits: Vec<Vec<(String, String)>> = vec![Vec::new(); n];
        for i in 0..n {
            if roles[i] != ShipTokenRole::Content { continue; }
            let adj_num = (i > 0 && roles[i - 1] == ShipTokenRole::Numeric)
                || (i + 1 < n && roles[i + 1] == ShipTokenRole::Numeric);
            let q = table.get(&cores[i]);
            let fun = max_pool_sim(q, &func_bank);
            let opb = max_pool_sim(q, &op_all_bank);
            for (field, rows, foreign) in enum_banks.iter() {
                if field == "currency" && adj_num {
                    let exact = ship_currency_name_exact(&cores[i])
                        .or_else(|| ship_currency_symbol(&words[i]))
                        .filter(|c| rows.iter().any(|(r, _)| r.as_str() == *c));
                    if let Some(code) = exact {
                        logs.push(format!(
                            "   🏷️ [ENUM VALUE / EXACT] \"{}\" → {} = {} | 통화 명칭·기호 표와 완전일치합니다. 수치 토큰과 맞닿아 있을 때만 적용합니다.",
                            cores[i], field, code
                        ));
                        enum_hits[i].push((field.clone(), code.to_string()));
                        continue;
                    }
                }
                if q.iter().all(|&v| v == 0.0) { continue; }
                let lab = max_pool_sim(q, foreign);
                if let Some((code, cos, gap)) = ship_enum_resolve(q, &cores[i], rows, fun, opb, lab) {
                    logs.push(format!(
                        "   🏷️ [ENUM VALUE / COSINE] \"{}\" → {} = {} | cos {:.4} | 1·2위 격차 {:+.4}",
                        cores[i], field, code, cos, gap
                    ));
                    enum_hits[i].push((field.clone(), code));
                }
            }
        }

        for i in 0..n {
            if roles[i] != ShipTokenRole::Numeric { continue; }
            if numerics_raw.iter().any(|x| x.token == i) { continue; }
            let (value, grouped) = ship_numeric_value(&cores[i]);
            if value.is_empty() { continue; }
            let mut currency = String::new();
            let residue = ship_numeric_residue(&cores[i]);
            let exact_cur = ship_currency_name_exact(&residue)
                .or_else(|| ship_currency_symbol(&words[i]))
                .filter(|c| {
                    enum_banks
                        .iter()
                        .any(|(f, rows, _)| f == "currency" && rows.iter().any(|(r, _)| r.as_str() == *c))
                });
            if let Some(code) = exact_cur {
                currency = code.to_string();
                logs.push(format!(
                    "   💱 [CURRENCY / EXACT] \"{}\" → {} | 수치에 붙은 통화 명칭·기호가 표와 완전일치합니다.",
                    words[i], code
                ));
            } else if !residue.is_empty() {
                let q = table.get(&residue);
                let fun = max_pool_sim(q, &func_bank);
                let opb = max_pool_sim(q, &op_all_bank);
                if let Some((_, rows, foreign)) = enum_banks.iter().find(|(f, _, _)| f == "currency") {
                    let lab = max_pool_sim(q, foreign);
                    if let Some((code, _, _)) = ship_enum_resolve(q, &residue, rows, fun, opb, lab) {
                        currency = code;
                    }
                }
            }
            if currency.is_empty() {
                for j in [i + 1, i.wrapping_sub(1)] {
                    if j >= n || roles[j] != ShipTokenRole::Content { continue; }
                    if let Some((_, code)) = enum_hits[j].iter().find(|(f, _)| f == "currency") {
                        currency = code.clone();
                        roles[j] = ShipTokenRole::Unit;
                        logs.push(format!(
                            "   💱 [UNIT BIND] \"{}\" 는 수치 {} 의 통화 단위 {} 로 결속됩니다.",
                            cores[j], value, code
                        ));
                        break;
                    }
                }
            }
            numerics_raw.push(ShipNumeric { token: i, value, grouped, operator: String::new(), currency });
        }

        let mut op_of_token: Vec<Option<String>> = vec![None; n];
        for i in 0..n {
            if roles[i] == ShipTokenRole::Operator {
                op_of_token[i] = match op_exact[i] {
                    Some(k) => Some(k.to_string()),
                    None => ship_op_key(table.get(&cores[i]), &op_keys, &op_bias_banks, &op_prej_banks),
                };
            }
        }
        for num in numerics_raw.iter_mut() {
            let i = num.token;
            let residue = ship_numeric_residue(&cores[i]);
            if !residue.is_empty() {
                if let Some(k) = crate::utils::ai_utils::comparator_exact(&[residue.clone()]) {
                    num.operator = k.to_string();
                } else {
                    let q = table.get(&residue);
                    let lab = max_pool_sim(q, &label_bank);
                    let fun = max_pool_sim(q, &func_bank);
                    let opb = max_pool_sim(q, &op_all_bank);
                    if opb > lab && opb > fun {
                        if let Some(k) = ship_op_key(q, &op_keys, &op_bias_banks, &op_prej_banks) { num.operator = k; }
                    }
                }
            }
            if !num.operator.is_empty() { continue; }
            for exact_only in [true, false] {
                let mut found: Option<String> = None;
                'search: for dist in 1..=2usize {
                    for j in [i + dist, i.wrapping_sub(dist)] {
                        if j >= n { continue; }
                        let (lo, hi) = if j > i { (i + 1, j) } else { (j + 1, i) };
                        let clear = (lo..hi).all(|k| {
                            matches!(roles[k], ShipTokenRole::Content | ShipTokenRole::Unit)
                        });
                        if !clear { continue; }
                        if roles[j] != ShipTokenRole::Operator { continue; }
                        if exact_only && op_exact[j].is_none() { continue; }
                        if let Some(k) = op_of_token[j].clone() {
                            found = Some(k);
                            break 'search;
                        }
                    }
                }
                if let Some(k) = found {
                    num.operator = k;
                    break;
                }
            }
        }
        numerics_raw.sort_by(|a, b| a.token.cmp(&b.token));
        for num in numerics_raw.iter() {
            logs.push(format!(
                "   🔢 [NUMERIC] \"{}\" → 값 {} | 연산자 {} | 통화 {}",
                cores[num.token],
                num.value,
                if num.operator.is_empty() { "-" } else { &num.operator },
                if num.currency.is_empty() { "-" } else { &num.currency }
            ));
        }

        let mut identifiers: Vec<(usize, String)> = Vec::new();
        for i in 0..n {
            if roles[i] == ShipTokenRole::Identifier { identifiers.push((i, cores[i].clone())); }
        }

        let mut variants: Vec<Vec<String>> = vec![Vec::new(); n];
        for i in 0..n {
            if roles[i] != ShipTokenRole::Content || raw_variants[i].is_empty() { continue; }
            let q = table.get(&cores[i]);
            if q.iter().all(|&v| v == 0.0) { continue; }
            let mut rival = f32::MIN;
            for j in 0..n {
                if j == i || roles[j] != ShipTokenRole::Content { continue; }
                let s = cosine_similarity(q, table.get(&cores[j]));
                if s > rival { rival = s; }
            }
            let mut scored: Vec<(String, f32)> = raw_variants[i]
                .iter()
                .map(|v| (v.clone(), cosine_similarity(q, table.get(v))))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut kept: Vec<String> = Vec::new();
            let mut dropped: Vec<String> = Vec::new();
            if rival == f32::MIN {
                if let Some((v, _)) = scored.first() { kept.push(v.clone()); }
            } else {
                for (v, s) in scored.into_iter() {
                    if s > rival {
                        kept.push(v);
                    } else {
                        dropped.push(v);
                    }
                }
            }
            if !dropped.is_empty() {
                logs.push(format!(
                    "   ✂️ [VARIANT GATE] \"{}\" 의 변형 {:?} 폐기 (자기 표면형과의 cos 가 질의 내 다른 토큰 최고 cos {:.4} 이하)",
                    cores[i], dropped, rival
                ));
            }
            variants[i] = kept;
        }

        logs.push(format!(
            "   🧩 [TOKEN ROLES] {:?}",
            (0..n).map(|i| format!("{}:{:?}", words[i], roles[i])).collect::<Vec<_>>()
        ));

        ShipQueryLayers {
            roles,
            variants,
            numerics: numerics_raw,
            identifiers,
            doc_mentions,
            relation_marks,
            enum_hits,
            temporal,
            logs,
        }
    }
}

pub const SHIP_FUNCTION_PIVOTS: &str =
    "show me, show, display, tell me, let me know, list, list them, give me, \
     find, search for, look up, retrieve, group, group them, group by, bundle, bundle them, \
     only, just, only those, those only, among, among those, of those, from those, out of these, \
     please, I want, I need, I would like, \
     also, as well, too, in addition, along with, together, \
     summarize, sort, sort by, order by, arrange by";

pub const SHIP_FUNCTION_PIVOTS_ML: &str =
    "zeig mir, zeigen, anzeigen, sag mir, teile mir mit, auflisten, gib mir, \
     finde, suche, nachschlagen, gruppieren, bündeln, \
     nur, lediglich, nur diese, unter, davon, aus diesen, \
     bitte, ich möchte, ich brauche, auch, ebenfalls, zusammen, \
     zusammenfassen, sortieren, sortieren nach, \
     montre-moi, montrer, afficher, dis-moi, fais-moi savoir, lister, donne-moi, \
     trouve, cherche, rechercher, regrouper, grouper, \
     seulement, uniquement, ceux-là seulement, parmi, de ceux-ci, \
     s'il te plaît, je veux, j'ai besoin, aussi, également, ensemble, \
     résumer, trier, trier par, \
     muéstrame, mostrar, ver, dime, indícame, listar, dame, \
     encuentra, busca, consultar, agrupar, agrupa, \
     solo, únicamente, solo esos, entre, de esos, de estos, \
     por favor, quiero, necesito, también, además, junto con, \
     resumir, ordenar, ordenar por, \
     mostrami, mostra, visualizza, dimmi, fammi sapere, elenca, dammi, \
     trova, cerca, consultare, raggruppa, raggruppare, \
     soltanto, solo quelli, tra, di quelli, \
     per favore, voglio, ho bisogno, anche, inoltre, insieme, \
     riassumi, ordina, ordina per, \
     mostre-me, exibir, diga-me, me informe, me dê, \
     encontre, busque, agrupe, \
     apenas, somente, apenas esses, desses, \
     quero, preciso, além disso, juntamente com, \
     laat me zien, toon, weergeven, vertel me, laat weten, lijst, geef me, \
     zoek, opzoeken, groepeer, groeperen, \
     alleen, slechts, alleen deze, onder, van deze, \
     alsjeblieft, ik wil, ik heb nodig, ook, eveneens, samen, \
     samenvatten, sorteren, sorteren op, \
     ukaž mi, zobraz, ukázat, řekni mi, dej vědět, vypiš, dej mi, \
     najdi, hledej, vyhledat, seskup, seskupit, \
     pouze, jen, jen tyto, mezi, z těchto, \
     prosím, chci, potřebuji, také, rovněž, společně, \
     shrň, seřadit, seřadit podle, \
     أرني, اعرض, أظهر, أخبرني, أعلمني, اسرد, أعطني, \
     ابحث, ابحث عن, استعلم, جمع, اجمع, صنف, \
     فقط, فحسب, هذه فقط, من بين, من هذه, \
     من فضلك, أريد, أحتاج, أيضا, كذلك, مع, \
     لخص, رتب, رتب حسب, \
     見せて, 表示して, 教えて, 知らせて, 一覧にして, 出して, \
     探して, 検索して, 照会して, まとめて, グループにして, \
     だけ, のみ, それだけ, のうち, その中で, \
     お願いします, したい, 必要です, も, また, 一緒に, \
     要約して, 並べ替えて, 順に並べて, \
     给我看, 显示, 展示, 告诉我, 列出, 给我, \
     查找, 搜索, 查询, 分组, 归类, \
     只, 仅, 只要这些, 其中, 这些中, \
     请, 我要, 我需要, 也, 还有, 一起, \
     总结, 排序, 按顺序排列, \
     보여줘, 보여주세요, 표시해줘, 알려줘, 알려주고, 알려주세요, \
     나열해줘, 목록으로, 찾아줘, 검색해줘, 조회해줘, \
     묶어서, 묶어줘, 그룹으로, 정리해줘, \
     것만, 인 것만, 한 것만, 오직, 단지, \
     중에서, 중에, 그중에서, 가운데, \
     부탁해, 하고 싶어, 필요해, 또한, 역시, 같이, 함께, \
     요약해줘, 정렬해줘, 순서대로";

pub fn ship_function_pivot_phrases() -> Vec<String> {
    crate::logic::anchor_phrases(SHIP_FUNCTION_PIVOTS, SHIP_FUNCTION_PIVOTS_ML)
}

pub const SHIP_RELATION_PIVOTS: &str =
    "linked, linked to, connected, connected to, related, related to, associated, associated with, \
     corresponding, corresponding to, attached, referenced, tied to, matching, \
     linked document, connected document, related document, related documents, \
     its linked, the associated one, the corresponding one";

pub const SHIP_RELATION_PIVOTS_ML: &str =
    "verknüpft, verknüpft mit, verbunden, verbunden mit, zugehörig, zugeordnet, \
     entsprechend, beigefügt, referenziert, dazugehörig, \
     verknüpftes Dokument, zugehöriges Dokument, zugehörige Dokumente, \
     lié, lié à, connecté, associé, associé à, correspondant, joint, référencé, rattaché, \
     document lié, document associé, documents associés, \
     vinculado, vinculado a, conectado, relacionado, relacionado con, asociado, \
     correspondiente, adjunto, referenciado, \
     documento vinculado, documento relacionado, documentos relacionados, \
     collegato, collegato a, connesso, correlato, correlato a, associato, \
     corrispondente, allegato, referenziato, \
     documento collegato, documenti correlati, \
     conectado a, relacionado a, associado a, correspondente, anexo, \
     documento vinculado, documentos relacionados, \
     gekoppeld, gekoppeld aan, verbonden, gerelateerd, gerelateerd aan, geassocieerd, \
     overeenkomstig, bijgevoegd, gerefereerd, \
     gekoppeld document, gerelateerd document, gerelateerde documenten, \
     propojený, propojený s, spojený, související, související s, přiřazený, \
     odpovídající, přiložený, odkazovaný, \
     propojený dokument, související dokument, související dokumenty, \
     مرتبط, مرتبط بـ, متصل, ذو صلة, مرتبط به, مرافق, مقابل, مرفق, مشار إليه, \
     المستند المرتبط, المستندات ذات الصلة, \
     紐づく, 紐づいた, 紐づけられた, 関連する, 関連した, 連携した, 対応する, 添付の, 参照された, \
     関連書類, 紐づく書類, 対応する書類, \
     关联的, 关联, 相关的, 相关, 对应的, 附带的, 引用的, 挂钩的, \
     关联单据, 相关单据, 对应单据, \
     연결된, 연결, 연계된, 연계, 관련된, 관련, 대응되는, 딸린, 붙은, 참조된, \
     연결 문서, 관련 서류, 대응 서류";

pub fn ship_relation_pivot_phrases() -> Vec<String> {
    crate::logic::anchor_phrases(SHIP_RELATION_PIVOTS, SHIP_RELATION_PIVOTS_ML)
}

pub fn ship_relation_exact(core: &str) -> bool {
    let norm = ship_normalize_token(core);
    if norm.chars().count() < 2 { return false; }
    for raw in [SHIP_RELATION_PIVOTS, SHIP_RELATION_PIVOTS_ML] {
        for p in raw.split(',') {
            let p = ship_normalize_token(p);
            if p.chars().count() < 2 { continue; }
            if norm == p { return true; }
            if let Some(rest) = norm.strip_prefix(p.as_str()) {
                if crate::utils::ai_utils::short_tail_ok(rest, 3) {
                    return true;
                }
            }
        }
    }
    false
}
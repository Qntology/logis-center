use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use serde_json::{json, Value};
use crate::store::VectorStore;
use crate::model::LogisModel;
use crate::scheduler::translit::generate_transliteration_aliases;
use tauri::Emitter;

/// [SYNONYM EXPANSION] 생성된 별칭을 item_chunks 에 추가 행으로 저장합니다.
///
/// 저장 벡터는 원본 청크와 **동일한 v3 형식 인지 3중 합성**을 사용합니다.
///   ① chunk  = 별칭 그 자체
///   ② anchor = indexing_anchor_text() (원본과 동일한 라벨 개념 축)
///   ③ local  = "{leaf_label} {별칭}"
/// Text/Address 는 (0.25 / 0.10 / 0.65) 이므로 별칭이 벡터를 지배합니다.
///
/// property / property_format 을 원본과 동일하게 두는 이유:
///   lib.rs 의 STAGE-4B(조건 매칭 Column 트랙)와 STAGE-4C(property 타겟 검색)가
///   property 문자열로 동작하기 때문입니다. 별칭에 다른 property 를 주면
///   그 두 경로에서 별칭이 통째로 배제됩니다.
pub async fn upsert_alias_chunks(
    store: &VectorStore,
    model: &LogisModel,
    item_id: &str,
    base_chunk_id: &str,
    page_type: &str,
    doc_lang: &str,
    chunk_meta: &crate::nl_convert::ChunkMetadata,
    aliases: &(String, String),
    cc: &str,
    bcc: &str,
    ref_val: &str,
    search_mode: &str,
) -> usize {
    let mut saved = 0usize;

    if aliases.0.trim().is_empty() && aliases.1.trim().is_empty() {
        return 0;
    }

    let anchor_text = crate::utils::ai_utils::indexing_anchor_text(doc_lang, page_type, &chunk_meta.property);
    let leaf = crate::utils::ai_utils::indexing_leaf_label(doc_lang, page_type, &chunk_meta.property);

    let variants: [(&str, &String); 2] = [("tn", &aliases.0), ("tr", &aliases.1)];

    // 🌟 [ORIGIN VALUE FALLBACK] 음차 품질은 LLM 편차가 큽니다.
    //    (log 실측: 'Cable' → '불랙드' 처럼 명백한 오음차가 섞임)
    //    별칭 벡터가 통째로 빗나가면 그 행은 영구적으로 죽은 벡터가 되어
    //    저장 비용만 쓰고 검색에는 한 번도 기여하지 못합니다.
    //    원본 값을 낮은 비중으로 섞어, 음차가 빗나가도 최소한 원본 표기 축으로는
    //    반응하도록 만들어 별칭 행이 리콜 보험 역할을 하게 합니다.
    let origin_value = chunk_meta.value_part.trim().to_string();

    // 🌟 [CROSSOVER / BATCH-AHEAD] 이 청크의 별칭에 필요한 텍스트를 먼저 전부 모읍니다.
    //
    //  ── 무엇이 문제였나 ──
    //   변종(native / roman)마다 get_embedding_batch(vec![3개]) 를 개별 호출해
    //   청크 20개 × 변종 2개 = 최대 40회 왕복이 발생했습니다.
    //   그 사이사이에 생성 모델이 올라와 있으면 왕복마다 피크가 재현됩니다.
    //   텍스트를 먼저 모아 '한 번' 만 호출하면 왕복이 1회로 줄고,
    //   호출 시점도 한 곳으로 모여 페이즈 판정이 가능해집니다.
    let mut pending_alias: Vec<(&str, String, String)> = Vec::new(); // (suffix, alias, localized)
    for (suffix, alias) in variants.iter() {
        let a = alias.trim();
        if a.is_empty() { continue; }

        // 🌟 [VALUE-DOMINANT LOCALIZED] "{짧은 문서언어 라벨} {별칭} {원본값}"
        //    leaf 는 indexing_leaf_label() 이 뽑은 단일 라벨(예: '상품명')이므로
        //    값이 희석되지 않습니다. 원본값을 뒤에 붙여 두 표기 체계를 한 벡터에 담습니다.
        let localized = {
            let mut s = String::new();
            if !leaf.trim().is_empty() { s.push_str(leaf.trim()); s.push(' '); }
            s.push_str(a);
            if !origin_value.is_empty() && !a.eq_ignore_ascii_case(&origin_value) {
                s.push(' ');
                s.push_str(&origin_value);
            }
            s
        };
        pending_alias.push((suffix, a.to_string(), localized));
    }
    if pending_alias.is_empty() { return 0; }

    // 🌟 [CROSSOVER] 임베딩을 부르기 직전에 페이즈를 선언합니다.
    //    생성 모델이 상주 중이면 예산을 보고 유지하거나 양보시킵니다.
    let _ = model.enter_embedding_phase("alias chunk embedding").await;

    let mut batch_texts: Vec<String> = Vec::with_capacity(pending_alias.len() * 3);
    for (_, a, localized) in pending_alias.iter() {
        batch_texts.push(a.clone());
        batch_texts.push(anchor_text.clone());
        batch_texts.push(localized.clone());
    }
    // 🌟 [CROSSOVER] 청킹 · activation 관측 · 캐시는 get_embedding_batch 내부가 담당합니다.
    //    여기서 anchor_text 를 변종 수만큼 중복해 넣어도 배치 내 중복 제거가
    //    한 벌로 접어 주므로 실연산은 늘지 않습니다.
    let batch_embs = model
        .get_embedding_batch(batch_texts.clone())
        .await
        .unwrap_or_else(|_| vec![vec![0.0; 384]; batch_texts.len()]);
    if batch_embs.len() < pending_alias.len() * 3 { return 0; }

    for (vi, (suffix, alias_owned, localized)) in pending_alias.iter().enumerate() {
        let a = alias_owned.as_str();
        let embs = &batch_embs[vi * 3..vi * 3 + 3];

        let (w_chunk, w_anchor, w_local) = match chunk_meta.property_format.as_str() {
            "Text" | "Address" | "Synthesis" => (0.25f32, 0.10f32, 0.65f32),
            _ => (0.40f32, 0.30f32, 0.30f32),
        };

        let mut final_vec = vec![0.0f32; 384];
        for d in 0..384 {
            final_vec[d] = embs[0][d] * w_chunk
                + embs[1][d] * w_anchor
                + embs[2][d] * w_local;
        }
        let norm: f32 = final_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for d in 0..384 { final_vec[d] /= norm; }
        }

        let alias_chunk_id = format!("{}_{}", base_chunk_id, suffix);
        let _ = store.upsert_chunk(
            &alias_chunk_id,
            item_id,
            page_type,
            a,
            &chunk_meta.property,
            &chunk_meta.property_format,
            a,
            Some(final_vec),
            Some(cc),
            Some(bcc),
            Some(ref_val),
            Some(search_mode),
        ).await;
        saved += 1;

        // 🌟 [ALIAS INDEX LOG] 정방향 로그의 [ALIAS CHUNK HIT] 와 chunk_id 로 대조하기 위해
        //    어떤 id 로 무엇이 저장됐는지 남깁니다.
        //    지금까지는 "저장은 됐는데 검색에 안 잡힌다" 를 로그만으로 증명할 수 없었습니다.
        println!(
            "      💾 [ALIAS INDEXED] chunk_id='{}' | property='{}' | alias='{}' | localized='{}'",
            alias_chunk_id, chunk_meta.property, a, localized
        );
    }

    saved
}

// =====================================================================
// 🌟 [CLOUD-SYNC LOCAL EMBEDDING] 클라우드(Cloudflare)가 "구조화만" 수행하고 내려보낸 아이템을
//    로컬 임베딩 모델로 벡터화 + item_chunks 인덱싱하는 재사용 파이프라인입니다.
//    - GPU 유무와 무관하게 임베딩은 항상 Client App 트랙에서만 수행됩니다.
//    - scheduler 의 PHASE A~E 와 동일한 v3 형식 인지 3중 합성 벡터를 생성합니다.
// =====================================================================
pub async fn index_item_chunks(
    store: &VectorStore,
    model: &LogisModel,
    item_id: &str,
    item_type: &str,
    doc_lang: &str,
    item_json: &Value,
    is_detail: bool,
    cc: &str,
    bcc: &str,
    ref_val: &str,
    search_mode: &str,
    url: &str,
    cancel: &Arc<AtomicBool>,
    app_handle: &tauri::AppHandle,
    task_id: &str,
    skip_transliteration: bool,
) -> anyhow::Result<usize> {
    let page_type = item_type;

    let emit = |msg: &str| {
        println!("{}", msg);
        let _ = app_handle.emit("task-console-log", json!({"task_id": task_id, "text": format!("{}\n", msg)}));
    };

    if cancel.load(Ordering::Relaxed) {
        return Ok(0);
    }

    // 🌟 [MODE GUARD] mode 가 비면 item_chunks 의 mode 컬럼이 빈 문자열이 되고,
    //    STAGE-4 의 `mode = 'commerce'` 필터에서 전량 탈락합니다.
    //    (analytic 트랙에서 청크 검색이 0건이던 원인)
    //    호출부가 빈 값을 넘겨도 안전하도록 여기서 방어합니다.
    let search_mode = if search_mode.trim().is_empty() {
        item_json.get("mode").and_then(|v| v.as_str()).unwrap_or("commerce")
    } else {
        search_mode
    };

    // 🌟 [DOMAIN CONSISTENCY GATE] search_mode 와 page_type 의 도메인이 어긋나면
    //    스키마를 '로드하기 전에' 차단합니다.
    //  ── 판정 근거 ──
    //   서식 코드 목록을 여기에 다시 적지 않습니다.
    //   canonical_bias_type() 이 무역 서식 코드 55종을 'shipping_doc' 으로
    //   접는다는 사실 하나만 사용하므로, 서식이 늘어도 이 코드는 수정 대상이 아닙니다.
    // 🌟 [DOMAIN CONSISTENCY GATE / 단방향]
    //
    //  ── mode 값의 실제 집합 ──
    //   lib.rs 의 ai_search_complex 가 진실의 원천입니다.
    //     match search_mode.as_str() { "shipping" => .., "analytic" => .., _ => commerce }
    //   즉 mode 는 commerce / shipping / analytic 3종이며 "trading" 은 존재하지 않습니다.
    //   ("trading" 은 scheduler/trading.rs 라는 파이프라인 이름일 뿐입니다)
    //
    //  ── 왜 단방향인가 ──
    //   shipping 모드에서 commerce 스키마(tracking/배송 축)가 등장하는 것은 정상입니다.
    //   배송 축은 두 도메인에 걸쳐 있기 때문입니다.
    //   막아야 하는 것은 그 반대, 즉 commerce·analytic 모드에서
    //   무역 서식 87필드 스키마가 로드되는 경우뿐입니다.
    //
    //  ── 실측 사고 (log.txt) ──
    //   [DB-FETCH] Filter: Some("mode = 'commerce'")
    //   [EMBED-LOCAL] 20 pending item(s) detected.
    //   [SCHEMA] 🚢 'FC' 조건부 로드: 카테고리 10개 | 필드 87개
    //   commerce 로 스캔했는데 무역 서식 20건(FC/CN/DN/IP/BE/WR/SR/BK/CSI/
    //   ID/ED/FCR/POD/AN/DO/AWB/SWB/HBL/BL)이 잡혔습니다.
    //   이는 그 문서들의 mode 물리 컬럼이 'commerce' 로 잘못 태깅되어 있다는 뜻입니다.
    //   (뿌리는 lib.rs 의 upsert_items — Part 2 에서 함께 고칩니다)
    //
    //  ── 판정 근거 ──
    //   서식 코드 목록을 여기에 다시 적지 않습니다.
    //   canonical_bias_type() 이 무역 서식 55종을 'shipping_doc' 으로 접는다는
    //   사실 하나만 사용하므로, 서식이 늘어도 이 코드는 수정 대상이 아닙니다.
    {
        let declared_mode = if search_mode.trim().is_empty() { "commerce" } else { search_mode.trim() };
        let is_trade_schema =
            crate::utils::bias_schema::canonical_bias_type(page_type) == "shipping_doc";
        if is_trade_schema && declared_mode != "shipping" {
            emit(&format!(
                "  ⏭️ [DOMAIN MISMATCH] item_id='{}' type='{}' 은 무역 서식(shipping_doc)인데 mode='{}' 로 요청되었습니다. 무역 스키마 로드 없이 청크 인덱싱을 건너뜁니다.",
                item_id, page_type, declared_mode
            ));
            return Ok(0);
        }
    }
    let natural_text = crate::nl_convert::json_to_natural_language(item_json);
    let raw_chunks = crate::nl_convert::split_natural_language_to_chunks(&natural_text);
    if raw_chunks.is_empty() {
        return Ok(0);
    }
    let fields = if is_detail {
        crate::parsing::get_detail_schema_fields(page_type, url, doc_lang)
    } else {
        crate::parsing::get_list_schema_fields(page_type, url, doc_lang)
    };

    // 🌟 [SCHEMA GUARD] analytics 트랙(click / hover / change / report)은 bias.json 에
    //    대응 스키마가 없어 fields 가 항상 비어 있습니다.
    //    뱅크가 비면 PLINKO 는 전 청크를 unclassified 로 떨어뜨리므로,
    //    임베딩 배치를 헛돌리지 않고 여기서 즉시 종료합니다.
    //    (아이템 레벨 벡터는 reindex_pending_embeddings 가 이미 만들어 두었습니다)
    // 🌟 [CHUNK TYPE GUARD] 비검색 타입은 스키마 필드 조회 전에 구조적으로 차단합니다.
    //    fields.is_empty() 판정만으로는 bias.json 에 우연히 남은 키가 있으면
    //    불필요한 임베딩 배치가 실행됩니다.
    const CHUNK_EXCLUDE_TYPES: [&str; 10] = [
        "pages", "page", "talk", "prompt", "ai_search",
        "question", "answer", "team", "user", "member",
    ];
    if CHUNK_EXCLUDE_TYPES.iter().any(|t| page_type == *t) {
        emit(&format!(
            "  ⏭️ [CHUNK SKIP] type='{}' 은 검색/음차/청크 인덱싱 대상이 아닙니다.",
            page_type
        ));
        return Ok(0);
    }

    // 🌟 [PAGE CACHE GUARD] 페이지 셀렉터 캐시는 page_type 이 도메인 타입(tracking/goods/...)
    //    이므로 위 문자열 목록으로는 절대 걸러지지 않습니다.
    //    실제 사고 사례: 서버 index.ts 의 home 문서
    //      { table:'pages', type:'tracking', data:{ origin, link, item:true, node:true } }
    //    가 items 로 새어 들어와 청크 11건 + 음차 3건이 생성되었습니다.
    //    셀렉터 캐시는 #global-search 의 검색 대상이 아니므로 구조 마커로 즉시 차단합니다.
    {
        let is_page_cache = item_json.get("table")
                .and_then(|v| v.as_str())
                .map_or(false, |t| t == "pages" || t == "page")
            || item_json.get("node").is_some()
            || item_json.get("item").is_some();
        if is_page_cache {
            emit(&format!(
                "  ⏭️ [CHUNK SKIP / PAGE CACHE] item_id='{}' (type='{}') 는 페이지 셀렉터 캐시이므로 청크 인덱싱과 음차를 모두 생략합니다.",
                item_id, page_type
            ));
            return Ok(0);
        }
    }
    if fields.is_empty() {
        emit(&format!(
            "  ⏭️ [CHUNK SKIP] type='{}' 에 대응하는 스키마 필드가 없어 청크 인덱싱을 건너뜁니다. (아이템 벡터는 별도 생성됨)",
            page_type
        ));
        return Ok(0);
    }

    // 🌟 [DISCOVERY-GATED BANK] 필드 뱅크 임베딩을 '실제로 쓰일 때만' 만듭니다.
    //
    //  ── 근거 ──
    //   run_phase_b_pipeline 은 confirmed = true 청크의 property 를 절대
    //   덮어쓰지 않습니다(CONFIRMED PROTECT). 따라서 발견 모드 청크가 0개면
    //   필드 뱅크는 CONFIRM FLAG 진단 로그를 찍는 것 외에 결과에 기여가 없습니다.
    //
    //  ── 실측 (log.txt) ──
    //   "[PLINKO MODE] 확인 모드: 8개 청크 | 발견 모드: 0개 청크" 가
    //   아이템 20건 전부에서 동일하게 나왔습니다.
    //   그런데 그때마다 shipping_doc 필드 뱅크(중복 포함 약 134개 × 각 수십~수백 구)를
    //   통째로 임베딩했습니다. 이것이 백그라운드 인덱싱이 임베딩 모델을
    //   장시간 붙들고 있던 이유이며, 생성 모델 전환과 VRAM 이 겹친 원인입니다.
    //
    //  ── 무엇을 남기는가 ──
    //   idx_field_names 는 항상 전부 채웁니다. run_phase_b_pipeline 의
    //   SCHEMA CANONICALIZE (id → id,link) 가 이 목록을 진실의 원천으로 쓰기 때문에
    //   비우면 property 정규화가 깨집니다.
    //   비우는 것은 뱅크(phrase_embs)뿐이며, plinko_game_for_indexing 은
    //   is_empty() 를 이미 검사하므로 추가 분기 없이 안전하게 건너뜁니다.
    let needs_discovery_bank = raw_chunks.iter().any(|(_, _, confirmed)| !*confirmed);
    if !needs_discovery_bank {
        emit(&format!(
            "  ⚡ [BANK SKIP] 발견 모드 청크 0개 → 필드 뱅크 {}개 임베딩을 생략합니다. (확정 property 유지)",
            fields.len()
        ));
    }
    let mut idx_field_names: Vec<String> = Vec::new();
    let mut idx_field_phrase_embs: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut idx_field_phrase_weights: Vec<Vec<f32>> = Vec::new();
    let mut idx_field_formats: Vec<String> = Vec::new();
    // 🌟 [DEDUP GUARD] bias_schema 가 같은 필드를 두 번 등록해도 열이 겹치지 않게 합니다.
    //    exclusive_assign_for_indexing 의 position() 은 첫 인덱스만 찾으므로
    //    중복 열은 배정 불가능한 '유령 열' 이 되어 행렬만 부풀립니다.
    let mut seen_field_names: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (fname, _, bias_target, _) in &fields {
        if !seen_field_names.insert(fname.clone()) {
            continue;
        }
        let phrase_embs = if !needs_discovery_bank {
            Vec::new()
        } else {
            let (mut phrases, mut weights_inner) =
                crate::utils::ai_utils::split_bias_phrases_weighted_full(bias_target);
            let bridge_ph = crate::utils::ai_utils::abstract_bridge_field_phrases(fname);
            for p in bridge_ph {
                if phrases.iter().any(|e| e == &p) { continue; }
                phrases.push(p);
                weights_inner.push(1.0);
            }
            let embs = if phrases.is_empty() {
                vec![vec![0.0f32; 384]]
            } else {
                model.get_embedding_batch(phrases.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; phrases.len()])
            };
            idx_field_phrase_weights.push(weights_inner);
            embs
        };
        if !needs_discovery_bank {
            idx_field_phrase_weights.push(Vec::new());
        }

        let fmt_str = {
            let lower = fname.to_lowercase();
            let keys: Vec<String> = lower.split(',').map(|s| s.trim().to_string()).collect();
            let has = |k: &str| keys.iter().any(|x| x == k);

            if keys.iter().any(|k| k.contains("insight") || k.contains("summary") || k.contains("analysis")) {
                "Synthesis".to_string()
            } else if keys.iter().any(|k| k.contains("tracking_number") || k == "barcode" || k == "gtin" || k == "mpn") {
                "TrackingCode".to_string()
            } else if has("id") || has("code") || has("no") || has("index") || has("stock_keeping_unit") {
                "Identifier".to_string()
            } else if keys.iter().any(|k| k.contains("link") || k.contains("url")) {
                "Link".to_string()
            } else if keys.iter().any(|k| k.contains("date") || k.ends_with("_at")) {
                "Date".to_string()
            } else if keys.iter().any(|k| {
                k.ends_with("phone") || k == "tel" || k == "telephone" || k == "mobile"
                    || k == "cellphone" || k == "contact" || k == "number"
            }) {
                "Phone".to_string()
            } else if keys.iter().any(|k| k == "address" || k.ends_with("_address")) {
                "Address".to_string()
            } else if keys.iter().any(|k| {
                k.contains("status") || k.contains("payment_method") || k.contains("payment_origin")
                    || k.contains("condition") || k.contains("currency") || k == "bank" || k == "card"
            }) {
                "Enum".to_string()
            } else if keys.iter().any(|k| {
                k.contains("price") || k.contains("amount") || k.contains("quantity") || k.contains("weight")
                    || k == "width" || k == "height" || k == "length" || k.contains("fee")
                    || k.contains("discount") || k.contains("usage_") || k.contains("threshold")
                    || k.contains("duration")
            }) {
                "Numeric".to_string()
            } else {
                "Text".to_string()
            }
        };

        idx_field_names.push(fname.clone());
        idx_field_phrase_embs.push(phrase_embs);
        // 🌟 weights 는 위 분기에서 이미 push 되었습니다. 여기서 다시 push 하면
        //    names 와 weights 의 인덱스가 어긋나 weighted_max_pool_sim 이
        //    다른 필드의 가중치를 읽게 됩니다.
        idx_field_formats.push(fmt_str);
    }

    let model_for_embed = model.clone();
    let enriched_chunks = crate::nl_convert::run_phase_b_pipeline(
        &raw_chunks,
        doc_lang,
        page_type,
        &idx_field_names,
        &idx_field_phrase_embs,
        &idx_field_phrase_weights,
        &idx_field_formats,
        move |text: String| {
            let m = model_for_embed.clone();
            async move { m.get_embedding(text).await.unwrap_or(vec![0.0; 384]) }
        },
    ).await;

    // 🌟 [SCHEMA WHITELIST] 스키마에 없는 property 는 검색 축이 아닙니다.
    //
    //  ── 실측 사고 ──
    //   [13]✓ property='rel_lc'     | score=0.0000 | text='Its rel lc is 3209045268'
    //   [14]✓ property='started_at' | score=0.0000 | text='Its started at is 2026-08-28T00:00:00'
    //   score=0.0000 은 '필드 뱅크에 그 property 가 아예 없다' 는 뜻입니다.
    //   rel_lc 는 릴레이 내부 인덱스, started_at 은 정규화 파생 축이라
    //   사용자가 검색할 대상이 아닙니다. confirmed 보호를 타고 들어와
    //   의미 없는 벡터 2건이 저장되었습니다.
    //
    //  ── 판정 근거 ──
    //   idx_field_names 는 이 doc_type 의 스키마가 소유한 필드 목록입니다.
    //   그 밖의 property 는 정의상 검색 축이 될 수 없습니다.
    //   어휘 사전이 아니라 '스키마 소속 여부' 라는 구조 사실입니다.
    let indexable_chunks: Vec<(usize, &crate::nl_convert::ChunkMetadata)> = enriched_chunks.iter()
        .enumerate()
        .filter(|(_, c)| {
            if c.property == "unclassified" { return false; }
            let in_schema = idx_field_names.iter().any(|f| {
                f == &c.property
                    || f.split(',').any(|k| k.trim() == c.property.as_str())
            });
            if !in_schema {
                println!(
                    "  🚫 [SCHEMA WHITELIST] property='{}' 는 '{}' 스키마에 없는 축이라 청크 인덱싱에서 제외합니다. (text=\"{}\")",
                    c.property, page_type, c.chunk_text
                );
            }
            in_schema
        })
        .collect();
    if indexable_chunks.is_empty() {
        return Ok(0);
    }

    // =====================================================================
    // 🌟 [CROSSOVER / PHASE SEPARATION] 임베딩을 음차 '앞' 으로 전부 끌어옵니다.
    // ---------------------------------------------------------------------
    //  ── 무엇이 문제였나 ──
    //   기존 순서는
    //     ① chunk 임베딩 → ② 음차(Qwen3.5 2B) → ③ anchor 임베딩
    //     → ④ localized 임베딩 → ⑤ 별칭 임베딩(청크마다)
    //   이었습니다. ②를 사이에 두고 임베딩이 앞뒤로 갈라져 있어
    //   Qwen3.5(약 2GB)와 임베딩이 최소 두 번 교차 상주합니다.
    //
    //  ── 왜 순서를 바꿔도 되는가 ──
    //   anchor_texts / localized_texts 는 chunk_meta 의 property 와
    //   value_part 만으로 만들어지며, 음차 결과에 전혀 의존하지 않습니다.
    //   즉 ②보다 먼저 계산할 수 있는데 뒤에 있었을 뿐입니다.
    //   ⑤만 음차 결과가 필요하므로 그것만 뒤에 남깁니다.
    //
    //  ── 효과 ──
    //   임베딩 페이즈 1회 → 생성 페이즈 1회 → 임베딩 페이즈 1회 로
    //   교차 횟수가 고정되고, 각 페이즈 안에서는 스왑이 발생하지 않습니다.
    // =====================================================================
    let _ = model.enter_embedding_phase("index_item_chunks / pre-compute").await;

    let chunk_texts: Vec<String> = indexable_chunks.iter().map(|(_, c)| c.chunk_text.clone()).collect();

    let mut anchor_texts: Vec<String> = Vec::with_capacity(indexable_chunks.len());
    let mut localized_texts: Vec<String> = Vec::with_capacity(indexable_chunks.len());
    for (_, cm) in indexable_chunks.iter() {
        let a = crate::utils::ai_utils::indexing_anchor_text(doc_lang, page_type, &cm.property);
        let leaf = crate::utils::ai_utils::indexing_leaf_label(doc_lang, page_type, &cm.property);
        let v = cm.value_part.trim();
        let l = if v.is_empty() { leaf.clone() } else { format!("{} {}", leaf, v) };
        anchor_texts.push(a);
        localized_texts.push(l);
    }

    // 🌟 [SINGLE BATCH] 세 종류를 한 번에 넣어 왕복을 3회 → 1회로 줄입니다.
    //
    //  ── 청킹을 여기서 하지 않는 이유 ──
    //   호출부에서 쪼개면 scheduler.rs 의 다른 수십 개 호출부는 보호받지 못합니다.
    //   get_embedding_batch 내부가 adaptive_embed_batch 로 여유에 맞춰 쪼개고,
    //   activation 여유 관측과 캐시 적재까지 함께 수행합니다.
    //   여기서는 '한 번에 넘긴다' 는 사실만 남깁니다.
    //
    //  ── 길이 보존 ──
    //   get_embedding_batch 는 입력과 같은 길이를 반환하는 것이 계약이므로
    //   아래 슬라이싱이 항상 안전합니다.
    let total_n = indexable_chunks.len();
    let mut combined: Vec<String> = Vec::with_capacity(total_n * 3);
    combined.extend(chunk_texts.iter().cloned());
    combined.extend(anchor_texts.iter().cloned());
    combined.extend(localized_texts.iter().cloned());

    let mut combined_embs = model.get_embedding_batch(combined.clone()).await
        .unwrap_or_else(|_| vec![vec![0.0; 384]; combined.len()]);
    if combined_embs.len() < combined.len() {
        combined_embs.resize(combined.len(), vec![0.0; 384]);
    }
    emit(&format!(
        "  🧬 [CROSSOVER] 임베딩 선계산 완료: 텍스트 {}건 | 자유 VRAM {}MB",
        combined.len(), model.get_free_vram_mb()
    ));

    let chunk_embs: Vec<Vec<f32>> = combined_embs[0..total_n].to_vec();
    let anchor_embs: Vec<Vec<f32>> = combined_embs[total_n..total_n * 2].to_vec();
    let localized_embs: Vec<Vec<f32>> = combined_embs[total_n * 2..total_n * 3].to_vec();

    let metas: Vec<&crate::nl_convert::ChunkMetadata> =
        indexable_chunks.iter().map(|(_, c)| *c).collect();

    // 🌟 [ANALYTIC TRANSLIT SKIP] 전처리에서 이미 음차를 완료한 경우 건너뜁니다.
    //    이제 이 호출 시점에는 임베딩이 필요한 계산이 전부 끝나 있으므로,
    //    generate_transliteration_aliases 가 임베딩을 반환시키고
    //    Qwen3.5 를 올려도 두 가중치가 겹치지 않습니다.
    let alias_pairs = if skip_transliteration {
        vec![(String::new(), String::new()); metas.len()]
    } else {
        generate_transliteration_aliases(
            model, &metas, doc_lang, page_type, cancel, app_handle, task_id,
        ).await
    };

    let _ = store.delete_chunks_by_item(item_id).await;

    let mut saved = 0usize;

    for (ei, (ci, chunk_meta)) in indexable_chunks.iter().enumerate() {
        let chunk_id = format!("{}_{}", item_id, ci);

        let (w_chunk, w_anchor, w_local) = match chunk_meta.property_format.as_str() {
            "Text" | "Address" | "Synthesis" => (0.25f32, 0.10f32, 0.65f32),
            _ => (0.40f32, 0.30f32, 0.30f32),
        };

        let mut final_vec = vec![0.0f32; 384];
        for d in 0..384 {
            final_vec[d] = chunk_embs[ei][d] * w_chunk
                + anchor_embs[ei][d] * w_anchor
                + localized_embs[ei][d] * w_local;
        }
        let norm: f32 = final_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for d in 0..384 { final_vec[d] /= norm; }
        }

        let _ = store.upsert_chunk(
            &chunk_id,
            item_id,
            page_type,
            &chunk_meta.chunk_text,
            &chunk_meta.property,
            &chunk_meta.property_format,
            &chunk_meta.value_part,
            Some(final_vec),
            Some(cc),
            Some(bcc),
            Some(ref_val),
            Some(search_mode),
        ).await;
        saved += 1;

        saved += upsert_alias_chunks(
            store, model, item_id, &chunk_id, page_type, doc_lang,
            chunk_meta, &alias_pairs[ei], cc, bcc, ref_val, search_mode,
        ).await;
    }

    emit(&format!(
        "  🧩 [CLOUD-SYNC INDEX] item_id='{}' | 청크 {}건 로컬 인덱싱 완료 (type='{}')",
        item_id, saved, page_type
    ));

    Ok(saved)
}

// =====================================================================
// 🌟 [SINGLE UPSERT v4]
// ---------------------------------------------------------------------
//  v3 까지는 도메인 테이블(sales/tracking/event)과 items 미러 테이블에
//  같은 문서를 두 번 저장했습니다. 그런데 store.rs 의 resolve_table 이
//  v4 부터 두 호출을 모두 items 로 접기 때문에,
//  그대로 두면 '같은 행에 delete → add' 를 두 번 수행하는 낭비가 됩니다.
//
//  또한 두 번째 호출의 digest 가 첫 번째와 다르면
//  upsert_item 의 스킵 가드가 매번 통과되어 무한 재쓰기가 발생할 수 있습니다.
//
//  → 저장 지점을 이 헬퍼 하나로 모읍니다.
//    호출부는 target_table 을 계속 넘겨도 되지만(가독성 유지),
//    실제 물리 저장은 정확히 1회만 일어납니다.
pub async fn save_item(
    store: &VectorStore,
    table_hint: &str,
    id: &str,
    type_: &str,
    data: Value,
    vector: Option<Vec<f32>>,
    from: &str,
    to: &str,
    cc: &str,
    bcc: &str,
    ref_val: &str,
    digest: Option<&str>,
) {
    // 🌟 [TABLE HINT PRIORITY] hint 가 물리 테이블을 명시하면 그것을 우선합니다.
    //    기존에는 hint 를 버리고 type_ 로만 라우팅했기 때문에,
    //    pages 를 저장하려면 type_ 에 "pages" 를 넣어야 했고
    //    그 값이 upsert_item 안에서 data.type 을 덮어써 도메인 타입(goods/order)을 파괴했습니다.
    //    hint 로 테이블을 정하면 type_ 에 실제 도메인 타입을 그대로 넘길 수 있습니다.
    let table = match table_hint {
        "pages" | "page" => "pages",
        "users" | "member" | "team" | "user" => "users",
        _ => match type_ {
            "member" | "team" | "user" => "users",
            "pages" | "page" => "pages",
            _ => "items",
        },
    };

    let _ = store.upsert_item(
        table, id, type_, data, vector,
        None, // 🌟 scheduler 경로는 텍스트 전용이라 비전 벡터 없음
        Some(from), Some(to), Some(cc), Some(bcc), Some(ref_val), digest
    ).await;
}
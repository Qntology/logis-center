use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use serde_json::json;
use crate::model::LogisModel;
use crate::scheduler::TRANSLIT_MEM_CACHE;
use tauri::Emitter;


fn translit_cache_key(word: &str, lang: &str) -> String {
    format!("{}\u{1}{}", lang.trim().to_lowercase(), word.trim())
}

// =====================================================================
// 🌟 [SYNONYM EXPANSION] 청크 값의 2-pass 음차 별칭 생성 / 저장
// ---------------------------------------------------------------------
// 흐름:
//   원문 "Cable Knit Cardigan"
//     → 1차: 문서 언어 표기로 음차   "케이블 니트 카디건"   (transliteration_native)
//     → 2차: 원문 표기로 역음차      "keibeul nit kadigeon" (transliteration_roman)
//   두 별칭을 동일 item_id / 동일 property 로 item_chunks 에 추가 저장합니다.
//   store.rs 의 search_chunks() 가 item_id 기준으로 점수를 합산하므로,
//   별칭 하나만 매칭돼도 원본 item 이 그대로 상위 랭크됩니다.
//
// 언어 하드코딩이 없는 이유:
//   1차 목표 표기 = native_script_sample()  → detect_document_language 결과 + bias.json
//   2차 목표 표기 = 원문 값 그 자체          → 언어 테이블 자체가 불필요
// =====================================================================

// =====================================================================
// 🌟 [TRANSLIT CACHE HELPER] Dexie 캐시 조회 / 저장 (프론트 경유)
// =====================================================================

/// 음차 캐시를 조회합니다. ① 프로세스 전역 메모리 → ② 프론트엔드 Dexie 순서입니다.
///
/// 반환값 계약:
///   Some(("네이티브", "로마자")) → 캐시 히트
///   Some(("", ""))              → 네거티브 캐시 히트 ('음차 불가' 로 이미 확정된 값)
///   None                        → 캐시 미스 (레코드 자체가 없음 / 통신 실패)
///
/// ⚠️ 호출부는 Some 이면 값이 비어 있어도 '히트' 로 취급해야 합니다.
///    기존 구현은 빈 값을 미스로 보고 매번 LLM 을 다시 불렀습니다.
async fn query_translit_cache(
    app_handle: &tauri::AppHandle,
    word: &str,
    lang: &str,
) -> Option<(String, String)> {
    let key = translit_cache_key(word, lang);

    // ── ① 프로세스 전역 메모리 캐시 ──
    if let Ok(map) = TRANSLIT_MEM_CACHE.lock() {
        if let Some(hit) = map.get(&key) {
            println!(
                "  💾 [TRANSLIT CACHE / MEM HIT] '{}' (lang='{}') → native='{}' | roman='{}'",
                word, lang, hit.0, hit.1
            );
            return Some(hit.clone());
        }
    }

    // ── ② 프론트엔드 Dexie 영구 캐시 ──
    let request_id = uuid::Uuid::new_v4().to_string();
    let (tx, rx) = tokio::sync::oneshot::channel::<Vec<(String, String)>>();

    {
        let mut map = crate::scheduler::TRANSLIT_PENDING.lock().unwrap();
        map.insert(request_id.clone(), tx);
    }

    let _ = app_handle.emit("translit-cache-query", json!({
        "request_id": request_id,
        "word": word,
        "lang": lang
    }));

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        rx
    ).await;

    // 어떤 경로로 끝나든 pending 엔트리는 반드시 회수합니다. (누수 방지)
    let _ = crate::scheduler::TRANSLIT_PENDING.lock().unwrap().remove(&request_id);

    match result {
        Ok(Ok(candidates)) => {
            if candidates.is_empty() {
                println!(
                    "  🔍 [TRANSLIT CACHE / MISS] '{}' (lang='{}') — Dexie 에 레코드가 없습니다.",
                    word, lang
                );
                None
            } else {
                let hit = candidates[0].clone();
                if let Ok(mut map) = TRANSLIT_MEM_CACHE.lock() {
                    map.insert(key, hit.clone());
                }
                println!(
                    "  💾 [TRANSLIT CACHE / DEXIE HIT] '{}' (lang='{}') → native='{}' | roman='{}'",
                    word, lang, hit.0, hit.1
                );
                Some(hit)
            }
        },
        Ok(Err(_)) => {
            println!(
                "  ⚠️ [TRANSLIT CACHE] '{}' (lang='{}') 응답 채널이 닫혔습니다. 캐시 미스로 처리합니다.",
                word, lang
            );
            None
        },
        Err(_) => {
            println!(
                "  ⚠️ [TRANSLIT CACHE] '{}' (lang='{}') 프론트엔드 응답 5초 타임아웃. 캐시 미스로 처리합니다.",
                word, lang
            );
            None
        }
    }
}

/// 음차 결과를 캐시에 저장합니다.
/// ① 프로세스 전역 메모리에 즉시 반영 ② 프론트엔드 Dexie 에 영구 저장 요청(fire-and-forget)
///
/// native / roman 이 모두 빈 문자열이면 '음차 불가' 라는 판정 자체를 저장합니다(네거티브 캐시).
/// 이 값이 없으면 다음 태스크에서 같은 판정을 위해 LLM 을 또 호출하게 됩니다.
fn save_translit_cache(
    app_handle: &tauri::AppHandle,
    word: &str,
    lang: &str,
    native: &str,
    roman: &str,
) {
    let key = translit_cache_key(word, lang);
    if let Ok(mut map) = TRANSLIT_MEM_CACHE.lock() {
        map.insert(key, (native.to_string(), roman.to_string()));
    }

    if native.trim().is_empty() && roman.trim().is_empty() {
        println!(
            "  💾 [TRANSLIT CACHE / SAVE-NEGATIVE] '{}' (lang='{}') — 음차 불가 판정을 영구 저장합니다.",
            word, lang
        );
    } else {
        println!(
            "  💾 [TRANSLIT CACHE / SAVE] '{}' (lang='{}') → native='{}' | roman='{}'",
            word, lang, native, roman
        );
    }

    let _ = app_handle.emit("translit-cache-save", json!({
        "word": word,
        "lang": lang,
        "native": native,
        "roman": roman
    }));
}

/// 🌟 [CROSS-LANGUAGE TRANSLITERATION] 전처리 단계에서 사용하는 교차 언어 음차.
///    방향: 영어 단어 → 문서 언어(한글/일어/중어 등)
///    한글→한글 같은 동일 언어 음차는 수행하지 않습니다.
///    한글→영어(로마자) 역방향도 함께 생성합니다.
///
///    이 함수는 `run_analytic_structuring` 에서 호출되며,
///    Qwen3.5 가 이미 로드되어 있어야 합니다.
pub async fn transliterate_cross_language(
    model: &LogisModel,
    text: &str,
    doc_lang: &str,
    cancel: &Arc<AtomicBool>,
    app_handle: &tauri::AppHandle,
    task_id: &str,
) -> (String, String) {
    let _ = (app_handle, task_id);
    let src = text.trim().to_string();
    if src.is_empty() { return (String::new(), String::new()); }

    let src_is_latin = crate::nl_convert::is_latin_dominant(&src);
    let sample = crate::nl_convert::native_script_sample(doc_lang, "", "");
    let target_is_latin = crate::nl_convert::is_latin_dominant(&sample);

    // 동일 문자 체계 → 음차 불필요
    if src_is_latin == target_is_latin && !src_is_latin {
        // 한글→한글: 무의미. 로마자 역방향만 시도.
        if let Some(roman) = crate::nl_convert::try_any_ascii_transliteration(&src) {
            return (String::new(), roman);
        }
        return (String::new(), String::new());
    }

    // 🌟 [PROMPT / SANITIZE FIX]
    //  ── 무엇이 문제였나 ──
    //   ① crate::prompts::transliteration_prompt(&src, doc_lang) 는 ISO 코드("ko")를
    //      [TARGET LANGUAGE] 에 그대로 꽂아, 모델이 목표 표기 체계를 인식하지 못했습니다.
    //      (nl_convert::build_transliteration_prompt 는 lang_code_to_full_name 으로 "korean" 을 넣습니다)
    //   ② transliteration 객체에서 .values().next() 로 '아무 항목이나 하나' 를 꺼내
    //      다단어 문장의 첫 단어도 아닌 임의 값이 native 로 저장되었습니다.
    //      (로그 실측: "상품3" → '산마두', "사용자" → '수용자')
    //   ③ G1(원문 동일) / G2(표기 체계 반전) / G3(길이 상한) 게이트를 통과시키지 않아
    //      명백한 환각도 그대로 별칭 벡터가 되었습니다.
    //  ── 해결 ──
    //   nl_convert 가 이미 갖고 있는 프롬프트 빌더와 정화기를 그대로 재사용합니다.

    // 영어 원문 → 문서 언어(비라틴) 방향
    if src_is_latin && !target_is_latin {
        let prompt = crate::nl_convert::build_transliteration_prompt(&src, doc_lang);
        let res = model
            .call_qwen3_5_transliteration(&prompt, Some(cancel.clone()))
            .await
            .unwrap_or_default();
        let (_t, native) = crate::nl_convert::sanitize_transliteration_dual(&res, &src);
        let roman = crate::nl_convert::try_any_ascii_transliteration(&src).unwrap_or_default();
        if native.is_empty() {
            println!("[ANALYTIC] ⚪ [TRANSLIT REJECT] '{}' 의 문서언어 음차가 게이트를 통과하지 못해 폐기했습니다.", src);
        }
        return (native, roman);
    }

    // 비라틴 원문 → 로마자(라틴) 역방향
    if !src_is_latin && target_is_latin {
        if let Some(roman) = crate::nl_convert::try_any_ascii_transliteration(&src) {
            return (String::new(), roman);
        }
        let prompt = crate::nl_convert::build_transliteration_prompt(&src, "en");
        let res = model
            .call_qwen3_5_transliteration(&prompt, Some(cancel.clone()))
            .await
            .unwrap_or_default();
        let (_t, roman) = crate::nl_convert::sanitize_transliteration_dual(&res, &src);
        if roman.is_empty() {
            println!("[ANALYTIC] ⚪ [TRANSLIT REJECT] '{}' 의 로마자 음차가 게이트를 통과하지 못해 폐기했습니다.", src);
        }
        return (String::new(), roman);
    }

    (String::new(), String::new())
}
/// 🌟 [SYNONYM EXPANSION] 청크 배열에 대해 2-pass 음차 별칭을 생성합니다.
/// 반환값은 입력 청크와 같은 길이의 (native, roman) 배열입니다.
///
/// 동일 값(value_part)은 캐시로 재사용하므로 LLM 호출이 값의 종류 수만큼만 발생합니다.
/// 🌟 [DEXIE CACHE] 생성 전에 Dexie 캐시를 먼저 조회하고,
///    캐시 히트 시 Qwen3.5 호출을 완전히 생략합니다.
pub async fn generate_transliteration_aliases(
    model: &LogisModel,
    chunks: &[&crate::nl_convert::ChunkMetadata],
    doc_lang: &str,
    page_type: &str,
    cancel: &Arc<AtomicBool>,
    app_handle: &tauri::AppHandle,
    task_id: &str,
) -> Vec<(String, String)> {
    let emit = |msg: &str| {
        println!("{}", msg);
        let _ = app_handle.emit("task-console-log", json!({"task_id": task_id, "text": format!("{}
", msg)}));
    };

    // 🌟 [TRANSLIT TYPE GUARD] 비검색 타입은 음차 생성 자체가 무의미합니다.
    //    pages/talk/prompt 는 셀렉터 캐시·채팅 말풍선이라 값 음차가 필요 없습니다.
    //    team/user/member 는 통계 문서라 음차 대상이 아닙니다.
    const TRANSLIT_EXCLUDE_TYPES: [&str; 10] = [
        "pages", "page", "talk", "prompt", "ai_search",
        "question", "answer", "team", "user", "member",
    ];
    if TRANSLIT_EXCLUDE_TYPES.iter().any(|t| page_type == *t) {
        return vec![(String::new(), String::new()); chunks.len()];
    }

    let mut out: Vec<(String, String)> = vec![(String::new(), String::new()); chunks.len()];
    let mut cache: std::collections::HashMap<String, (String, String)> = std::collections::HashMap::new();
    let mut made = 0usize;
    let mut reused = 0usize;
    let mut skipped = 0usize;

    for (i, cm) in chunks.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }

        if !crate::nl_convert::needs_transliteration(cm) {
            skipped += 1;
            continue;
        }

        let src = cm.value_part.trim().to_string();
        if src.is_empty() {
            skipped += 1;
            continue;
        }

        // 🌟 [CACHE LOOKUP] ① 아이템 로컬 → ② 프로세스 전역 메모리 → ③ Dexie 영구
        if let Some(hit) = cache.get(&src) {
            out[i] = hit.clone();
            reused += 1;
            continue;
        }

        if let Some(dexie_hit) = query_translit_cache(app_handle, &src, doc_lang).await {
            // 🌟 [NEGATIVE CACHE] 빈 값도 '음차 불가로 이미 확정된 사실' 이므로 히트로 인정합니다.
            //    기존 구현은 빈 값을 미스로 보고 Qwen3.5 를 매번 다시 호출했습니다.
            let is_negative = dexie_hit.0.trim().is_empty() && dexie_hit.1.trim().is_empty();
            cache.insert(src.clone(), dexie_hit.clone());
            out[i] = dexie_hit;
            if is_negative {
                skipped += 1;
                println!(
                    "  ⚪ [TRANSLIT CACHE / NEGATIVE HIT] '{}' 는 이전에 '음차 불가' 로 확정된 값입니다. LLM 을 호출하지 않습니다.",
                    src
                );
            } else {
                reused += 1;
                println!("  💾 [DEXIE CACHE HIT] '{}' (Qwen3.5 생략)", src);
            }
            continue;
        }

        // 1차 음차 가능 여부 판정.
        // 원문과 같은 표기 체계로만 변환 가능한 환경이면 스킵합니다.
        if !crate::nl_convert::can_transliterate(&src, doc_lang) {
            cache.insert(src.clone(), (String::new(), String::new()));
            save_translit_cache(app_handle, &src, doc_lang, "", "");
            skipped += 1;
            continue;
        }
        // 🌟 [SAME-SCRIPT BLOCK] 원문과 대상이 같은 문자 체계면 음차가 성립하지 않습니다.
        //    한글 문서를 한글로 음차하는 것은 오음차(수용자←사용자)만 양산합니다.
        //    이 경우 영어 단어가 포함되어 있으면 영어→한글 방향으로 전환하고,
        //    순수 한글이면 음차 자체를 스킵합니다.
        let src_is_latin = crate::nl_convert::is_latin_dominant(&src);
        let target_is_latin = crate::nl_convert::is_latin_dominant(
            &crate::nl_convert::native_script_sample(doc_lang, "", "")
        );
        if src_is_latin == target_is_latin {
            if src_is_latin && !doc_lang.is_empty() && doc_lang != "en" {
                // 영어 원문 → 문서 언어(비라틴) 방향: 계속 진행
            } else {
                cache.insert(src.clone(), (String::new(), String::new()));
                save_translit_cache(app_handle, &src, doc_lang, "", "");
                skipped += 1;
                continue;
            }
        }

        println!(" 🔄 [SYNONYM PASS-1] '{}' (property='{}')", src, cm.property);
        println!("    SOURCE = '{}'", src);

        // 🌟 [LANGUAGE TRACK SPLIT] 표기 체계별로 단어를 분리하여 트랙별 처리합니다.
        // 비라틴 단어(한글 등) → target "english" (로마자 전사)
        // 라틴 단어(영어 등)   → target doc_lang (문서 언어 스크립트 전사)
        let (non_latin_words, latin_words) = crate::nl_convert::split_words_by_script(&src);
        let is_mixed = !non_latin_words.is_empty() && !latin_words.is_empty();

        // 🌟 [CROSS-LANG DIRECTION]
        //    기존: 비라틴 원문 → 로마자(라틴) + 라틴 원문 → 문서언어(비라틴)
        //    변경: 영어 단어 → 문서언어(비라틴) 만 수행.
        //           비라틴 원문(한글) → 한글 음차는 무의미하므로 스킵.
        //           한글 → 로마자(라틴) 는 유지 (검색 역방향 리콜용).
        let doc_lang_is_latin = crate::nl_convert::is_latin_dominant(
            &crate::nl_convert::native_script_sample(doc_lang, "", "")
        );
        let skip_native_translit = !src_is_latin && !doc_lang_is_latin;
        //    한글→한글 음차는 스킵하되, 한글→로마자(역방향)는 유지.
        //    영어→한글 은 정상 수행.

        let s1_transliteration = if skip_native_translit && !is_mixed {
            // 🌟 동일 언어 음차 스킵: 한글→한글 음차는 무의미.
            //    로마자(역방향)만 생성합니다.
            println!("    ⚪ [SAME-SCRIPT SKIP] '{}' → '{}' 동일 문자 체계 음차 스킵 (로마자 역방향만 생성)",
                src, doc_lang);
            String::new()
        } else if is_mixed {
            println!("    [TRACK SPLIT] 비라틴: {:?} | 라틴: {:?}", non_latin_words, latin_words);
            // Track A: 비라틴 단어 → 로마자 전사 (target: english)
            // 🌟 [ANY_ASCII FIRST] 비라틴→라틴 방향은 any_ascii로 처리 가능하면 LLM 생략
            let mut track_a_transliteration = String::new();
            if !non_latin_words.is_empty() {
                // 🌟 [ANY_ASCII FIRST] 단어 단위로 any_ascii 시도.
                //    전체 조인이 실패해도 단어별 시도가 성공할 수 있으므로 양쪽 모두 시도합니다.
                let joined_non_latin = non_latin_words.join(" ");
                if let Some(ascii_result) = crate::nl_convert::try_any_ascii_transliteration(&joined_non_latin) {
                    track_a_transliteration = ascii_result;
                    println!("    TRACK-A METHOD = any_ascii full (LLM skipped)");
                    println!("    TRACK-A RESULT = '{}'", track_a_transliteration);
                } else if let Some(ascii_words_result) = crate::nl_convert::try_any_ascii_transliteration_words(&non_latin_words) {
                    track_a_transliteration = ascii_words_result;
                    println!("    TRACK-A METHOD = any_ascii per-word (LLM skipped)");
                    println!("    TRACK-A RESULT = '{}'", track_a_transliteration);
                } else {
                    let p_a = crate::nl_convert::build_transliteration_prompt_for_words(&non_latin_words, "english");
                    let raw_a = model
                        .call_qwen3_5_transliteration(&p_a, Some(cancel.clone()))
                        .await
                        .unwrap_or_default();
                    println!("    TRACK-A RAW (non-latin→latin) = '{}'", raw_a.replace('\n', "\n"));
                    let (_t_a, tr_a) = crate::nl_convert::sanitize_transliteration_dual_for_words(&raw_a, &non_latin_words);
                    track_a_transliteration = tr_a;
                }
            }
            // Track B: 라틴 단어 → 문서 언어 스크립트 전사 (target: doc_lang)
            let mut track_b_transliteration = String::new();
            if !latin_words.is_empty() {
                let p_b = crate::nl_convert::build_transliteration_prompt_for_words(&latin_words, doc_lang);
                let raw_b = model
                    .call_qwen3_5_transliteration(&p_b, Some(cancel.clone()))
                    .await
                    .unwrap_or_default();
                println!("    TRACK-B RAW (latin→{}) = '{}'", doc_lang, raw_b.replace('\n', "\n"));
                let (_t_b, tr_b) = crate::nl_convert::sanitize_transliteration_dual_for_words(&raw_b, &latin_words);
                track_b_transliteration = tr_b;
            }
            println!("    TRACK-A TRANSLITERATION= '{}'", track_a_transliteration);
            println!("    TRACK-B TRANSLITERATION= '{}'", track_b_transliteration);
            // 🌟 [TRACK-B LATIN RESIDUE RETRY]
            //    Track B 는 라틴 → 문서 언어(비라틴) 음차입니다.
            //    결과가 여전히 라틴 문자를 포함하면 음차 실패입니다.
            //    (로그 실측: "RITMO" → " ritmo" → trim 후 "ritmo" = 라틴 잔존)
            //    실패 단어만 추출하여 Qwen3.5 2B 로 1회 재음차합니다.
            //    재음차도 라틴이면 원본 라틴 단어를 그대로 유지합니다.
            if !track_b_transliteration.is_empty() && !latin_words.is_empty() {
                let track_b_words: Vec<String> = track_b_transliteration
                    .split_whitespace()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                let mut failed_indices: Vec<usize> = Vec::new();
                let mut failed_originals: Vec<String> = Vec::new();
                for (i, w) in track_b_words.iter().enumerate() {
                    if crate::nl_convert::is_latin_dominant(w) {
                        // 비라틴이어야 할 단어가 라틴 → 음차 실패
                        if i < latin_words.len() {
                            failed_indices.push(i);
                            failed_originals.push(latin_words[i].clone());
                        }
                    }
                }
                if !failed_originals.is_empty() {
                    println!("    🔧 [TRACK-B LATIN RESIDUE] 음차 실패(라틴 잔존) 단어 {:?} 발견 → 재음차 수행", failed_originals);
                    let p_retry = crate::nl_convert::build_transliteration_prompt_for_words(&failed_originals, doc_lang);
                    let raw_retry = model
                        .call_qwen3_5_transliteration(&p_retry, Some(cancel.clone()))
                        .await
                        .unwrap_or_default();
                    println!("    TRACK-B RETRY RAW = '{}'", raw_retry.replace('\n', "\n"));
                    let (_t_retry, tr_retry) = crate::nl_convert::sanitize_transliteration_dual_for_words(&raw_retry, &failed_originals);
                    let retry_words: Vec<String> = tr_retry
                        .split_whitespace()
                        .map(|s| s.to_string())
                        .collect();
                    let mut track_b_parts: Vec<String> = track_b_words.clone();
                    for (fi, &orig_idx) in failed_indices.iter().enumerate() {
                        if let Some(new_w) = retry_words.get(fi) {
                            if !new_w.is_empty() && !crate::nl_convert::is_latin_dominant(new_w) {
                                println!("    🔧 [TRACK-B RETRY FIX] '{}' → '{}'", latin_words[orig_idx], new_w);
                                track_b_parts[orig_idx] = new_w.clone();
                            } else {
                                println!("    ⚠️ [TRACK-B RETRY SKIP] '{}' 재음차 결과 '{}' 도 라틴이라 원본 유지", latin_words[orig_idx], new_w);
                                track_b_parts[orig_idx] = latin_words[orig_idx].clone();
                            }
                        } else {
                            println!("    ⚠️ [TRACK-B RETRY MISS] '{}' 재음차 결과 매핑 실패. 원본 유지", latin_words[orig_idx]);
                            track_b_parts[orig_idx] = latin_words[orig_idx].clone();
                        }
                    }
                    track_b_transliteration = track_b_parts.join(" ");
                    println!("    TRACK-B TRANSLITERATION (after retry)= '{}'", track_b_transliteration);
                }
            }
            // 🌟 [LANGUAGE-CONSISTENT MERGE] 원본 단어 순서 병합(혼용) 대신
            //    언어별 통일 문자열을 생성합니다.
            //    native(비라틴 통일) = 원본 비라틴 단어 + 라틴 단어의 문서 언어 음차
            //    roman(라틴 통일)   = 비라틴 단어의 로마자 음차 + 원본 라틴 단어
            //    Qwen3.5 가 일부 단어를 잘못 음차해도 언어 그룹 자체는 유지됩니다.
            let mut korean_unified_parts: Vec<String> = Vec::new();
            for w in &non_latin_words {
                korean_unified_parts.push(w.clone());
            }
            if !track_b_transliteration.is_empty() {
                korean_unified_parts.push(track_b_transliteration.clone());
            }
            let korean_unified = korean_unified_parts.join(" ");

            let mut english_unified_parts: Vec<String> = Vec::new();
            if !track_a_transliteration.is_empty() {
                english_unified_parts.push(track_a_transliteration.clone());
            }
            for w in &latin_words {
                english_unified_parts.push(w.clone());
            }
            let english_unified = english_unified_parts.join(" ");

            println!("    [LANG-UNIFIED] native(ko) = '{}'", korean_unified);
            println!("    [LANG-UNIFIED] roman(en) = '{}'", english_unified);

            // 🌟 혼용 모드에서는 언어 통일 문자열 2개를 직접 반환합니다.
            //    이후 PASS-2 를 건너뛰고 assign_transliterations 에 바로 전달합니다.
            //    반환 형식: "native|||roman" 구분자로 임시 인코딩
            format!("{}|||{}", korean_unified, english_unified)
        } else {
            // 단일 스크립트: 기존 로직 그대로
            let p1 = crate::nl_convert::build_transliteration_prompt(&src, doc_lang);
            let raw1 = model
                .call_qwen3_5_transliteration(&p1, Some(cancel.clone()))
                .await
                .unwrap_or_default();
            println!("    PASS-1 RAW   = '{}'", raw1.replace('\n', "\n"));
            let (_t, tr) = crate::nl_convert::sanitize_transliteration_dual(&raw1, &src);
            tr
        };

        // 🌟 [MIXED SCRIPT RE-TRANSLITERATION]
        //    PASS-1 결과에서 한글+라틴 혼용 단어(예: "시IELD")가 발견되면
        //    해당 단어만 재음차하여 순수 목표 스크립트로 교정합니다.
        //    (Qwen3.5 가 간헐적으로 일부 문자만 변환하고 나머지를 원문 그대로 남기는 문제 대응)
        //    🌟 [MIXED MODE GUARD] 혼용 모드에서는 "|||" 구분자가 포함된 언어 통일 문자열이므로
        //    mixed-script 감지 및 재음차 로직을 건너뜁니다.
        let mut s1 = s1_transliteration.clone();
        if !s1_transliteration.contains("|||") {
            let mixed_words = crate::nl_convert::find_mixed_script_words(&s1);
            if !mixed_words.is_empty() {
                println!("    🔧 [MIXED SCRIPT DETECTED] 혼용 단어 {:?} 발견 → 재음차 수행", mixed_words);
                // 혼용 단어의 '원문 형태'를 역추적합니다.
                // 혼용 단어는 PASS-1 프롬프트에 넣었던 원문 단어에서 파생되었으므로,
                // 원문 단어 목록에서 해당 혼용 단어를 만든 원문을 찾습니다.
                // 판정: 혼용 단어의 라틴 부분과 원문 단어가 포함 관계이면 매칭.
                let mut retranslate_pairs: Vec<(String, String)> = Vec::new(); // (원문단어, 혼용단어)
                let src_words: Vec<&str> = src.split_whitespace().collect();
                for mw in &mixed_words {
                    // 혼용 단어에서 라틴 부분만 추출하여 원문과 매칭
                    let latin_part: String = mw.chars().filter(|c| c.is_ascii_alphabetic()).collect();
                    let mut matched_src = mw.clone(); // 폴백: 혼용 단어 자체
                    for sw in &src_words {
                        let sw_lower = sw.to_lowercase();
                        let latin_lower = latin_part.to_lowercase();
                        if !latin_lower.is_empty() && sw_lower.contains(&latin_lower) {
                            matched_src = sw.to_string();
                            break;
                        }
                    }
                    retranslate_pairs.push((matched_src, mw.clone()));
                }
                if !retranslate_pairs.is_empty() {
                    let retranslate_words: Vec<String> = retranslate_pairs.iter().map(|(s, _)| s.clone()).collect();
                    println!("    🔧 [MIXED RE-TRANSLATE] 원문 단어 {:?} 재음차 요청", retranslate_words);
                    let p_re = crate::nl_convert::build_transliteration_prompt_for_words(&retranslate_words, doc_lang);
                    let raw_re = model
                        .call_qwen3_5_transliteration(&p_re, Some(cancel.clone()))
                        .await
                        .unwrap_or_default();
                    println!("    🔧 [MIXED RE-TRANSLATE RAW] = '{}'", raw_re.replace('\n', "\n"));
                    let (_t_re, tr_re) = crate::nl_convert::sanitize_transliteration_dual_for_words(&raw_re, &retranslate_words);
                    if !tr_re.is_empty() {
                        // 재음차 결과를 단어별로 매핑하여 혼용 단어 교체
                        let re_results: Vec<&str> = tr_re.split_whitespace().collect();
                        let mut replacements: Vec<(String, String)> = Vec::new();
                        for (i, (_src_w, mixed_w)) in retranslate_pairs.iter().enumerate() {
                            if let Some(new_w) = re_results.get(i) {
                                let new_word = new_w.to_string();
                                // 재음차 결과도 여전히 혼용이면 폐기
                                let still_mixed = crate::nl_convert::find_mixed_script_words(&new_word);
                                if still_mixed.is_empty() && !new_word.is_empty() {
                                    replacements.push((mixed_w.clone(), new_word));
                                } else {
                                    println!("    ⚠️ [MIXED RE-TRANSLATE SKIP] '{}' 재음차 결과 '{}' 도 혼용이라 폐기", mixed_w, new_word);
                                }
                            }
                        }
                        if !replacements.is_empty() {
                            println!("    🔧 [MIXED SCRIPT FIXED] 교체: {:?}", replacements);
                            s1 = crate::nl_convert::replace_mixed_words(&s1, &replacements);
                        }
                    }
                }
            } else {
                // 혼용 모드: "|||" 구분자 포함 문자열은 mixed-script 교정 대상 아님
                println!("    [MIXED MODE] 언어 통일 문자열이므로 mixed-script 교정 생략");
            }
        }

        // 🌟 [MIXED MODE FAST PATH] 혼용 모드에서는 언어 통일 문자열이 이미 생성되어 있으므로
        //    PASS-2 를 건너뛰고 직접 pair 를 조립합니다.
        let pair: (String, String) = if s1_transliteration.contains("|||") {
            let mut parts = s1_transliteration.splitn(2, "|||");
            let native_candidate = parts.next().unwrap_or("").trim().to_string();
            let roman_candidate = parts.next().unwrap_or("").trim().to_string();
            println!("    PASS-1 RESULT (mixed native) = '{}'", native_candidate);
            println!("    PASS-1 RESULT (mixed roman)  = '{}'", roman_candidate);
            println!("    PASS-2 SKIPPED (혼용 모드: 언어 통일 별칭이 이미 양방향으로 생성됨)");
            // 🌟 [ANY_ASCII FAST PATH for roman] roman 후보가 비어있고 native 가 비라틴이면
            //    any_ascii 로 로마자 변환을 시도합니다.
            let mut final_roman = roman_candidate.clone();
            if final_roman.is_empty() && !native_candidate.is_empty() && !crate::nl_convert::is_latin_dominant(&native_candidate) {
                if let Some(ascii_result) = crate::nl_convert::try_any_ascii_transliteration(&native_candidate) {
                    final_roman = ascii_result;
                    println!("    PASS-2 METHOD = any_ascii (LLM skipped)");
                    println!("    PASS-2 RESULT = '{}'", final_roman);
                }
            }
            // 원문과 완전히 동일하면 폐기
            let native_final = if !native_candidate.is_empty() && !native_candidate.eq_ignore_ascii_case(&src) {
                native_candidate
            } else {
                String::new()
            };
            let roman_final = if !final_roman.is_empty() && !final_roman.eq_ignore_ascii_case(&src) {
                final_roman
            } else {
                String::new()
            };
            (native_final, roman_final)
        } else {
            // ── 단일 스크립트 경로 (기존 로직 유지) ──
            println!("    PASS-1 TRANSLITERATION  = '{}'", s1_transliteration);
            println!("    PASS-1 RESULT= '{}'", s1);
            // 2차: 1차 결과를 '원문 표기 체계'로 되돌립니다.
            //      🌟 [ANY_ASCII FAST PATH] 비라틴 → 라틴 방향이면 any_ascii 를 먼저 시도합니다.
            //      any_ascii 가 성공하면 LLM 호출 없이 즉시 확정합니다.
            let mut s2 = String::new();
            if !s1.is_empty() {
                // 1차 결과와 원문의 표기 체계가 달라야 2차 역음차가 성립합니다.
                if crate::nl_convert::is_latin_dominant(&s1) != crate::nl_convert::is_latin_dominant(&src) {
                    // 🌟 [REVERSE TARGET] 2차는 1차 결과를 '원문의 표기 체계'로 되돌립니다.
                    //    원문이 라틴이면 → 1차 결과가 비라틴(문서 언어) → 2차 타겟은 "english"
                    //    원문이 비라틴이면 → 1차 결과가 라틴(로마자) → 2차 타겟은 doc_lang
                    let second_target = if crate::nl_convert::is_latin_dominant(&src) {
                        "english"
                    } else {
                        doc_lang
                    };
                    // 🌟 [ANY_ASCII GATE] 비라틴 → 라틴 방향이면 any_ascii 우선 시도
                    if crate::nl_convert::is_latin_dominant(&src) && !crate::nl_convert::is_latin_dominant(&s1) {
                        if let Some(ascii_result) = crate::nl_convert::try_any_ascii_transliteration(&s1) {
                            s2 = ascii_result;
                            println!("    PASS-2 SOURCE = '{}'", s1);
                            println!("    PASS-2 METHOD = any_ascii (LLM skipped)");
                            println!("    PASS-2 RESULT = '{}'", s2);
                        }
                    }
                    // any_ascii 가 실패하거나 해당 방향이 아니면 LLM 폴백
                    if s2.is_empty() {
                        let p2 = crate::nl_convert::build_transliteration_prompt(&s1, second_target);
                        let raw2 = model
                            .call_qwen3_5_transliteration(&p2, Some(cancel.clone()))
                            .await
                            .unwrap_or_default();
                        let (s2_t, s2_tr) = crate::nl_convert::sanitize_transliteration_dual(&raw2, &s1);
                        s2 = if !s2_t.is_empty() { s2_t } else { s2_tr };
                        println!("    PASS-2 SOURCE = '{}'", s1);
                        println!("    PASS-2 TARGET = '{}'", second_target);
                        println!("    PASS-2 RAW    = '{}'", raw2.replace('\n', "\n"));
                        println!("    PASS-2 RESULT = '{}'", s2);
                    }
                } else {
                    println!("    PASS-2 SKIPPED (PASS-1 결과가 비어있거나 표기 체계 미반전)");
                }
            } else {
                println!("    PASS-2 SKIPPED (PASS-1 결과가 비어있음)");
            }
            crate::nl_convert::assign_transliterations(&src, &s1, &s2)
        };

        let final_pair = pair;
        
        if final_pair.0.is_empty() && final_pair.1.is_empty() {
            emit(&format!(
                "      ⚪ [SYNONYM SKIP] '{}' | 표기 체계가 뒤집히지 않아 별칭을 폐기했습니다. (property='{}')",
                src, cm.property
            ));
        } else {
            made += 1;
            emit(&format!(
                "      🔤 [SYNONYM EXPANSION] '{}' → native='{}' | roman='{}' (property='{}')",
                src, final_pair.0, final_pair.1, cm.property
            ));
            // 🌟 [LANG-CONSISTENCY LOG] 혼용 소스에서 언어 통일 별칭이 생성된 경우 추가 로그
            if is_mixed && (!final_pair.0.is_empty() || !final_pair.1.is_empty()) {
                emit(&format!(
                    "      🔤 [LANG-UNIFIED] 원본 혼용 → ko='{}' / en='{}' 로 언어별 분리 저장",
                    final_pair.0, final_pair.1
                ));
            }
        }
        cache.insert(src.clone(), final_pair.clone());

        // 🌟 [DEXIE CACHE SAVE] LLM 으로 생성한 결과를 Dexie 에 영구 저장합니다.
        //    다음 태스크(또는 앱 재시작 후)부터는 Qwen3.5 호출 없이 캐시 히트됩니다.
        //    ⚠️ out[i] 할당(move) '전에' 호출해야 borrow-after-move 를 피합니다.
        save_translit_cache(app_handle, &src, doc_lang, &final_pair.0, &final_pair.1);

        out[i] = final_pair;
    }

    if made > 0 || reused > 0 {
        emit(&format!(
            "  🔤 [SYNONYM EXPANSION / Qwen3.5-2B] 별칭 생성 {}건 | 캐시 재사용 {}건 | 대상 외 {}건",
            made, reused, skipped
        ));
    }

    out
}
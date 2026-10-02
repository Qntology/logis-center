use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use serde_json::json;
use crate::model::LogisModel;
use crate::scheduler::TRANSLIT_MEM_CACHE;
use tauri::Emitter;


fn translit_lang(lang: &str) -> String {
    let t = lang.trim();
    if t.is_empty() {
        return String::new();
    }
    crate::utils::bias_schema::lang_code_of(t)
}

fn translit_cache_key(word: &str, lang: &str) -> String {
    format!("{}\u{1}{}", translit_lang(lang), word.trim())
}

static TRANSLIT_RECHECKED: once_cell::sync::Lazy<std::sync::Mutex<Option<std::collections::HashSet<String>>>> =
    once_cell::sync::Lazy::new(|| std::sync::Mutex::new(None));

fn recheck_ledger_path() -> std::path::PathBuf {
    crate::utils::get_app_dir().join("cache").join("translit_recheck.json")
}

fn first_recheck(word: &str, lang: &str, engine: &str) -> bool {
    let key = format!("{}\u{1}{}", translit_cache_key(word, lang), engine);
    let mut guard = match TRANSLIT_RECHECKED.lock() {
        Ok(g) => g,
        Err(_) => return false,
    };
    if guard.is_none() {
        let loaded: std::collections::HashSet<String> = std::fs::read_to_string(recheck_ledger_path())
            .ok()
            .and_then(|s| serde_json::from_str::<Vec<String>>(&s).ok())
            .map(|v| v.into_iter().collect())
            .unwrap_or_default();
        *guard = Some(loaded);
    }
    let set = match guard.as_mut() {
        Some(s) => s,
        None => return false,
    };
    if !set.insert(key) {
        return false;
    }
    let path = recheck_ledger_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let list: Vec<&String> = set.iter().collect();
    if let Ok(body) = serde_json::to_string(&list) {
        let _ = std::fs::write(&path, body);
    }
    true
}

async fn dexie_translit_lookup(
    app_handle: &tauri::AppHandle,
    word: &str,
    lang: &str,
) -> Option<Vec<(String, String)>> {
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

    let _ = crate::scheduler::TRANSLIT_PENDING.lock().unwrap().remove(&request_id);

    match result {
        Ok(Ok(candidates)) => Some(candidates),
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

async fn query_translit_cache(
    app_handle: &tauri::AppHandle,
    word: &str,
    lang: &str,
) -> Option<(String, String)> {
    let key = translit_cache_key(word, lang);
    let canon = translit_lang(lang);

    if let Ok(map) = TRANSLIT_MEM_CACHE.lock() {
        if let Some(hit) = map.get(&key) {
            println!(
                "  💾 [TRANSLIT CACHE / MEM HIT] '{}' (lang='{}') → native='{}' | roman='{}'",
                word, canon, hit.0, hit.1
            );
            return Some(hit.clone());
        }
    }

    let raw = lang.trim().to_lowercase();
    let mut langs: Vec<String> = vec![canon.clone()];
    if !raw.is_empty() && raw != canon {
        langs.push(raw);
    }
    for name in crate::utils::bias_schema::lang_names_of(&canon) {
        if !langs.iter().any(|l| l == name) {
            langs.push(name.to_string());
        }
    }
    for (li, q_lang) in langs.iter().enumerate() {
        let candidates = dexie_translit_lookup(app_handle, word, q_lang).await?;
        let hit = match candidates.first() {
            Some(h) => h.clone(),
            None => continue,
        };
        if let Ok(mut map) = TRANSLIT_MEM_CACHE.lock() {
            map.insert(key.clone(), hit.clone());
        }
        if li == 0 {
            println!(
                "  💾 [TRANSLIT CACHE / DEXIE HIT] '{}' (lang='{}') → native='{}' | roman='{}'",
                word, canon, hit.0, hit.1
            );
        } else {
            println!(
                "  🔁 [TRANSLIT CACHE / LEGACY LANG] '{}' 는 예전 언어 키 '{}' 로 저장되어 있었습니다 → 정규 언어 코드 '{}' 로 옮겨 저장합니다. 이미지 문서(언어 이름 'korean')와 텍스트 문서(언어 코드 'ko')가 같은 값의 별칭을 서로 다른 캐시 줄로 나눠 갖지 않게 합니다.",
                word, q_lang, canon
            );
            save_translit_cache(app_handle, word, &canon, &hit.0, &hit.1);
        }
        return Some(hit);
    }
    println!(
        "  🔍 [TRANSLIT CACHE / MISS] '{}' (lang='{}') — Dexie 에 레코드가 없습니다.",
        word, canon
    );
    None
}

fn save_translit_cache(
    app_handle: &tauri::AppHandle,
    word: &str,
    lang: &str,
    native: &str,
    roman: &str,
) {
    let key = translit_cache_key(word, lang);
    let canon = translit_lang(lang);
    if let Ok(mut map) = TRANSLIT_MEM_CACHE.lock() {
        map.insert(key, (native.to_string(), roman.to_string()));
    }

    if native.trim().is_empty() && roman.trim().is_empty() {
        println!(
            "  💾 [TRANSLIT CACHE / SAVE-NEGATIVE] '{}' (lang='{}') — 음차 불가 판정을 영구 저장합니다.",
            word, canon
        );
    } else {
        println!(
            "  💾 [TRANSLIT CACHE / SAVE] '{}' (lang='{}') → native='{}' | roman='{}'",
            word, canon, native, roman
        );
    }

    let _ = app_handle.emit("translit-cache-save", json!({
        "word": word,
        "lang": canon,
        "native": native,
        "roman": roman,
        "engine": crate::model::lang_llm::engine_tag(lang)
    }));
}

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
    if let Some((native, roman, code)) = crate::nl_convert::canonical_entity_alias(&src, doc_lang) {
        println!(
            "[ANALYTIC] 🌐 [CANONICAL ALIAS] '{}' → native='{}' | roman='{}' (국가 코드 {}) — 국가명은 언어마다 정해진 이름이 있는 닫힌 어휘라 음차 대신 국가명 표의 표기를 씁니다.",
            src, native, roman, code
        );
        return (native, roman);
    }

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
        let native = crate::nl_convert::gate_native_alias(&src, native, "ANALYTIC");
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
    let mut phonetic_dropped = 0usize;
    let mut canonical_made = 0usize;
    let mut mixed_dropped = 0usize;
    let lang_engine_ready = crate::model::lang_llm::lang_engine_available(doc_lang);
    let recheck_engine = crate::model::lang_llm::engine_tag(doc_lang);
    let recheck_key = format!("{}@{}", recheck_engine, crate::nl_convert::TRANSLIT_GATE_REV);

    let mut generation_ready = false;
    let mut engine_label = String::from("Qwen3.5-2B");
    let mut lang_engine_used = false;
    let mut second_read: Vec<String> = Vec::new();
    let mut _translit_binding = crate::model::lang_llm::TranslitBinding::none();
    macro_rules! ensure_generation {
        () => {
            if !generation_ready {
                match model
                    .enter_translit_generation(
                        doc_lang,
                        Some(cancel.clone()),
                        "transliteration (first cache miss)",
                        task_id,
                    )
                    .await
                {
                    Ok((engine, binding)) => {
                        engine_label = engine.label();
                        lang_engine_used = engine.is_lang();
                        _translit_binding = binding;
                        generation_ready = true;
                        emit(&format!("  🔤 [TRANSLIT ENGINE] 이번 아이템의 음차 엔진: {}", engine_label));
                    }
                    Err(e) => {
                        println!("  ⚠️ [CROSSOVER] 음차용 Qwen3.5 전환 실패: {}. 이번 값은 건너뜁니다.", e);
                    }
                }
            }
        };
    }

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

        if let Some((native, roman, code)) = crate::nl_convert::canonical_entity_alias(&src, doc_lang) {
            let pair = (native, roman);
            let same_cached = TRANSLIT_MEM_CACHE
                .lock()
                .ok()
                .and_then(|m| m.get(&translit_cache_key(&src, doc_lang)).cloned())
                .map_or(false, |c| c == pair);
            if same_cached {
                reused += 1;
            } else {
                emit(&format!(
                    "      🌐 [CANONICAL ALIAS] '{}' → native='{}' | roman='{}' (국가 코드 {} · property='{}') | 국가명은 소리를 옮기는 값이 아니라 언어마다 정해진 이름이 있는 닫힌 어휘입니다. LLM 음차('China'→'신화', 'Germany'→'게르만이') 대신 국가명 표의 문서 언어 표기를 쓰고, 캐시에 남은 이전 음차도 이 값으로 덮어씁니다.",
                    src, pair.0, pair.1, code, cm.property
                ));
                crate::utils::score_dynamics::record_baseline("indexing.translit_canonical", 1.0);
                canonical_made += 1;
                save_translit_cache(app_handle, &src, doc_lang, &pair.0, &pair.1);
            }
            cache.insert(src.clone(), pair.clone());
            out[i] = pair;
            continue;
        }

        let cached_hit = query_translit_cache(app_handle, &src, doc_lang).await.filter(|hit| {
            match crate::nl_convert::cached_translit_recheck(&src, &hit.0, doc_lang, lang_engine_ready)
                .filter(|_| first_recheck(&src, doc_lang, &recheck_key))
            {
                Some(why) => {
                    emit(&format!(
                        "  🔁 [TRANSLIT CACHE / RECHECK] '{}' 캐시 별칭 native='{}' 을 다시 만듭니다: {} (엔진 {} · 게이트 {} 기준 최초 1회 · 결과가 같아도 이 조합으로는 다시 만들지 않습니다)",
                        src, hit.0, why, recheck_engine, crate::nl_convert::TRANSLIT_GATE_REV
                    ));
                    false
                }
                None => true,
            }
        });
        if let Some(mut dexie_hit) = cached_hit {
            let reglued = crate::nl_convert::reglue_native_alias(&src, &dexie_hit.0);
            if reglued != dexie_hit.0 {
                emit(&format!(
                    "  🔗 [TRANSLIT REGLUE] '{}' 캐시 별칭 '{}' → '{}' | 원문에서 공백 없이 이어진 단어('-' 등으로 붙은 합성어)는 문서 언어 표기에서도 붙여 씁니다. 같은 원문 단어가 언제나 같은 별칭 표기로 색인되게 하는 표기 일관성 교정입니다. 별칭은 item_chunks 의 벡터 청크로만 저장되고 FTS 색인(items 의 text·masked_text·data)에는 들어가지 않으므로, 이 교정이 바꾸는 것은 별칭 청크의 임베딩 문장뿐입니다.",
                    src, dexie_hit.0, reglued
                ));
                dexie_hit.0 = reglued;
                save_translit_cache(app_handle, &src, doc_lang, &dexie_hit.0, &dexie_hit.1);
            }
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

        if !crate::nl_convert::can_transliterate(&src, doc_lang) {
            cache.insert(src.clone(), (String::new(), String::new()));
            save_translit_cache(app_handle, &src, doc_lang, "", "");
            skipped += 1;
            continue;
        }
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

        // 🌟 [CROSSOVER] 여기서부터 Qwen3.5 를 부를 수 있습니다.
        //    any_ascii 로만 끝나는 경로도 있지만, 그 판정이 트랙별로 흩어져 있어
        //    호출 직전마다 개별 판정하면 분기가 폭발합니다.
        //    캐시를 통과한 값은 대부분 LLM 을 필요로 하므로 여기서 한 번 전환합니다.
        ensure_generation!();

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
            if !track_b_transliteration.is_empty() && !latin_words.is_empty() {
                track_b_transliteration = crate::nl_convert::gate_native_alias(
                    &latin_words.join(" "),
                    track_b_transliteration,
                    "TRACK-B",
                );
                if track_b_transliteration.is_empty() {
                    phonetic_dropped += 1;
                }
            }
            let digits_only = !non_latin_words.is_empty()
                && non_latin_words.iter().all(|w| crate::nl_convert::is_digit_word(w));
            let mut korean_unified_parts: Vec<String> = Vec::new();
            for w in &non_latin_words {
                korean_unified_parts.push(w.clone());
            }
            if !track_b_transliteration.is_empty() {
                korean_unified_parts.push(track_b_transliteration.clone());
            }
            let korean_unified = if !latin_words.is_empty() && track_b_transliteration.is_empty() {
                String::new()
            } else if digits_only {
                crate::nl_convert::place_digit_words(&src, &track_b_transliteration)
                    .unwrap_or_else(|| korean_unified_parts.join(" "))
            } else {
                korean_unified_parts.join(" ")
            };
            let mut english_unified_parts: Vec<String> = Vec::new();
            if !track_a_transliteration.is_empty() {
                english_unified_parts.push(track_a_transliteration.clone());
            }
            for w in &latin_words {
                english_unified_parts.push(w.clone());
            }
            let english_unified = if digits_only {
                crate::nl_convert::strip_special_chars_for_transliteration(&src)
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                english_unified_parts.join(" ")
            };
            println!("    [LANG-UNIFIED] native(ko) = '{}'", korean_unified);
            println!("    [LANG-UNIFIED] roman(en) = '{}'", english_unified);
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
            if crate::nl_convert::is_latin_dominant(&src) && !tr.is_empty() {
                let gated = crate::nl_convert::gate_native_alias(&src, tr, "PASS-1");
                if gated.is_empty() {
                    phonetic_dropped += 1;
                }
                gated
            } else {
                tr
            }
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
        if !s1_transliteration.contains("|||") {
            let leftover: Vec<String> = crate::nl_convert::find_mixed_script_words(&s1)
                .into_iter()
                .filter(|w| !src.split_whitespace().any(|sw| sw == w))
                .collect();
            if !leftover.is_empty() {
                emit(&format!(
                    "    🚫 [MIXED SCRIPT LEFTOVER] '{}' → '{}' | 재음차 뒤에도 한 단어 안에 두 문자 체계가 섞인 조각 {:?} 이 남았습니다. 이런 별칭은 어느 언어의 질의와도 맞지 않고 FTS 에 깨진 토큰만 남기므로 쓰지 않습니다.",
                    src, s1, leftover
                ));
                mixed_dropped += 1;
                s1 = String::new();
            }
        }

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

        let final_pair = (crate::nl_convert::reglue_native_alias(&src, &pair.0), pair.1);

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
            if is_mixed && (!final_pair.0.is_empty() || !final_pair.1.is_empty()) {
                emit(&format!(
                    "      🔤 [LANG-UNIFIED] 원본 혼용 → ko='{}' / en='{}' 로 언어별 분리 저장",
                    final_pair.0, final_pair.1
                ));
            }
        }
        cache.insert(src.clone(), final_pair.clone());
        let latin_left = !crate::nl_convert::split_words_by_script(&src).1.is_empty();
        if lang_engine_used && final_pair.0.is_empty() && latin_left && !second_read.contains(&src) {
            second_read.push(src.clone());
        } else {
            save_translit_cache(app_handle, &src, doc_lang, &final_pair.0, &final_pair.1);
        }
        out[i] = final_pair;
    }

    if !second_read.is_empty() {
        emit(&format!(
            "  🔁 [TRANSLIT SECOND READ] {} 가 원문 라틴 단어를 게이트를 통과하는 문서 언어 표기로 옮기지 못한 값 {}건을 Qwen3.5-2B(다국어 어휘)로 한 번 더 읽습니다: {:?} | 단일 언어 모델이 외국어 단어 읽기에 실패한 자리만 다국어 모델로 넘기고, 이 결과까지 게이트를 통과하지 못해야 음차 불가로 저장합니다.",
            engine_label,
            second_read.len(),
            second_read
        ));
    }
    for src in second_read.iter() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let slots: Vec<usize> = chunks
            .iter()
            .enumerate()
            .filter(|(_, c)| c.value_part.trim() == src.as_str())
            .map(|(j, _)| j)
            .collect();
        let mut pair = match slots.first() {
            Some(&j) => out[j].clone(),
            None => continue,
        };
        let (non_latin, latin) = crate::nl_convert::split_words_by_script(src);
        let latin_src = latin.join(" ");
        let prompt = if non_latin.is_empty() {
            crate::nl_convert::build_transliteration_prompt(src, doc_lang)
        } else {
            crate::nl_convert::build_transliteration_prompt_for_words(&latin, doc_lang)
        };
        let raw = model
            .call_base_transliteration(&prompt, Some(cancel.clone()))
            .await
            .unwrap_or_default();
        println!("    SECOND-READ RAW (Qwen3.5-2B) = '{}'", raw.trim());
        let (_t, tr) = if non_latin.is_empty() {
            crate::nl_convert::sanitize_transliteration_dual(&raw, src)
        } else {
            crate::nl_convert::sanitize_transliteration_dual_for_words(&raw, &latin)
        };
        let gated = crate::nl_convert::gate_native_alias(&latin_src, tr, "SECOND-READ");
        if !gated.trim().is_empty()
            && !crate::nl_convert::is_latin_dominant(&gated)
            && crate::nl_convert::find_mixed_script_words(&gated).is_empty()
        {
            let native = if non_latin.is_empty() { gated.clone() } else { format!("{} {}", non_latin.join(" "), gated) };
            let native = crate::nl_convert::reglue_native_alias(src, &native);
            if pair.1.is_empty() {
                let roman = crate::nl_convert::try_any_ascii_transliteration(&native).unwrap_or_default();
                pair = crate::nl_convert::assign_transliterations(src, &native, &roman);
            } else {
                pair.0 = native;
            }
            made += 1;
            crate::utils::score_dynamics::record_baseline("indexing.translit_second_read", 1.0);
            emit(&format!(
                "      🔤 [SYNONYM EXPANSION / SECOND READ] '{}' → native='{}' | roman='{}' | Qwen3.5-2B 가 읽은 표기가 발음·글자읽기 게이트를 통과했습니다.",
                src, pair.0, pair.1
            ));
        } else {
            crate::utils::score_dynamics::record_baseline("indexing.translit_second_read", 0.0);
            emit(&format!(
                "      ⚪ [TRANSLIT SECOND READ / NONE] '{}' | Qwen3.5-2B 의 표기도 게이트를 통과하지 못해 지금 결과(native='{}' · roman='{}')를 저장합니다.",
                src, pair.0, pair.1
            ));
        }
        save_translit_cache(app_handle, src, doc_lang, &pair.0, &pair.1);
        for j in slots {
            out[j] = pair.clone();
        }
    }
    if made > 0 || reused > 0 || phonetic_dropped > 0 || canonical_made > 0 || mixed_dropped > 0 {
        emit(&format!(
            "  🔤 [SYNONYM EXPANSION / {}] 별칭 생성 {}건 | 국가명 정규 별칭 {}건 | 캐시 재사용 {}건 | 대상 외 {}건 | 발음·글자읽기 게이트 폐기 {}건 | 혼용 조각 폐기 {}건",
            if generation_ready { engine_label.as_str() } else { "캐시" },
            made, canonical_made, reused, skipped, phonetic_dropped, mixed_dropped
        ));
    }

    if generation_ready {
        emit(&format!(
            "  🔁 [CROSSOVER] 음차 구간에서 {} 를 1회 올려 {}건을 처리했습니다. {}",
            engine_label,
            made,
            model.crossover_report()
        ));
    } else {
        emit("  ⚡ [CROSSOVER] 음차가 전부 캐시로 해결되어 Qwen3.5 를 올리지 않았습니다. (전환 0회)");
    }

    out
}
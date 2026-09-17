use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::Mutex;
use serde_json::{Value, json};
use anyhow::Result;
use tauri::Emitter;
use crate::store::{Task, VectorStore};
use crate::model::LogisModel;
use crate::utils::logger::log_task_progress;
use crate::parsing::PugMode;
pub const ANALYTIC_EVENT_TYPES: [&str; 4] = ["click", "hover", "change", "touch"];
pub const ANALYTIC_SEARCH_TYPES: [&str; 5] = ["click", "hover", "change", "report", "touch"];
pub fn event_type_anchor_phrases(event_type: &str) -> Vec<String> {
    // ① bias.json 우선 (semantic + bias)
    if let Some(node) = crate::parsing::BIAS_DICT
        .get("analytic_event_filters")
        .and_then(|v| v.get(event_type))
    {
        let mut out: Vec<String> = Vec::new();
        for field in ["semantic", "bias"] {
            if let Some(s) = node.get(field).and_then(|v| v.as_str()) {
                for p in crate::utils::ai_utils::split_bias_phrases_full(s) {
                    if !out.iter().any(|e| e == &p) {
                        out.push(p);
                    }
                }
            }
        }
        if !out.is_empty() {
            return out;
        }
    }
    let raw = match event_type {
        "click" => "click, clicked, pressed, tapped, selected, chose, picked, opened, pushed the button, selection, choice, purchase intent",
        "hover" => "hover, hovered, mouse over, lingered, dwelled, looked at, browsed, scanned, glanced, viewed without clicking, attention, interest",
        "change" => "change, changed, typed, entered, input, filled in, edited, modified, toggled, switched, picked an option, selected a value, form entry, keyword typed",
        "report" => "report, summary, overview, behaviour flow, user journey, pattern, trend, analysis, insight, statistics, aggregated result, most frequent",
        _ => "",
    };
    crate::utils::ai_utils::split_bias_phrases_full(raw)
}
pub fn event_type_prejudice_phrases(event_type: &str) -> Vec<String> {
    if let Some(s) = crate::parsing::BIAS_DICT
        .get("analytic_event_filters")
        .and_then(|v| v.get(event_type))
        .and_then(|n| n.get("prejudice"))
        .and_then(|v| v.as_str())
    {
        let p = crate::utils::ai_utils::split_bias_phrases_full(s);
        if !p.is_empty() {
            return p;
        }
    }
    let mut out: Vec<String> = Vec::new();
    for other in ANALYTIC_SEARCH_TYPES.iter() {
        if *other == event_type {
            continue;
        }
        for p in event_type_anchor_phrases(other) {
            if !out.iter().any(|e| e == &p) {
                out.push(p);
            }
        }
    }
    out
}

/// 🌟 [EVENT EXACT MATCH] bias.json 의 exact_match 배열로 완전일치 판정합니다.
///  time_filters / season_filters 의 exact_match 와 동일한 계약이며,
///  일치하면 벡터 경쟁도 LLM 도 거치지 않고 즉시 확정합니다.
pub fn event_type_exact_match(word: &str) -> Option<String> {
    let w = word.trim().to_lowercase();
    if w.is_empty() {
        return None;
    }
    let obj = crate::parsing::BIAS_DICT
        .get("analytic_event_filters")
        .and_then(|v| v.as_object())?;
    for (key, node) in obj {
        if !ANALYTIC_SEARCH_TYPES.iter().any(|t| t == key) {
            continue;
        }
        if let Some(arr) = node.get("exact_match").and_then(|v| v.as_array()) {
            if arr
                .iter()
                .any(|x| x.as_str().map_or(false, |s| s.trim().to_lowercase() == w))
            {
                return Some(key.clone());
            }
        }
    }
    None
}

/// 🌟 [EVENT PREFIX MATCH] 교착어 어절을 위해 exact_match 배열을 '접두 사전' 으로 재사용합니다.
///  "클릭한게" 는 exact_match 에 없지만 "클릭" 이 그 접두이므로 click 으로 확정됩니다.
///  사전은 bias.json 이 소유하므로 코드에는 어떤 언어의 어휘도 등장하지 않습니다.
pub fn event_type_prefix_key(word: &str) -> Option<String> {
    let (key, _stem) =
        crate::utils::ai_utils::prefix_match_filter_stem("analytic_event_filters", word)?;
    if ANALYTIC_SEARCH_TYPES.iter().any(|t| t == &key) {
        Some(key)
    } else {
        None
    }
}

/// 🌟 [MORPHOLOGICAL VARIANTS] 교착어 어절 하나에서 '검색 가능한 원형 후보'를 만들어 냅니다.
///  ── 근거 3단 ──
///   ① Stanza Lemma      : 모델이 직접 알려준 원형 ("클릭한게" → "클릭")
///   ② bias.json 접두 사전: exact_match 원소가 표면형의 접두이면 그 원소를 어간으로 확정
///   ③ 문자 접두 n-gram   : ①②가 모두 실패했을 때의 순수 구조 폴백
///  ── 왜 필요한가 ──
///   commerce 는 질의에 다른 토큰('제품')이 함께 있어 shared_prefix_stems 로 어간을 얻지만,
///   "가장 많이 클릭한게 뭐야?" 처럼 어간을 공유하는 형제 토큰이 없는 질의에서는
///   그 장치가 동작하지 않습니다. 세 근거를 순서대로 시도합니다.
///  ── 상한 ──
///   토큰당 최대 3개. 청크 폭발을 막고, 무의미한 조각은 SURPRISAL/편견 게이트가 걸러냅니다.
pub fn morphological_variants(word: &str, lemma: &str) -> Vec<String> {
    let surface = word.trim();
    let mut out: Vec<String> = Vec::new();
    if surface.chars().count() < 2 {
        return out;
    }

    fn push(v: &mut Vec<String>, surface: &str, cand: String) {
        let c = cand.trim().to_string();
        if c.is_empty() { return; }
        if c.chars().count() < 2 { return; }
        if c == surface { return; }
        if v.iter().any(|e| e == &c) { return; }
        v.push(c);
    }

    // ① Stanza Lemma 원형
    let l = lemma.trim();
    if !l.is_empty() && l != surface {
        let lc: String = l.chars().filter(|c| c.is_alphanumeric()).collect();
        let sc: String = surface.chars().filter(|c| c.is_alphanumeric()).collect();
        if !lc.is_empty()
            && sc.chars().count() > lc.chars().count()
            && (sc.starts_with(&lc) || sc.ends_with(&lc))
        {
            push(&mut out, surface, lc);
        } else if !lc.is_empty() {
            push(&mut out, surface, l.to_string());
        }
    }

    // ② bias.json exact_match 접두 사전
    for cat in ["analytic_event_filters", "time_filters", "season_filters"] {
        if let Some((_, stem)) = crate::utils::ai_utils::prefix_match_filter_stem(cat, surface) {
            push(&mut out, surface, stem);
        }
    }

    // ③ 문자 접두 n-gram (사전 없이 동작하는 최후 폴백)
    let chars: Vec<char> = surface.chars().collect();
    if chars.len() >= 3 {
        let hi = (chars.len() - 1).min(4);
        for n in 2..=hi {
            push(&mut out, surface, chars[..n].iter().collect::<String>());
        }
    }

    if out.len() > 3 {
        out.truncate(3);
    }
    out
}

/// 🌟 [ATTRIBUTE STRIP] PUG 한 줄에서 속성부(`[...]`)를 완전히 제거하고
///    `{indent}{tag} | {text}` 형태만 남깁니다.
///    pug_line_parts 는 속성값 내부의 파이프를 오인하지 않는 안전 파서이므로
///    `option[value="우체국|https://..."]` 같은 라인에서도 값이 깨지지 않습니다.
pub fn strip_pug_attributes(pug: &str) -> String {
    let mut out = String::new();
    for line in pug.lines() {
        if line.trim().is_empty() { continue; }
        let (indent, tag, _attrs, value) = crate::utils::ai_utils::pug_line_parts(line);
        let v = value.trim();
        if tag.is_empty() && v.is_empty() { continue; }
        let pad = " ".repeat(indent);
        if tag.is_empty() {
            out.push_str(&format!("{}| {}\n", pad, v));
        } else if v.is_empty() {
            out.push_str(&format!("{}{}\n", pad, tag));
        } else {
            out.push_str(&format!("{}{} | {}\n", pad, tag, v));
        }
    }
    out
}

/// 🌟 [HTML → SEMANTIC PUG] 원시 outerHTML 을 '속성이 하나도 없는 PUG' 로 접습니다.
///    ListMode 를 쓰는 이유:
///      · id / class / style 을 이미 버립니다.
///      · input[value] 와 selected option 의 텍스트는 살립니다.
///        (NoAttributesMode 는 select/option 을 통째로 버려 change 이벤트가 빈 값이 됩니다)
///    그 뒤 strip_pug_attributes 로 잔여 속성 대괄호까지 제거합니다.
pub fn html_to_semantic_pug(html: &str) -> String {
    let t = html.trim();
    if t.is_empty() { return String::new(); }
    let cleaned = crate::parsing::pre_clean_html(t);
    let pug = crate::parsing::convert_to_clean_pug(&cleaned, PugMode::ListMode, None);
    strip_pug_attributes(&pug)
}

/// 🌟 [BLOCK JOIN] action / relate 는 문자열 배열(outerHTML 목록)입니다.
///    각 요소를 개별 PUG 로 접은 뒤 구분선으로 이어 붙입니다.
fn html_array_to_semantic_pug(val: Option<&Value>, max_blocks: usize) -> String {
    let arr = match val.and_then(|v| v.as_array()) {
        Some(a) => a,
        None => {
            // 문자열 단일 값으로 들어오는 경우(구버전 페이로드)도 흡수합니다.
            if let Some(s) = val.and_then(|v| v.as_str()) {
                return html_to_semantic_pug(s);
            }
            return String::new();
        }
    };

    let mut out = String::new();
    let mut used = 0usize;
    for item in arr {
        if used >= max_blocks { break; }
        let s = match item.as_str() { Some(x) => x, None => continue };
        let block = html_to_semantic_pug(s);
        if block.trim().is_empty() { continue; }
        if !out.is_empty() { out.push_str("---\n"); }
        out.push_str(&block);
        used += 1;
    }
    out
}

/// 🌟 [RAW DETECT] 아직 구조화되지 않은 원시 이벤트인지 판정합니다.
///    Cron 이 사라졌으므로 action 은 배열(outerHTML 목록)로만 도착합니다.
fn is_raw_event(data: &Value) -> bool {
    data.get("action").map_or(false, |v| v.is_array())
        || data.get("relate").map_or(false, |v| v.is_array())
}

/// 🌟 [STRUCTURING PROBE] 모델 로드 없이 실제 처리 가능한 원시 이벤트 수를 사전 확인합니다.
///    structure_pending_analytics 에서 모델 로드 전에 호출하여,
///    처리할 항목이 0건이면 모델 로드 없이 조기 반환합니다.
///    원시 이벤트가 존재해도 HTML → PUG 변환 결과가 비어있으면 처리 불가능으로 간주합니다.
pub fn count_pending_structuring_targets(
    raw_docs: &[crate::store::TradeDocument],
    limit: usize,
) -> usize {
    let mut count = 0usize;
    for doc in raw_docs.iter() {
        if count >= limit { break; }
        let data: Value = serde_json::from_str(&doc.json_data).unwrap_or(json!({}));
        if !is_raw_event(&data) { continue; }
        let target_pug = html_array_to_semantic_pug(data.get("action"), 1);
        if target_pug.trim().is_empty() { continue; }
        count += 1;
    }
    count
}

// =====================================================================
// 🌟 [U-1 / 의도 에피소드 분할]
// ---------------------------------------------------------------------
//  ── 무엇이 문제였나 ──
//   흐름 리포트의 묶음 단위가 group_key = "{from}|{ref}" 즉 '한 사용자 × 한 페이지'
//   입니다. 그런데 같은 페이지 안에서도 의도는 바뀝니다.
//     10:02 카디건 클릭 / 10:03 색상 변경 / 10:04 코트 호버   ← 탐색·비교
//     10:31 배송 조회   / 10:32 반품 정책 호버                ← 다른 의도
//   현재는 5건이 한 흐름으로 합성되어 cross_action_flow 가
//   "카디건을 보다가 배송을 조회했다" 는 하나의 서사로 뭉개지고,
//   intent_evolution 이 전환점을 짚지 못합니다.
//
//  ── 문서 트랙과의 결정적 차이 ──
//   문서 추출에서는 시간이 없어 순위·인덱스로 치환해야 했습니다.
//   여기는 created_at_ts 가 실재하므로 '의미 거리' 와 '시간 간격' 이라는
//   두 개의 독립 신호를 결합할 수 있습니다.
//
//  ── 왜 시간을 '증폭기' 로만 쓰는가 ──
//   시간 간격이 단독으로 경계를 만들면, 사용자가 잠깐 자리를 비운 경우
//   (Δt 큼, 의미 동일)를 잘라 버립니다. 그래서
//     결합 거리 = 의미 거리 × (1 + 시간 z⁺)
//   로 두어, 의미가 같으면 시간이 아무리 벌어져도 0 이 되게 합니다.
//   z⁺ = max(0, z) 이므로 평균보다 짧은 간격은 증폭하지 않습니다.
//
//  ── 임계값에 상수를 쓰지 않는 방법 ──
//   이 코드베이스의 확립된 패턴을 그대로 씁니다.
//     trading.rs   CONTINUATION DRIFT : 격차 > 분포 표준편차
//     scheduler.rs EVIDENCE DEDUP     : dedup_floor = μ + 3σ
//     ai_utils.rs  T-2 TIE GATE       : 마진 < 꼬리 표준편차
//   따라서 '인접 결합 거리 > 그 세션 분포의 μ + σ' 를 경계로 봅니다.
//
//  ── 추가 모델 비용 0 ──
//   행동 문장 임베딩은 reindex_pending_embeddings 가 어차피 만드는 것과
//   같은 텍스트입니다. 여기서 배치 1회로 미리 만들 뿐입니다.
// =====================================================================

/// 인접 행동의 결합 거리로 의도 에피소드 경계를 찾습니다.
///
///  ── 인자 ──
///   records : 시간순 정렬된 행동 레코드. 각 원소는 "at" 키(epoch ms)를 가져야 합니다.
///   embeds  : records 와 같은 길이의 행동 문장 임베딩
///
///  ── 반환 ──
///   [start, end) 구간 목록. 분할하지 않으면 [(0, n)] 하나입니다.
pub fn split_into_episodes(records: &[Value], embeds: &[Vec<f32>]) -> Vec<(usize, usize)> {
    let n = records.len();
    if n == 0 { return Vec::new(); }
    // 인접 쌍이 2개 미만이면 분포를 추정할 수 없습니다.
    // (경계 후보가 1개뿐이면 μ+σ 판정이 항상 거짓이 되어 무의미합니다)
    if n < 4 || embeds.len() != n { return vec![(0, n)]; }

    // ── ① 의미 거리 ──
    let mut sem: Vec<f32> = Vec::with_capacity(n - 1);
    for i in 0..(n - 1) {
        let a = &embeds[i];
        let b = &embeds[i + 1];
        if a.iter().all(|x| *x == 0.0) || b.iter().all(|x| *x == 0.0) {
            sem.push(0.0);
            continue;
        }
        let cos = crate::utils::ai_utils::cosine_similarity(a, b);
        sem.push((1.0 - cos).max(0.0));
    }

    // ── ② 시간 간격의 z 값 (양수만) ──
    let ats: Vec<i64> = records
        .iter()
        .map(|r| r.get("at").and_then(|v| v.as_i64()).unwrap_or(0))
        .collect();
    let dts: Vec<f64> = (0..(n - 1))
        .map(|i| (ats[i + 1] - ats[i]).max(0) as f64)
        .collect();
    let dt_mu = dts.iter().sum::<f64>() / (dts.len() as f64);
    let dt_var = dts.iter().map(|d| (d - dt_mu) * (d - dt_mu)).sum::<f64>() / (dts.len() as f64);
    let dt_sd = dt_var.max(0.0).sqrt();
    let dt_z: Vec<f32> = dts
        .iter()
        .map(|d| {
            if dt_sd <= 1e-9 { 0.0 } else { (((d - dt_mu) / dt_sd) as f32).max(0.0) }
        })
        .collect();

    // ── ③ 결합 거리 ──
    let d: Vec<f32> = (0..(n - 1)).map(|i| sem[i] * (1.0 + dt_z[i])).collect();

    // ── ④ 경계 판정 : 격차 > 분포의 μ + σ ──
    let d_mu = d.iter().sum::<f32>() / (d.len() as f32);
    let d_var = d.iter().map(|x| (x - d_mu) * (x - d_mu)).sum::<f32>() / (d.len() as f32);
    let d_sd = d_var.max(0.0).sqrt();
    // 전부 같은 거리면(표준편차 0) 자를 근거가 없습니다.
    if d_sd <= 1e-6 { return vec![(0, n)]; }
    let cut_at = d_mu + d_sd;

    let mut bounds: Vec<usize> = Vec::new();
    for i in 0..(n - 1) {
        if d[i] > cut_at { bounds.push(i + 1); }
    }
    if bounds.is_empty() { return vec![(0, n)]; }

    // ── ⑤ 구간 조립 ──
    let mut segs: Vec<(usize, usize)> = Vec::new();
    let mut start = 0usize;
    for b in bounds.into_iter() {
        if b > start { segs.push((start, b)); }
        start = b;
    }
    if start < n { segs.push((start, n)); }

    // ── ⑥ 단독 구간 병합 ──
    //   흐름 리포트는 records.len() >= 2 를 요구합니다.
    //   1건짜리 에피소드는 리포트가 되지 못하고 통째로 버려지므로,
    //   더 가까운 이웃에 붙여 정보 유실을 막습니다.
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for seg in segs.into_iter() {
        if seg.1 - seg.0 >= 2 {
            merged.push(seg);
            continue;
        }
        match merged.last_mut() {
            Some(prev) => prev.1 = seg.1,
            None => merged.push(seg),
        }
    }
    // 첫 구간이 단독으로 남았고 뒤에 구간이 있으면 뒤에 붙입니다.
    if merged.len() >= 2 && merged[0].1 - merged[0].0 < 2 {
        let head = merged.remove(0);
        merged[0].0 = head.0;
    }
    if merged.is_empty() { return vec![(0, n)]; }
    merged
}

// =====================================================================
// 🌟 [U-3 / 시간 감쇠 유사도 융합]
// ---------------------------------------------------------------------
//  ── 무엇이 문제였나 ──
//   analytic 검색은 STAGE-4 계열에서 청크 코사인만 쌓고,
//   STAGE-5 병합·정규화까지 시간이 단 한 번도 점수에 개입하지 않습니다.
//   created_at 은 STAGE-6 리포트 레코드에 표시용으로만 실립니다.
//
//   행동 로그는 문서와 성격이 다릅니다.
//     무역 문서   B/L 은 3개월이 지나도 유효합니다. 의미만이 관련성입니다.
//     사용자 행동 1년 전 클릭은 대부분 무의미합니다. 시의성이 관련성의 일부입니다.
//
//  ── 기존 기간 필터로는 왜 부족한가 ──
//   analytic_query_prompt 의 규칙 1번은
//   "명시적 시간 표현이 없으면 time_intent 를 '' 로 반환하고 문맥으로 추측하지 말라"
//   입니다. 이 규칙은 옳습니다. "가장 많이 클릭한게 뭐야?" 에 기간을 상상해
//   넣으면 안 되기 때문입니다.
//   그러나 그 결과 기간 조건이 없는 질의가 다수이고,
//   그 경우 build_scope_filter 가 created_at 조건을 만들지 않아
//   1년 전 행동과 어제 행동이 완전히 동등하게 경쟁합니다.
//   즉 문제는 프롬프트가 아니라 '하드 컷 외에 시의성을 반영할 축이 없다' 는 구조입니다.
//
//  ── 감쇠 상수를 어떻게 없애는가 ──
//   지수 감쇠 exp(−λΔt) 의 λ 는 그 자체가 매직 상수입니다.
//   그래서 λ 를 직접 두지 않고 '반감기' 로 표현한 뒤,
//   반감기를 회수 집합의 경과 시간 '중앙값' 에서 얻습니다.
//     활발한 사이트 → 기록이 최근에 몰림 → 중앙값 작음 → 급격한 감쇠
//     뜸한 사이트   → 기록이 넓게 퍼짐   → 중앙값 큼   → 완만한 감쇠
//   중앙값 시점에서 가중치가 정확히 0.5 가 되므로
//   회수 집합의 절반은 증폭되고 절반은 감쇠합니다.
//   사이트마다 자동으로 다른 λ 가 나오고 새 상수가 생기지 않습니다.
//
//   이 발상은 기존 패턴과 같은 계보입니다.
//     TITLE FLOOR  자기선언 분포의 평균
//     dedup_floor  분포의 μ + 3σ
//     U-3          경과 시간 분포의 중앙값
//
//  ── 명시적 기간이 있으면 감쇠를 끕니다 ──
//   사용자가 '지난달' 을 지정했다면 그 구간 안에서는 최신이 더 중요하지 않습니다.
//   이미 SQL 이 구간을 잘랐는데 그 안에서 또 감쇠를 걸면
//   구간 앞부분이 부당하게 눌립니다.
// =====================================================================

/// 회수 집합의 경과 시간 분포에서 반감기(ms)를 유도합니다.
///
///  ── 반환 ──
///   Some(반감기 ms) / 표본이 부족하거나 전부 동시각이면 None.
///   None 이면 호출부는 감쇠를 적용하지 않습니다(현행 동작).
pub fn derive_recency_half_life(ages_ms: &[i64]) -> Option<f64> {
    let mut v: Vec<f64> = ages_ms
        .iter()
        .filter(|a| **a >= 0)
        .map(|a| *a as f64)
        .collect();
    // 표본이 4건 미만이면 중앙값이 분포를 대표하지 못합니다.
    if v.len() < 4 { return None; }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = v[v.len() / 2];
    // 전 기록이 사실상 동시각이면 감쇠가 의미 없습니다.
    if median <= 1000.0 { return None; }
    Some(median)
}

/// 경과 시간에 대한 시간 감쇠 가중치를 산출합니다.
///
///  ── 성질 ──
///   age = 0        → 1.0
///   age = 반감기   → 0.5
///   age → ∞        → 0 에 수렴하되 0 이 되지는 않습니다.
///
///  ── 왜 0 이 되면 안 되는가 ──
///   가중치가 0 이면 오래된 기록이 검색 결과에서 완전히 사라집니다.
///   그러나 '작년에 이 사용자가 무엇을 했는가' 는 여전히 유효한 질문이고,
///   시간 필터가 없는 질의에서 답이 통째로 없어지면 리콜 손실입니다.
///   감쇠는 '순위를 낮추는 것' 이지 '배제하는 것' 이 아닙니다.
pub fn recency_weight(age_ms: i64, half_life_ms: f64) -> f32 {
    if half_life_ms <= 0.0 { return 1.0; }
    let a = age_ms.max(0) as f64;
    let w = 0.5f64.powf(a / half_life_ms);
    (w as f32).clamp(0.0, 1.0)
}
///  반환값 = 구조화에 성공한 이벤트 건수
pub async fn run_analytic_structuring(
    store: &VectorStore,
    model: &LogisModel,
    cancel: &Arc<AtomicBool>,
    app_handle: &tauri::AppHandle,
    task_id: &str,
    limit: usize,
) -> Result<usize> {
    let app_handle_clone = app_handle.clone();
    let tid = task_id.to_string();
    let emit_term = move |msg: &str| {
        println!("{}", msg);
        let _ = app_handle_clone.emit(
            "task-console-log",
            json!({ "task_id": tid, "text": format!("{}\n", msg) })
        );
    };

    if cancel.load(Ordering::Relaxed) {
        return Ok(0);
    }

    let type_list = ANALYTIC_EVENT_TYPES
        .iter()
        .map(|t| format!("'{}'", t))
        .collect::<Vec<_>>()
        .join(", ");
    let filter = format!(
        "mode = 'analytic' AND updated_at = 0 AND type IN ({})",
        type_list
    );

    let raw_logs = store
        .get_all_items("items", 500, 0, Some(filter))
        .await
        .unwrap_or_default();

    if raw_logs.is_empty() {
        return Ok(0);
    }

    // ── 구조화 대상만 추립니다. (이미 문장이 확정된 행은 제외) ──
    let mut targets: Vec<(crate::store::TradeDocument, Value)> = Vec::new();
    for doc in raw_logs {
        if targets.len() >= limit { break; }
        let data: Value = serde_json::from_str(&doc.json_data).unwrap_or(json!({}));
        if !is_raw_event(&data) { continue; }
        targets.push((doc, data));
    }

        if targets.is_empty() {
        return Ok(0);
    }

    // ── ① [PRE-FILTER / MODEL-LOAD GATE] 모델 로드 전에 실제 요약 가능한 텍스트가 있는지 사전 확인 ──
    //    원시 이벤트가 존재해도 HTML → PUG 변환 결과가 비어있으면 모델 로드가 무의미합니다.
    //    기존에는 3건이 들어와도 전부 요약 실패(빈 결과)로 스킵되는 경우에도 모델이 로드되었습니다.
    //    이제 사전 필터링으로 처리 가능한 항목이 0건이면 모델 로드 없이 즉시 반환합니다.
    let mut pre_checked: Vec<(crate::store::TradeDocument, Value, String)> = Vec::new();
    for (doc, data) in targets.into_iter() {
        if cancel.load(Ordering::Relaxed) {
            break;
        }
        let target_pug = html_array_to_semantic_pug(data.get("action"), 1);
        if target_pug.trim().is_empty() {
            emit_term(&format!(
                "  ⚪ [ANALYTIC SKIP] id='{}' 는 대상 엘리먼트에서 읽을 수 있는 텍스트가 없어 건너뜁니다.",
                doc.id
            ));
            continue;
        }
        pre_checked.push((doc, data, target_pug));
    }

    if pre_checked.is_empty() {
        emit_term("[ANALYTIC] ⚪ 실제 요약 가능한 원시 이벤트가 0건이라 모델 로드를 건너뜁니다.");
        return Ok(0);
    }

    emit_term(&format!(
        "[ANALYTIC] 🧠 구조화 대상 원시 이벤트 {}건 발견. HTML → PUG → 속성 제거 → Qwen3.5 2B 요약을 시작합니다.",
        pre_checked.len()
    ));
    log_task_progress(app_handle, task_id, &json!({
        "category": "Analytic Structuring",
        "summary": format!("Summarizing {} behaviour event(s)...", pre_checked.len()),
        "spinner": "⠋"
    }));

    model
        .secure_vram_relay(
            crate::model::ModelSize::Qwen3_5,
            None,
            Some(cancel.clone()),
            false,
            None,
        )
        .await?;

    let now_ts = chrono::Utc::now().timestamp_millis();
    let mut processed = 0usize;

    // 흐름(report) 합성을 위해 (from, ref) 로 묶어 둡니다.
    let mut flow_groups: std::collections::HashMap<String, Vec<Value>> =
        std::collections::HashMap::new();
    let mut flow_envelope: std::collections::HashMap<String, crate::store::TradeDocument> =
        std::collections::HashMap::new();

    let total = pre_checked.len();
    for (idx, (doc, data, target_pug_pre)) in pre_checked.into_iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            emit_term("[ANALYTIC] 🛑 사용자 취소로 구조화를 중단합니다.");
            break;
        }
        // 🌟 [BUSY RECHECK / 매 반복]
        //
        //  ── 왜 cancel 만으로 부족한가 ──
        //   cancel 은 '사용자가 중단 버튼을 눌렀는가' 이고,
        //   스케줄러 태스크가 새로 시작한 것과는 무관하게 false 로 남습니다.
        //   structure_pending_analytics 상단의 ACTIVE_TASK_MEM 가드는
        //   함수 진입 시 1회뿐이라, 그 이후 시작된 태스크와 이 루프가 병렬로 돕니다.
        //
        //  ── 피해 규모 ──
        //   이 루프는 Qwen3.5 2B 를 상주시킨 채 돕니다.
        //   스케줄러가 secure_vram_relay 로 모델을 전환하려 할 때
        //   2GB 가 VRAM 에 남아 있으면 wait_for_vram_settle 이 목표치를 못 채우고,
        //   전환이 통째로 지연되거나 OOM 재시도 경로로 빠집니다.
        //
        //  ── 왜 break 인가 ──
        //   구조화는 폴링으로 재호출되는 백그라운드 작업이라 잔여분이 유실되지 않습니다.
        //   updated_at = 0 인 draft 는 다음 폴링에서 그대로 다시 잡힙니다.
        if crate::IS_SEARCHING.load(Ordering::SeqCst)
            || crate::ACTIVE_TASK_MEM.read().map(|m| m.is_some()).unwrap_or(false)
        {
            emit_term(&format!(
                "[ANALYTIC] ⏸️ 전경 작업(태스크/검색)이 시작되어 구조화를 중단합니다. (이번 회차 {}건 처리, 잔여분은 다음 폴링에서 재개)",
                processed
            ));
            break;
        }

        let percent = (((idx as f32) / (total as f32)) * 100.0) as i32;
        log_task_progress(app_handle, task_id, &json!({
            "category": format!("Analytic Structuring ({}/{})", idx + 1, total),
            "summary": format!("Summarizing behaviour event ({}%)...", percent),
            "spinner": "⠋"
        }));

        let link = data
            .get("link")
            .or_else(|| data.get("href"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let doc_lang = crate::utils::lang_utils::detect_document_language(
            &data.get("action").map(|v| v.to_string()).unwrap_or_default()
        );

        // ── ① HTML → PUG → 속성 전량 제거 (사전 계산된 값 재사용) ──
        let mut target_pug = target_pug_pre;
        let mut related_pug = html_array_to_semantic_pug(data.get("relate"), 8);

        if target_pug.trim().is_empty() {
            emit_term(&format!(
                "  ⚪ [ANALYTIC SKIP] id='{}' 는 대상 엘리먼트에서 읽을 수 있는 텍스트가 없어 건너뜁니다.",
                doc.id
            ));
            continue;
        }

        // ── ② 컨텍스트 상한 ──
        target_pug = model.truncate_pug_context(&target_pug, true, 1200, None).await;
        related_pug = model.truncate_pug_context(&related_pug, true, 2400, None).await;

        // 🌟 [SDS SCOPE / analytic]
        //
        //  ── 무엇이 문제였나 (실측) ──
        //   이 함수에는 enter_scope 가 한 번도 없었습니다. 그래서
        //   U-1 이 이미 호출하고 있는 record_baseline("analytic.episodes_per_group") 조차
        //   'unscoped' 로 떨어지거나 버려졌고, score_dynamics.json 의 scopes 가 {} 였습니다.
        //
        //  ── 왜 함수 진입부가 아니라 여기인가 ──
        //   run_analytic_structuring 은 서로 다른 사이트(cc)·페이지(ref)의 이벤트를
        //   한 번에 순회합니다. 진입부에서 한 번만 세우면 모든 사이트의 관측이
        //   한 스코프에 뭉쳐, 기획 2-3 이 정한 analytic 1차 스코프(cc)가 무의미해집니다.
        //   enter_scope 는 단순 덮어쓰기라 refine_primary 처럼 이관이 일어나지 않으므로
        //   문서마다 갈아끼워도 이전 스코프의 관측이 오염되지 않습니다.
        crate::utils::score_dynamics::enter_scope(
            "",
            crate::utils::score_dynamics::Track::Analytic,
            &doc.cc,
            &doc.r#ref,
        );
        // 🌟 [Phase 0 관측] PUG 접기 결과의 구조 밀도입니다.
        //    '요약이 실패하는 문서는 애초에 읽을 라인이 없었다' 를 나중에 확인하기 위한 사실이며,
        //    판정에는 전혀 쓰이지 않습니다.
        crate::utils::score_dynamics::record_baseline(
            "analytic.pug_lines",
            target_pug.lines().filter(|l| !l.trim().is_empty()).count() as f32,
        );
        crate::utils::score_dynamics::record_baseline(
            "analytic.related_pug_lines",
            related_pug.lines().filter(|l| !l.trim().is_empty()).count() as f32,
        );

        emit_term(&format!(
            "  🧩 [SEMANTIC PUG] id='{}' | type='{}' | link='{}'\n{}",
            doc.id, doc.r#type, link, target_pug.trim()
        ));

        // ── ③ Qwen3.5 2B 시맨틱 요약 ──
        let prompt = crate::prompts::analytic_semantic_prompt(
            &doc.r#type,
            &link,
            &doc_lang,
            &target_pug,
            if related_pug.trim().is_empty() { "(none)" } else { &related_pug },
        );

        let params = crate::openai_types::ChatCompletionParameters {
            messages: vec![
                crate::openai_types::ChatCompletionRequestMessage::User(
                    crate::openai_types::ChatCompletionRequestUserMessage {
                        content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(prompt),
                        name: None,
                    }
                )
            ],
            model: "qwen3.5".to_string(),
            max_tokens: Some(768),
            temperature: Some(0.0),
            top_p: Some(0.95),
            ..Default::default()
        };

        let res_text = if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
            gen.generate(
                params,
                Some(cancel.clone()),
                Some(format!("{}_sem_{}", task_id, idx)),
                None,
                None,
                None,
            )
            .await
            .unwrap_or_default()
        } else {
            String::new()
        };

        let parsed = crate::parsing::parse_json_from_llm(&res_text);

        let action = parsed.get("action").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        let summary = parsed.get("summary").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        let relate: Vec<String> = parsed
            .get("relate")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str().map(|s| s.trim().to_string()))
                    .filter(|s| !s.is_empty())
                    .collect()
            })
            .unwrap_or_default();

        if action.is_empty() && summary.is_empty() {
            // 🌟 [SDS] LLM 이 값을 만들지 못한 사실은 판정과 독립된 신호입니다.
            //    기획 T-3 의 '거절 트레이스' 와 같은 계보이며, 학습 특이도의 입력이 됩니다.
            //    GateKind::Format 을 재사용하는 이유는 merge.rs 와 동일합니다.
            //    전용 종류를 추가하면 열거형이 늘어 파일 스키마가 바뀌고
            //    SDS_RECIPE 세대 무효화가 필요해지므로 기존 축을 씁니다.
            crate::utils::score_dynamics::record_field_seen("action");
            crate::utils::score_dynamics::record_field_reject(
                "action",
                crate::utils::score_dynamics::GateKind::Format,
            );
            emit_term(&format!(
                "  ⚠️ [ANALYTIC EMPTY] id='{}' 요약 결과가 비어 있어 이번 라운드에서는 확정하지 않습니다.",
                doc.id
            ));
            continue;
        }

        // ── ④ 검색 본문 조립 ──
        let mut combined = String::new();
        if !action.is_empty() {
            combined.push_str(&action);
        }
        if !summary.is_empty() {
            if !combined.is_empty() { combined.push(' '); }
            combined.push_str(&summary);
        }
        if !relate.is_empty() {
            combined.push(' ');
            combined.push_str(&relate.join(", "));
        }

        // 🌟 [TRANSLIT IN STRUCTURIZATION] 전처리 단계에서 음차를 수행합니다.
        //    Qwen3.5 가 이미 로드되어 있으므로 별도 모델 로딩이 불필요합니다.
        //    방향: 영어 단어 → 문서 언어(한글/일어/중어 등) 로 음차.
        //    한글→한글 같은 동일 언어 음차는 수행하지 않습니다.
        let translit_native;
        let translit_roman;
        if !action.is_empty() {
            let (tn, tr) = crate::scheduler::translit::transliterate_cross_language(
                model, &action, &doc_lang, cancel, app_handle, task_id,
            ).await;
            translit_native = tn;
            translit_roman = tr;
        } else {
            translit_native = String::new();
            translit_roman = String::new();
        }

        let mut new_data = data.clone();
        if let Some(o) = new_data.as_object_mut() {
            o.insert("action".to_string(), json!(action.clone()));
            o.insert("summary".to_string(), json!(summary.clone()));
            o.insert("relate".to_string(), json!(relate.clone()));
            o.insert("text".to_string(), json!(combined.clone()));
            o.insert("masked_text".to_string(), json!(combined.clone()));
            o.insert("mode".to_string(), json!("analytic"));
            o.insert("updated_at".to_string(), json!(now_ts));
            // 🌟 전처리에서 확정한 음차 결과를 저장합니다.
            if !translit_native.is_empty() {
                o.insert("translit_native".to_string(), json!(translit_native));
            }
            if !translit_roman.is_empty() {
                o.insert("translit_roman".to_string(), json!(translit_roman));
            }
            // 🌟 구조화가 끝났으므로 벡터를 새로 만들어야 합니다.
            //    reindex_pending_embeddings 는 embed 플래그가 1이면 건너뛰므로 제거합니다.
            o.remove("embed");
        }

        let _ = store
            .upsert_item(
                "items",
                &doc.id,
                &doc.r#type,
                new_data.clone(),
                None,
                None, // 🌟 vision_vec: analytic 구조화 경로는 비전 벡터 없음
                Some(&doc.from),
                Some(&doc.to),
                Some(&doc.cc),
                Some(&doc.bcc),
                Some(&doc.r#ref),
                None,
            )
            .await;

        // 🌟 [SDS / analytic 구조화 관측]
        //  Phase 0 이므로 판정에는 전혀 개입하지 않고 사실만 남깁니다.
        //   · action_len   : 행동 문장의 길이. 너무 짧으면 요약 실패의 전조입니다.
        //   · relate_count : 관련 요소 수. U-1 에피소드 분할의 문맥 밀도 지표입니다.
        //   · field(action): 위 EMPTY 분기와 짝을 이루는 분모입니다.
        //     이 분모가 없으면 learned_specificity 가 '거절률 100%' 라는 거짓값을 냅니다.
        //     (실측: 비전 트랙이 정확히 그 상태입니다 — seen == reject, assigned == 0)
        crate::utils::score_dynamics::record_baseline(
            "analytic.action_len",
            action.chars().count() as f32,
        );
        crate::utils::score_dynamics::record_baseline(
            "analytic.relate_count",
            relate.len() as f32,
        );
        crate::utils::score_dynamics::record_baseline(
            "analytic.summary_len",
            summary.chars().count() as f32,
        );
        crate::utils::score_dynamics::record_field_seen("action");
        crate::utils::score_dynamics::record_field_assigned("action", 0.0);
        crate::utils::score_dynamics::record_field_seen("summary");
        if summary.is_empty() {
            crate::utils::score_dynamics::record_field_reject(
                "summary",
                crate::utils::score_dynamics::GateKind::Format,
            );
        } else {
            crate::utils::score_dynamics::record_field_assigned("summary", 0.0);
        }
        crate::utils::score_dynamics::record_field_seen("relate");
        if relate.is_empty() {
            crate::utils::score_dynamics::record_field_reject(
                "relate",
                crate::utils::score_dynamics::GateKind::Format,
            );
        } else {
            crate::utils::score_dynamics::record_field_assigned("relate", 0.0);
        }
        emit_term(&format!(
            "  ✅ [ANALYTIC STRUCTURED] id='{}' | action=\"{}\" | relate={}건",
            doc.id, action, relate.len()
        ));
        processed += 1;

        let group_key = format!("{}|{}", doc.from, doc.r#ref);
        flow_groups.entry(group_key.clone()).or_insert_with(Vec::new).push(json!({
            "at": doc.created_at_ts,
            "type": doc.r#type,
            "link": link,
            "action": action,
            "summary": summary,
            "relate": relate
        }));
        flow_envelope.entry(group_key).or_insert(doc);
    }

    // ── ⑤ 흐름 리포트(report) 합성 ──
    //    analytics-logis-center 의 cross_action_flow / intent_evolution /
    //    consistent_preferences 3축을 그대로 로컬에서 재현합니다.
    //
    // 🌟 [U-1] 묶음 단위를 '페이지' 에서 '의도 에피소드' 로 정제합니다.
    //
    //  ── 왜 여기서 임베딩을 만드는가 ──
    //   구조화 루프 안에서는 action 문장이 아직 확정되지 않은 시점이 섞여 있고,
    //   레코드마다 임베딩을 부르면 Qwen3.5 와 granite 사이를 왕복합니다.
    //   전 그룹의 전 레코드를 한 배치로 처리하면 페이즈 전환이 최대 2회입니다.
    //
    //  ── 스왑이 0 이 될 수도 있습니다 ──
    //   enter_embedding_phase 는 예산을 판정해, 여유가 있으면
    //   Qwen3.5 를 유지한 채 granite(97M) 를 얹습니다(COEXIST).
    //   여유가 없을 때만 스왑이 발생하며, 그 판정은 실측 기반입니다.
    let mut episode_embeds: std::collections::HashMap<String, Vec<Vec<f32>>> =
        std::collections::HashMap::new();
    {
        // 그룹별 행동 문장을 시간순으로 모읍니다.
        let mut order: Vec<String> = Vec::new();
        let mut texts: Vec<String> = Vec::new();
        let mut spans: Vec<(String, usize, usize)> = Vec::new();
        for (gk, recs) in flow_groups.iter() {
            if recs.len() < 2 { continue; }
            let mut sorted = recs.clone();
            sorted.sort_by_key(|r| r.get("at").and_then(|v| v.as_i64()).unwrap_or(0));
            let start = texts.len();
            for r in sorted.iter() {
                let a = r.get("action").and_then(|v| v.as_str()).unwrap_or("").trim();
                let s = r.get("summary").and_then(|v| v.as_str()).unwrap_or("").trim();
                let t = if !a.is_empty() {
                    a.to_string()
                } else if !s.is_empty() {
                    s.to_string()
                } else {
                    " ".to_string()
                };
                texts.push(t);
            }
            spans.push((gk.clone(), start, texts.len()));
            order.push(gk.clone());
        }
        if !texts.is_empty() {
            match model.enter_embedding_phase("analytic episode split").await {
                Ok(_) => {
                    let embs = model
                        .get_embedding_batch(texts.clone())
                        .await
                        .unwrap_or_else(|_| vec![vec![0.0; 384]; texts.len()]);
                    for (gk, s, e) in spans.into_iter() {
                        episode_embeds.insert(gk, embs[s..e].to_vec());
                    }
                    emit_term(&format!(
                        "[ANALYTIC] 🧬 [EPISODE EMBED] 흐름 그룹 {}개 · 행동 {}건을 배치 1회로 임베딩했습니다. {}",
                        order.len(),
                        texts.len(),
                        model.crossover_report()
                    ));
                    // 흐름 합성은 Qwen3.5 가 필요하므로 생성 페이즈로 되돌립니다.
                    if let Err(e) = model
                        .switch_to_generation(
                            crate::model::ModelSize::Qwen3_5,
                            Some(cancel.clone()),
                            None,
                            "analytic flow synthesis",
                        )
                        .await
                    {
                        emit_term(&format!(
                            "[ANALYTIC] ⚠️ 흐름 합성용 생성 모델 복귀 실패({}). 에피소드 분할 없이 진행합니다.",
                            e
                        ));
                        episode_embeds.clear();
                    }
                }
                Err(e) => {
                    emit_term(&format!(
                        "[ANALYTIC] ⚠️ 임베딩 페이즈 진입 실패({}). 에피소드 분할 없이 페이지 단위로 합성합니다.",
                        e
                    ));
                }
            }
        }
    }

    for (group_key, mut records) in flow_groups.into_iter() {
        if cancel.load(Ordering::Relaxed) { break; }
        if records.len() < 2 { continue; }
        let env_doc = match flow_envelope.get(&group_key) { Some(d) => d.clone(), None => continue };
        records.sort_by_key(|r| r.get("at").and_then(|v| v.as_i64()).unwrap_or(0));
        if records.len() > 24 { records.truncate(24); }

        // 🌟 [SDS SCOPE / analytic 흐름]
        //  문서 루프에서 마지막으로 세운 스코프가 이 그룹의 것이라는 보장이 없습니다.
        //  flow_groups 는 HashMap 이라 순회 순서가 문서 처리 순서와 무관합니다.
        //  그룹의 봉투 문서로 스코프를 다시 확정합니다.
        crate::utils::score_dynamics::enter_scope(
            "",
            crate::utils::score_dynamics::Track::Analytic,
            &env_doc.cc,
            &env_doc.r#ref,
        );
        crate::utils::score_dynamics::record_baseline(
            "analytic.flow_records",
            records.len() as f32,
        );
        // 🌟 [U-2 입력] 이벤트 타입 전이입니다.
        //  기획 U-2 의 1차 마르코프 전이 사전은 record_transition 이 유일한 입력인데,
        //  현재 코드베이스 어디에서도 이 함수가 호출되지 않아 영원히 냉간 상태였습니다.
        //  (실측: 전 스코프의 transition 이 {} )
        //  검색 경로의 도메인 전이와 키가 섞이지 않도록 'evt:' 접두어를 붙입니다.
        //  transition_prior 는 "from>" 접두 매칭으로 분모를 세므로 접두어가 있어도 동작합니다.
        for w in records.windows(2) {
            let a = w[0].get("type").and_then(|v| v.as_str()).unwrap_or("");
            let b = w[1].get("type").and_then(|v| v.as_str()).unwrap_or("");
            if a.is_empty() || b.is_empty() { continue; }
            crate::utils::score_dynamics::record_transition(
                &format!("evt:{}", a),
                &format!("evt:{}", b),
            );
        }

        // 🌟 [U-1] 의도 에피소드로 분할합니다.
        //    임베딩이 없으면(진입 실패 등) 단일 구간이 되어 기존 동작과 동일합니다.
        let segments: Vec<(usize, usize)> = match episode_embeds.get(&group_key) {
            Some(embs) => {
                let use_embs: Vec<Vec<f32>> = embs.iter().take(records.len()).cloned().collect();
                if use_embs.len() == records.len() {
                    split_into_episodes(&records, &use_embs)
                } else {
                    vec![(0, records.len())]
                }
            }
            None => vec![(0, records.len())],
        };
        if segments.len() > 1 {
            let spans: Vec<String> = segments
                .iter()
                .map(|(s, e)| {
                    let first = records[*s]
                        .get("action")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .chars()
                        .take(28)
                        .collect::<String>();
                    format!("[{}..{}) \"{}\"", s, e, first)
                })
                .collect();
            emit_term(&format!(
                "  ✂️ [EPISODE SPLIT] '{}' 의 행동 {}건이 의도 에피소드 {}개로 분리되었습니다: {}",
                group_key, records.len(), segments.len(), spans.join(" | ")
            ));
            crate::utils::score_dynamics::record_baseline(
                "analytic.episodes_per_group",
                segments.len() as f32,
            );
        }

        for (ep_idx, (seg_start, seg_end)) in segments.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) { break; }
            let records: Vec<Value> = records[*seg_start..*seg_end].to_vec();
            if records.len() < 2 { continue; }
            let records_json = serde_json::to_string_pretty(&records).unwrap_or_else(|_| "[]".to_string());
            let doc_lang = crate::utils::lang_utils::detect_document_language(&records_json);
            let prompt = crate::prompts::analytic_flow_prompt_scoped(
                &doc_lang,
                &records_json,
                ep_idx,
                segments.len(),
            );
        let params = crate::openai_types::ChatCompletionParameters {
            messages: vec![
                crate::openai_types::ChatCompletionRequestMessage::User(
                    crate::openai_types::ChatCompletionRequestUserMessage {
                        content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(prompt),
                        name: None,
                    }
                )
            ],
            model: "qwen3.5".to_string(),
            max_tokens: Some(1024),
            temperature: Some(0.0),
            top_p: Some(0.95),
            ..Default::default()
        };

        let res_text = if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
            gen.generate(
                params,
                Some(cancel.clone()),
                Some(format!("{}_flow", task_id)),
                None,
                None,
                None,
            )
            .await
            .unwrap_or_default()
        } else {
            String::new()
        };

        let parsed = crate::parsing::parse_json_from_llm(&res_text);
        let cross_action = parsed.get("cross_action_flow").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        let intent_evo = parsed.get("intent_evolution").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        let preferences = parsed.get("consistent_preferences").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        for (fname, fval) in [
            ("cross_action_flow", &cross_action),
            ("intent_evolution", &intent_evo),
            ("consistent_preferences", &preferences),
        ] {
            crate::utils::score_dynamics::record_field_seen(fname);
            if fval.is_empty() {
                crate::utils::score_dynamics::record_field_reject(
                    fname,
                    crate::utils::score_dynamics::GateKind::Format,
                );
            } else {
                crate::utils::score_dynamics::record_field_assigned(fname, 0.0);
            }
        }
        crate::utils::score_dynamics::record_baseline(
            "analytic.report_len",
            (cross_action.chars().count() + intent_evo.chars().count() + preferences.chars().count()) as f32,
        );
        if cross_action.is_empty() && intent_evo.is_empty() && preferences.is_empty() {
            crate::utils::score_dynamics::record_baseline("analytic.flow_empty", 1.0);
            continue;
        }
        crate::utils::score_dynamics::record_baseline("analytic.flow_empty", 0.0);

        let report_text = format!("{} {} {}", cross_action, intent_evo, preferences)
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");

        // 🌟 [DETERMINISTIC REPORT ID] 난수 대신 (사용자 + 페이지 + 일자) 기반 해시.
        //    같은 날 같은 페이지의 흐름은 새 문서를 만들지 않고 갱신되므로
        //    리포트가 무한히 불어나지 않습니다.
        //
        // 🌟 [U-1] 에피소드 인덱스를 키에 더합니다.
        //
        //  ── 왜 필요한가 ──
        //   에피소드가 2개인데 키가 같으면 두 번째가 첫 번째를 덮어써
        //   '탐색 → 이탈' 중 하나만 남습니다. 분할한 의미가 사라집니다.
        //
        //  ── 왜 여전히 결정론인가 ──
        //   ep_idx 는 시간순 구간의 순번이므로, 같은 입력을 다시 처리하면
        //   같은 번호가 나옵니다. 난수가 아니라 리포트가 불어나지 않습니다.
        //
        //  ── 단일 에피소드는 기존 키를 유지합니다 ──
        //   ep_idx 가 0 이고 구간이 하나뿐이면 접미사를 붙이지 않아,
        //   기존에 생성된 리포트 문서와 ID 가 그대로 이어집니다.
        let day_bucket = now_ts / 86_400_000;
        let ep_suffix = if segments.len() > 1 {
            format!("#ep{}", ep_idx)
        } else {
            String::new()
        };
        let report_id = crate::utils::hash::hash_id(&format!(
            "report{}{}{}{}",
            env_doc.from, env_doc.r#ref, day_bucket, ep_suffix
        ));

        // 🌟 [U-1] 에피소드 메타를 데이터에 남깁니다.
        //
        //  ── episode_index / episode_total ──
        //   '이 리포트가 그 페이지의 몇 번째 의도 구간인가' 를 검색과 진단에서
        //   구분할 수 있게 합니다. 분할이 없었으면 0 / 1 이므로
        //   기존 리포트와 의미가 동일합니다.
        //
        //  ── episode_started_at / episode_ended_at ──
        //   구간의 실제 시간 범위입니다. 리포트 검색(STAGE-6)에서
        //   기간 조건과 대조할 때 created_at(합성 시각)이 아니라
        //   '행동이 일어난 시각' 으로 걸러야 정확합니다.
        //
        //  ⚠️ canonical.rs 의 kind_of 규칙상 '_at' 접미사는 수치(날짜)로,
        //     'index' 는 수치로 확정되므로 별도 정규화가 필요 없습니다.
        let ep_started = records
            .first()
            .and_then(|r| r.get("at").and_then(|v| v.as_i64()))
            .unwrap_or(now_ts);
        let ep_ended = records
            .last()
            .and_then(|r| r.get("at").and_then(|v| v.as_i64()))
            .unwrap_or(now_ts);
        // 🌟 [SDS / U-1 관측] 에피소드의 실제 시간 폭입니다.
        //  기획 U-3 의 시간 감쇠 반감기는 '경과 시간 분포의 중앙값' 에서 유도되는데,
        //  그 분포를 만들 관측이 지금까지 한 건도 없었습니다.
        //  여기서 쌓아 두면 U-3 착수 시점에 λ 를 상수 없이 확정할 수 있습니다.
        crate::utils::score_dynamics::record_baseline(
            "analytic.episode_span_ms",
            (ep_ended - ep_started).max(0) as f32,
        );
        let report_data = json!({
            "id": report_id.clone(),
            "type": "report",
            "mode": "analytic",
            "cross_action_flow": cross_action,
            "intent_evolution": intent_evo,
            "consistent_preferences": preferences,
            "link": records.first().and_then(|r| r.get("link")).cloned().unwrap_or(json!("")),
            "origin": env_doc_origin(&env_doc),
            "episode_index": ep_idx,
            "episode_total": segments.len(),
            "episode_started_at": ep_started,
            "episode_ended_at": ep_ended,
            "text": report_text.clone(),
            "masked_text": report_text,
            "created_at": now_ts,
            "updated_at": now_ts
        });

        let report_bcc = crate::utils::hash::hash_id(&format!("report{}", env_doc.cc));

        let _ = store
            .upsert_item(
                "items",
                &report_id,
                "report",
                report_data,
                None,
                None, // 🌟 vision_vec: report 경로는 비전 벡터 없음
                Some(&env_doc.from),
                Some(&env_doc.to),
                Some(&env_doc.cc),
                Some(&report_bcc),
                Some(&env_doc.r#ref),
                None,
            )
            .await;

        emit_term(&format!(
            "  📊 [ANALYTIC REPORT] id='{}'{} | 기록 {}건을 흐름 리포트로 합성했습니다.",
            report_id,
            if segments.len() > 1 { format!(" (에피소드 {}/{})", ep_idx + 1, segments.len()) } else { String::new() },
            records.len()
        ));
        } // for (ep_idx, (seg_start, seg_end)) in segments
    }
    // 🌟 [SDS] 구조화 태스크 경계에서 관측을 확정합니다.
    //
    //  ── 왜 여기인가 ──
    //   이 함수는 process_analytic_task 와 스케줄러 폴링 양쪽에서 호출되는데,
    //   폴링 경로에는 태스크 종료 훅이 없습니다. 함수 자신이 경계를 책임집니다.
    //
    //  ── 중간 return 을 덮지 못하지만 유실은 아닙니다 ──
    //   위쪽 조기 return(대상 0건 / 사전 필터 통과 0건 / 모델 확보 실패)은
    //   enter_scope 이전이라 남길 관측 자체가 없고,
    //   cancel·busy break 는 루프를 빠져나와 이 지점을 지나갑니다.
    //   설령 놓치더라도 DIRTY=true 로 메모리에 남아 다음 flush 가 기록합니다.
    emit_term(&format!("[ANALYTIC] {}", crate::utils::score_dynamics::report()));
    crate::utils::score_dynamics::flush();
    crate::utils::score_dynamics::leave_scope();

    emit_term(&format!(
        "[ANALYTIC] ✅ 구조화 완료: {}건. 로컬 임베딩 파이프라인이 이어서 벡터화합니다.",
        processed
    ));

    Ok(processed)
}

/// 🌟 [ORIGIN] 리포트에도 origin 을 남겨야 resolveAnalyticsOrigins 가 다음 라운드에서
///    같은 사이트를 계속 발견합니다. TradeDocument 에는 origin 이 없으므로 data 에서 읽습니다.
fn env_doc_origin(doc: &crate::store::TradeDocument) -> Value {
    if let Ok(d) = serde_json::from_str::<Value>(&doc.json_data) {
        if let Some(o) = d.get("origin").and_then(|v| v.as_str()) {
            return json!(o);
        }
    }
    json!("")
}

/// 🌟 [REPORT OUTPUT NORMALIZE] Qwen3.5 가 "Do NOT output JSON" 지시를 어기고
///  { "headline": ..., "supporting_actions": [...], "closing": ... } 형태로 반환하는 경우를 흡수합니다.
///  ── 왜 코드로 흡수하는가 ──
///   프롬프트 지시만으로는 2B 모델의 JSON 관성을 100% 막을 수 없고,
///   말풍선에 원시 JSON 이 그대로 노출되면 사용자가 읽을 수 없습니다.
///   구조가 무엇이든 '문자열 잎' 만 순서대로 펼치면 항상 읽을 수 있는 텍스트가 됩니다.
///  ── JSON 이 아니면 원문을 그대로 돌려줍니다 (무해). ──
pub fn normalize_report_output(raw: &str) -> String {
    let mut t = raw.trim().to_string();
    if t.is_empty() {
        return t;
    }

    // ① 코드펜스 제거
    if t.starts_with("```") {
        if let Some(p) = t.find('\n') {
            t = t[p + 1..].to_string();
        }
        if let Some(p) = t.rfind("```") {
            t = t[..p].to_string();
        }
        t = t.trim().to_string();
    }

    // ② JSON 형태가 아니면 그대로 반환
    if !(t.starts_with('{') || t.starts_with('[')) {
        return t;
    }

    let parsed = crate::parsing::parse_json_from_llm(&t);
    if parsed.is_null() {
        return t;
    }

    fn flatten(v: &Value, out: &mut Vec<String>, bullet: bool) {
        match v {
            Value::String(s) => {
                let x = s.trim();
                if x.is_empty() {
                    return;
                }
                out.push(if bullet { format!("- {}", x) } else { x.to_string() });
            }
            Value::Number(n) => {
                out.push(if bullet { format!("- {}", n) } else { n.to_string() });
            }
            Value::Bool(b) => {
                out.push(if bullet { format!("- {}", b) } else { b.to_string() });
            }
            Value::Array(a) => {
                for it in a {
                    flatten(it, out, true);
                }
            }
            Value::Object(o) => {
                for (_, val) in o {
                    flatten(val, out, bullet);
                }
            }
            _ => {}
        }
    }

    let mut lines: Vec<String> = Vec::new();
    if let Some(obj) = parsed.as_object() {
        for (_, v) in obj {
            let is_list = v.is_array();
            flatten(v, &mut lines, is_list);
        }
    } else {
        flatten(&parsed, &mut lines, false);
    }

    let joined = lines.join("\n").trim().to_string();
    if joined.is_empty() {
        t
    } else {
        println!("[ANALYTIC] 🧽 [REPORT NORMALIZE] JSON 응답을 읽을 수 있는 텍스트로 평탄화했습니다.");
        joined
    }
}

// =====================================================================
// 🌟 [ANALYTIC QUERY PARSER]
// ---------------------------------------------------------------------
//  parse_commerce_query 의 3단 구조를 분석 도메인에 맞춰 압축했습니다.
//    ① 결정론 시간 가이드 (get_deterministic_time_guide + exact_match_filter_key)
//    ② Qwen3.5 2B 의미 파싱 (기간 / 이벤트 종류 / 키워드)
//    ③ Rust 가 기간을 epoch ms 로 재확정 (LLM 이 계산한 날짜는 신뢰하지 않음)
//  ── 왜 ③이 필요한가 ──
//   commerce 의 extract_numeric_conditions 도 같은 이유로 시간 문맥을
//   프롬프트에 주입만 하고, 최종 SQL 은 코드가 만듭니다.
//   LLM 이 '이번달' 을 2026-03-01 로 적어도 그 값이 실제 epoch 인지 보증할 수 없기 때문입니다.
// =====================================================================

fn ms_of(y: i32, m: u32, d: u32) -> i64 {
    chrono::NaiveDate::from_ymd_opt(y, m, d)
        .and_then(|dd| dd.and_hms_opt(0, 0, 0))
        .map(|nd| nd.and_utc().timestamp_millis())
        .unwrap_or(0)
}

fn month_start(y: i32, m: u32) -> i64 { ms_of(y, m, 1) }

fn next_month(y: i32, m: u32) -> (i32, u32) {
    if m == 12 { (y + 1, 1) } else { (y, m + 1) }
}

fn prev_month(y: i32, m: u32) -> (i32, u32) {
    if m == 1 { (y - 1, 12) } else { (y, m - 1) }
}

pub fn ymd_of(ts: i64) -> (i32, u32, u32) {
    match chrono::DateTime::from_timestamp_millis(ts) {
        Some(dt) => {
            let d = dt.naive_utc().date();
            (chrono::Datelike::year(&d), chrono::Datelike::month(&d), chrono::Datelike::day(&d))
        },
        None => (1970, 1, 1),
    }
}

/// 🌟 [TIME RANGE] time_filters 캐노니컬 키를 epoch ms 구간으로 확정합니다.
pub fn time_intent_range(intent: &str, now_ms: i64) -> Option<(i64, i64)> {
    let (y, m, d) = ymd_of(now_ms);
    match intent {
        "today" => {
            let s = ms_of(y, m, d);
            Some((s, s + 86_400_000 - 1))
        },
        "yesterday" => {
            let s = ms_of(y, m, d) - 86_400_000;
            Some((s, s + 86_400_000 - 1))
        },
        "this_month" => {
            let s = month_start(y, m);
            let (ny, nm) = next_month(y, m);
            Some((s, month_start(ny, nm) - 1))
        },
        "last_month" => {
            let (py, pm) = prev_month(y, m);
            let s = month_start(py, pm);
            Some((s, month_start(y, m) - 1))
        },
        "this_year" => Some((ms_of(y, 1, 1), ms_of(y + 1, 1, 1) - 1)),
        "last_year" => Some((ms_of(y - 1, 1, 1), ms_of(y, 1, 1) - 1)),
        "recently" => Some((now_ms - 7 * 86_400_000, now_ms)),
        _ => None,
    }
}

/// 🌟 [SEASON RANGE] season_filters 캐노니컬 키를 해당 연도의 구간으로 확정합니다.
///    time_intent 가 과거를 가리키면 호출부가 year 를 이미 낮춰서 넘깁니다.
pub fn season_range(season: &str, year: i32) -> Option<(i64, i64)> {
    match season {
        "spring" => Some((ms_of(year, 3, 1), ms_of(year, 6, 1) - 1)),
        "summer" => Some((ms_of(year, 6, 1), ms_of(year, 9, 1) - 1)),
        "autumn" => Some((ms_of(year, 9, 1), ms_of(year, 12, 1) - 1)),
        "winter" => Some((ms_of(year, 12, 1), ms_of(year + 1, 3, 1) - 1)),
        _ => None,
    }
}

/// 🌟 [DETERMINISTIC TIME] bias.json 의 exact_match 배열로 완전일치 판정합니다.
///    '오늘' / 'today' / '今日' 처럼 50개 언어의 시간·계절 표현이 리터럴로 등재되어 있어
///    코드에 다국어 어휘를 하나도 넣지 않고 확정할 수 있습니다.
pub fn deterministic_time_keys(query: &str) -> (String, String) {
    let mut t_key = String::new();
    let mut s_key = String::new();
    // 공백 토큰 → 실패 시 전체 문자열까지 시도합니다. (한국어는 '이번달' 처럼 붙어 옵니다)
    let mut candidates: Vec<String> = query
        .split_whitespace()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    candidates.push(query.trim().to_string());

    // ── 1차 : 완전일치 ──
    for c in &candidates {
        if t_key.is_empty() {
            if let Some(k) = crate::utils::ai_utils::exact_match_filter_key("time_filters", c) {
                t_key = k;
            }
        }
        if s_key.is_empty() {
            if let Some(k) = crate::utils::ai_utils::exact_match_filter_key("season_filters", c) {
                s_key = k;
            }
        }
        if !t_key.is_empty() && !s_key.is_empty() { break; }
    }

    // 🌟 ── 2차 : 접두 일치 (교착어 대응) ──
    //    '올해는' / '여름에' 처럼 조사가 붙어 완전일치가 실패한 경우를 구제합니다.
    //    exact_match 원소가 토큰의 접두일 때만 인정하므로 어휘 하드코딩이 없습니다.
    if t_key.is_empty() || s_key.is_empty() {
        for c in &candidates {
            if t_key.is_empty() {
                if let Some((k, stem)) =
                    crate::utils::ai_utils::prefix_match_filter_stem("time_filters", c)
                {
                    println!("[ANALYTIC] 🕒 [TIME PREFIX MATCH] '{}' ← 접두 '{}' → time_filters.{}", c, stem, k);
                    t_key = k;
                }
            }
            if s_key.is_empty() {
                if let Some((k, stem)) =
                    crate::utils::ai_utils::prefix_match_filter_stem("season_filters", c)
                {
                    println!("[ANALYTIC] 🌤️ [SEASON PREFIX MATCH] '{}' ← 접두 '{}' → season_filters.{}", c, stem, k);
                    s_key = k;
                }
            }
            if !t_key.is_empty() && !s_key.is_empty() { break; }
        }
    }

    (t_key, s_key)
}

/// 🌟 [STANZA LANG CODE] parse_commerce_query 와 동일한 매핑입니다.
///  모델 디렉터리가 없으면 tokenize_query_with_pos 가 공백 분할로 폴백하므로,
///  여기서는 매핑만 담당하고 존재 여부는 검사하지 않습니다.
pub fn stanza_lang_code(language: &str) -> &'static str {
    match language {
        "korean" | "ko" => "ko",
        "english" | "en" => "en",
        "japanese" | "ja" => "ja",
        "chinese" | "zh" | "zh-hans" | "zh-hant" | "zh-tw" | "zh-hk" => "zh-hans",
        "french" | "fr" => "fr",
        "german" | "de" => "de",
        "spanish" | "es" => "es",
        "italian" | "it" => "it",
        "portuguese" | "pt" => "pt",
        "dutch" | "nl" => "nl",
        "russian" | "ru" => "ru",
        "arabic" | "ar" => "ar",
        "thai" | "th" => "th",
        "hindi" | "hi" => "hi",
        "bengali" | "bn" => "bn",
        "telugu" | "te" => "te",
        "khmer" | "km" => "km",
        "greek" | "el" => "el",
        "hebrew" | "he" => "he",
        "vietnamese" | "vi" => "vi",
        _ => "en",
    }
}

/// 🌟 [STANZA TOKENIZE + MORPHOLOGY] 질의를 어절 단위로 쪼개고 UPOS 태그와 Lemma 원형을 함께 부착합니다.
///  ── 왜 Lemma 까지 필요한가 ──
///   Stanza 토크나이저는 교착어 어절을 통째로 한 토큰으로 돌려줍니다.
///     "클릭한게" → 1토큰  (클릭 + 한 + 게)
///   이 표면형은 bias.json 의 exact_match("클릭")와 완전일치하지 않고,
///   영어 구 뱅크("clicked", "pressed")와의 코사인도 낮아
///   슬라이딩 윈도우에서 NMS 후보가 단 하나도 만들어지지 않습니다.
///   (로그 실측: [NMS CANDIDATE] 0건 → EVENT FALLBACK 4종 전체)
///   commerce 의 parse_commerce_query 는 이미 lemma_session 을 돌려
///   '가디건찾아줘' → '가디건' 절단을 수행하고 있으므로, 같은 장치를 이식합니다.
///
///  ── 반환 ──
///   (표면형, UPOS 태그, Lemma 원형). 실패 시 태그/Lemma 는 빈 문자열입니다.
pub async fn tokenize_query_with_morphology(
    query: &str,
    lang_code: &str,
) -> Vec<(String, String, String)> {
    let words: Vec<String> = query
        .split_whitespace()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let fallback: Vec<(String, String, String)> = words
        .iter()
        .map(|w| (w.clone(), String::new(), String::new()))
        .collect();
    if words.is_empty() {
        return fallback;
    }

    let base_dir = crate::utils::get_app_dir().join("models").join("stanza");
    let lang_dir = base_dir.join(lang_code);
    if !lang_dir.exists() {
        println!(
            "[ANALYTIC] ⚠️ Stanza 모델 디렉터리가 없어 공백 분할로 폴백합니다: {:?}",
            lang_dir
        );
        return fallback;
    }

    struct UnsafePipelineWrapper(crate::stanza::StanzaPipeline);
    unsafe impl Send for UnsafePipelineWrapper {}

    let wrapper = match crate::stanza::StanzaPipeline::new(base_dir, lang_code).await {
        Ok(p) => UnsafePipelineWrapper(p),
        Err(e) => {
            println!("[ANALYTIC] ⚠️ Stanza 로드 실패({:?}). 공백 분할로 폴백합니다.", e);
            return fallback;
        }
    };
    let mut stanza = wrapper.0;

    let refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();

    // ONNX Export 시 고정된 시퀀스 길이를 그대로 존중합니다.
    let mut chunk_size = refs.len();
    for input_meta in &stanza.pos_session.inputs {
        let dims = &input_meta.dimensions;
        if dims.len() == 2 && dims.get(1) == Some(&Some(32)) {
            if let Some(&Some(fixed_seq)) = dims.get(0) {
                chunk_size = fixed_seq as usize;
            }
        }
    }
    if chunk_size == 0 {
        chunk_size = refs.len();
    }
    let mut padded = refs.clone();
    let valid_len = padded.len();
    while padded.len() < chunk_size {
        padded.push("<pad>");
    }

    let inputs = match stanza
        .preprocessor
        .encode_to_tensor(&padded, &stanza.pos_session, None, None)
    {
        Ok(v) => v,
        Err(e) => {
            println!("[ANALYTIC] ⚠️ Stanza encode 실패({:?}). 공백 분할로 폴백합니다.", e);
            return fallback;
        }
    };

    let outputs = match stanza.pos_session.run::<'_, '_, '_, i64, f32, _>(inputs) {
        Ok(v) => v,
        Err(e) => {
            println!("[ANALYTIC] ⚠️ Stanza POS 추론 실패({:?}). 공백 분할로 폴백합니다.", e);
            return fallback;
        }
    };

    let t = &outputs[0];
    let shape = t.shape();
    if shape.len() < 2 {
        return fallback;
    }
    let num_classes = if shape.len() == 3 {
        shape[2] as usize
    } else {
        shape[1] as usize
    };

    // ── ① UPOS 디코드 + Lemma 세션 입력용 POS ID 수집 ──
    let mut tags: Vec<String> = Vec::with_capacity(valid_len);
    let mut pos_ids: Vec<i64> = Vec::with_capacity(valid_len);
    for i in 0..valid_len {
        let mut max_val = f32::MIN;
        let mut max_idx = 0usize;
        for c in 0..num_classes {
            let v = if shape.len() == 3 { t[[0, i, c]] } else { t[[i, c]] };
            if v > max_val {
                max_val = v;
                max_idx = c;
            }
        }
        tags.push(
            stanza
                .preprocessor
                .upos_vocab
                .get(max_idx)
                .cloned()
                .unwrap_or_else(|| "X".to_string()),
        );
        pos_ids.push(max_idx as i64);
    }

    // ── ② Lemma 디코드 (commerce parse_commerce_query 와 동일 로직) ──
    let mut lemmas: Vec<String> = vec![String::new(); valid_len];
    if let Ok(lemma_inputs) = stanza.preprocessor.encode_to_tensor(
        &padded,
        &stanza.lemma_session,
        Some(&pos_ids),
        None,
    ) {
        if let Ok(lemma_outputs) = stanza
            .lemma_session
            .run::<'_, '_, '_, i64, f32, _>(lemma_inputs)
        {
            let lt = &lemma_outputs[0];
            let ls = lt.shape();
            if ls.len() == 3 || ls.len() == 4 {
                let is_4d = ls.len() == 4;
                let max_char_len = if is_4d { ls[2] as usize } else { ls[1] as usize };
                let lemma_classes = if is_4d { ls[3] as usize } else { ls[2] as usize };
                for i in 0..valid_len {
                    let mut lemma_str = String::new();
                    for j in 0..max_char_len {
                        let mut mv = f32::MIN;
                        let mut mi = 0usize;
                        for c in 0..lemma_classes {
                            let v = if is_4d { lt[[0, i, j, c]] } else { lt[[i, j, c]] };
                            if v > mv {
                                mv = v;
                                mi = c;
                            }
                        }
                        if let Some(&ch) = stanza.preprocessor.id_to_char.get(&(mi as i64)) {
                            if ch != '<' && ch != '>' && ch != '_' {
                                lemma_str.push(ch);
                            }
                        }
                    }
                    lemmas[i] = lemma_str.trim().to_string();
                }
            }
        }
    }

    let mut out: Vec<(String, String, String)> = Vec::with_capacity(valid_len);
    for i in 0..valid_len {
        out.push((words[i].clone(), tags[i].clone(), lemmas[i].clone()));
    }
    out
}

/// 🌟 [ANALYTIC SEARCH QUERY] #global-search 의 자연어 질의를 검색 컨텍스트로 변환합니다.
///  반환 형태는 lib.rs 의 STAGE-3 컨텍스트 계약과 동일하므로
///  build_scope_filter / build_dexie_plan / STAGE-4 를 그대로 재사용합니다.
pub async fn parse_analytic_search_query(
    task_id: &str,
    app_handle: &tauri::AppHandle,
    model: &LogisModel,
    query: String,
    language: &str,
    cancel: Arc<AtomicBool>,
) -> Result<Value> {
    let app_handle_clone = app_handle.clone();
    let tid = task_id.to_string();
    let emit_term = move |msg: &str| {
        println!("{}", msg);
        let _ = app_handle_clone.emit(
            "task-console-log",
            json!({ "task_id": tid, "text": format!("{}\n", msg) })
        );
    };

    emit_term("\n[ANALYTIC-QUERY] 🔍 행동 로그 질의 파싱 시작");
    emit_term(&format!("  질의: \"{}\"", query));
    // 🌟 [SDS SCOPE] 검색 경로는 process_task 를 거치지 않으므로 자체 스코프를 세웁니다.
    //    팀 식별자는 이 함수 시그니처에 없으므로 SDS 가 로드 시점에 바인딩한
    //    팀을 그대로 사용합니다(빈 문자열을 넘기면 기존 바인딩이 유지됩니다).
    crate::utils::score_dynamics::enter_scope(
        "",
        crate::utils::score_dynamics::Track::Analytic,
        "query",
        "",
    );
    let now_ms = chrono::Utc::now().timestamp_millis();
    let current_iso = chrono::DateTime::from_timestamp_millis(now_ms)
        .map(|dt| dt.naive_utc().format("%Y-%m-%dT%H:%M:%S").to_string())
        .unwrap_or_default();

    // ── ① 결정론 시간 가이드 ──
    let (deterministic_time, _) = crate::parsing::get_deterministic_time_guide(&query, language);
    let (det_time_key, det_season_key) = deterministic_time_keys(&query);

    if !det_time_key.is_empty() || !det_season_key.is_empty() {
        emit_term(&format!(
            "  ⚡ [EXACT MATCH] bias.json 완전일치로 확정: time='{}' | season='{}'",
            det_time_key, det_season_key
        ));
    }

    let time_context = format!(
        "- Current UTC time is \"{}\" (epoch ms {}).\n- The user locale language is \"{}\".\n{}",
        current_iso, now_ms, language, deterministic_time
    );

    // ── ② Qwen3.5 2B 의미 파싱 ──
    model
        .secure_vram_relay(
            crate::model::ModelSize::Qwen3_5,
            None,
            Some(cancel.clone()),
            false,
            None,
        )
        .await?;

    let prompt = crate::prompts::analytic_query_prompt(&query, &time_context, language);

    let params = crate::openai_types::ChatCompletionParameters {
        messages: vec![
            crate::openai_types::ChatCompletionRequestMessage::User(
                crate::openai_types::ChatCompletionRequestUserMessage {
                    content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(prompt),
                    name: None,
                }
            )
        ],
        model: "qwen3.5".to_string(),
        max_tokens: Some(512),
        temperature: Some(0.0),
        top_p: Some(0.95),
        ..Default::default()
    };

    let res_text = if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
        gen.generate(
            params,
            Some(cancel.clone()),
            Some(format!("{}_aq", task_id)),
            None,
            None,
            None,
        )
        .await
        .unwrap_or_default()
    } else {
        String::new()
    };

    let parsed = crate::parsing::parse_json_from_llm(&res_text);

    // 🌟 [EMBEDDING-BASED TIME/EVENT SELECTION]
    //    LLM 단독 결정 대신, 임베딩 코사인 + NMS 경쟁으로 선택합니다.
    //    ① bias.json 의 time_filters / season_filters / analytic_event_filters 구를 임베딩
    //    ② 질의 임베딩과 각 구 간 코사인 계산
    //    ③ SURPRISAL 게이트로 우연 공명 제거
    //    ④ 마진 부족 시에만 LLM 재판정 (기존 경로 유지)

    // ── ① 질의 임베딩 ──
    let query_emb = model.get_embedding(query.clone()).await.unwrap_or(vec![0.0; 384]);

    // ── ② time_filters / season_filters 뱅크 임베딩 ──
    let time_phrases = crate::utils::ai_utils::filter_category_phrases(&["time_filters"]);
    let season_phrases = crate::utils::ai_utils::filter_category_phrases(&["season_filters"]);
    let time_prej_phrases = crate::utils::ai_utils::filter_category_prejudice_phrases(&["time_filters"]);
    let season_prej_phrases = crate::utils::ai_utils::filter_category_prejudice_phrases(&["season_filters"]);

    let time_texts: Vec<String> = time_phrases.iter().map(|(_, _, p)| p.clone()).collect();
    let season_texts: Vec<String> = season_phrases.iter().map(|(_, _, p)| p.clone()).collect();
    let time_prej_texts: Vec<String> = time_prej_phrases.iter().map(|(_, _, p)| p.clone()).collect();
    let season_prej_texts: Vec<String> = season_prej_phrases.iter().map(|(_, _, p)| p.clone()).collect();

    let time_embs: Vec<Vec<f32>> = if time_texts.is_empty() { Vec::new() } else {
        model.get_embedding_batch(time_texts.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; time_texts.len()])
    };
    let season_embs: Vec<Vec<f32>> = if season_texts.is_empty() { Vec::new() } else {
        model.get_embedding_batch(season_texts.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; season_texts.len()])
    };
    let time_prej_embs: Vec<Vec<f32>> = if time_prej_texts.is_empty() { Vec::new() } else {
        model.get_embedding_batch(time_prej_texts.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; time_prej_texts.len()])
    };
    let season_prej_embs: Vec<Vec<f32>> = if season_prej_texts.is_empty() { Vec::new() } else {
        model.get_embedding_batch(season_prej_texts.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; season_prej_texts.len()])
    };

    // ── ③ SURPRISAL 게이트 + 뱅크 크기 편향 제거 ──
    let surprisal_score = |q: &Vec<f32>, idxs: &[usize], embs: &Vec<Vec<f32>>| -> (f32, f32) {
        let mut sims: Vec<f32> = Vec::new();
        for &i in idxs {
            if let Some(e) = embs.get(i) {
                if !e.iter().all(|&v| v == 0.0) {
                    sims.push(crate::utils::ai_utils::cosine_similarity(q, e));
                }
            }
        }
        if sims.is_empty() { return (f32::MIN, 0.0); }
        let n = sims.len() as f32;
        let mean: f32 = sims.iter().sum::<f32>() / n;
        let var: f32 = sims.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n;
        let sd = var.sqrt().max(1e-6);
        let mx = sims.iter().cloned().fold(f32::MIN, f32::max);
        let z = (mx - mean) / sd;
        let expect = (2.0 * n.max(2.0).ln()).sqrt();
        (z - expect, mx)
    };

    // time_filters 키별 인덱스 매핑
    let time_key_indices: Vec<(String, Vec<usize>)> = {
        let mut map: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
        for (i, (_, key, _)) in time_phrases.iter().enumerate() {
            map.entry(key.clone()).or_default().push(i);
        }
        map.into_iter().collect()
    };
    let season_key_indices: Vec<(String, Vec<usize>)> = {
        let mut map: std::collections::HashMap<String, Vec<usize>> = std::collections::HashMap::new();
        for (i, (_, key, _)) in season_phrases.iter().enumerate() {
            map.entry(key.clone()).or_default().push(i);
        }
        map.into_iter().collect()
    };

    // ── time_intent 임베딩 판정 ──
    let mut time_intent = String::new();
    let mut time_score = f32::MIN;
    for (key, idxs) in &time_key_indices {
        // 🌟 max_cos 는 진단용이며 판정에는 surprisal 만 씁니다.
        let (sur, _max_cos) = surprisal_score(&query_emb, idxs, &time_embs);
        if sur > time_score { time_score = sur; time_intent = key.clone(); }
    }
    // 편견 게이트: 경쟁 개념이 더 잘 설명하면 폐기
    if !time_prej_embs.is_empty() && !time_intent.is_empty() {
        let prej_score = crate::utils::ai_utils::max_pool_sim(&query_emb, &time_prej_embs);
        let own_score = time_embs.iter()
            .map(|e| crate::utils::ai_utils::cosine_similarity(&query_emb, e))
            .fold(f32::MIN, f32::max);
        if prej_score >= own_score {
            emit_term(&format!(
                "  🚫 [TIME PREJ GATE] time_intent='{}' 폐기 (prej {:.4} >= own {:.4})",
                time_intent, prej_score, own_score
            ));
            time_intent = String::new();
            time_score = f32::MIN;
        }
    }

    // ── season_intent 임베딩 판정 ──
    let mut season_intent = String::new();
    let mut season_score = f32::MIN;
    for (key, idxs) in &season_key_indices {
        let (sur, mx) = surprisal_score(&query_emb, idxs, &season_embs);
        if sur > season_score { season_score = sur; season_intent = key.clone(); }
    }
    if !season_prej_embs.is_empty() && !season_intent.is_empty() {
        let prej_score = crate::utils::ai_utils::max_pool_sim(&query_emb, &season_prej_embs);
        let own_score = season_embs.iter()
            .map(|e| crate::utils::ai_utils::cosine_similarity(&query_emb, e))
            .fold(f32::MIN, f32::max);
        if prej_score >= own_score {
            emit_term(&format!(
                "  🚫 [SEASON PREJ GATE] season_intent='{}' 폐기 (prej {:.4} >= own {:.4})",
                season_intent, prej_score, own_score
            ));
            season_intent = String::new();
            season_score = f32::MIN;
        }
    }

    // ── exact_match 가 있으면 임베딩 판정보다 우선 ──
    if !det_time_key.is_empty() { time_intent = det_time_key.clone(); }
    if !det_season_key.is_empty() { season_intent = det_season_key.clone(); }

    // ── 마진 부족 시에만 LLM 재판정 (기존 경로 유지) ──
    let need_time_llm = time_intent.is_empty() && det_time_key.is_empty() && time_score > -1.0;
    let need_season_llm = season_intent.is_empty() && det_season_key.is_empty() && season_score > -1.0;
    if need_time_llm || need_season_llm {
        emit_term("  ⚖️ [EMBED→LLM FALLBACK] 임베딩 마진 부족. LLM 재판정 수행.");
        // 기존 LLM 경로 (parsed 에서 가져오기)
        if need_time_llm {
            time_intent = parsed.get("time_intent").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        }
        if need_season_llm {
            season_intent = parsed.get("season_intent").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        }
    }

    emit_term(&format!(
        "  🕒 [EMBED-BASED TIME] time_intent='{}' (score {:+.4}) | season_intent='{}' (score {:+.4})",
        if time_intent.is_empty() { "-" } else { &time_intent }, time_score,
        if season_intent.is_empty() { "-" } else { &season_intent }, season_score
    ));

    // ── event_types NMS 배틀 결과 확정 ──
    // 슬라이딩 윈도우 + NMS 배틀에서 생존한 스팬 중 이벤트 타입만 추출합니다.
    // 기존 방식은 질의 전체 벡터 1개로 판정하여
    // '클릭한거 뭐야' 에서 '클릭' 이 '뭐야' 와 섞여 신호가 희석되는 문제가 있었습니다.
    // NMS 배틀은 각 단어 윈도우를 독립적으로 경쟁시키므로 이 문제가 없습니다.
    //
    // 🌟 [구조 설명]
    //   수정 1 에서 bank_defs 에 ("event", "click", 구) 등을 추가했으므로
    //   슬라이딩 윈도우 → SURPRISAL 채점 → NMS 배틀 파이프라인이
    //   time/season 과 동일하게 이벤트 타입도 처리합니다.
    //   카테고리별 확정 루프의 "event" 분기가 이미
    //   vec_events: Vec<(String, f32)> 를 채우고 있으므로
    //   여기서 그 결과를 그대로 소비합니다.
    let mut event_types: Vec<String> = Vec::new();
    // 🌟 [SDS 계측] 이벤트 타입 판정의 own / 차항 배열을 각각 모읍니다.
    //
    //  ── 왜 두 축을 분리해 모으는가 ──
    //   논의 4회차의 통합결론 1번이 "bias 단독 트레이스는 리콜 게이트,
    //   bias−prejudice 트레이스는 정밀도 게이트로 역할을 분리하라" 였습니다.
    //   그 분리(T-9)를 Phase 2 에서 실행하려면
    //   두 축의 분포가 실제로 다른 형상을 갖는지가 먼저 관측되어야 합니다.
    //   지금은 score = own - prej 하나로 합쳐져 있어 확인할 방법이 없습니다.
    let mut sds_own: Vec<f32> = Vec::new();
    let mut sds_net: Vec<f32> = Vec::new();
    for event_type in crate::analytic::ANALYTIC_SEARCH_TYPES.iter() {
        let anchor_phrases = crate::analytic::event_type_anchor_phrases(event_type);
        let prej_phrases = crate::analytic::event_type_prejudice_phrases(event_type);
        if anchor_phrases.is_empty() { continue; }
        let a_embs = model.get_embedding_batch(anchor_phrases.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; anchor_phrases.len()]);
        let p_embs = if prej_phrases.is_empty() { Vec::new() } else {
            model.get_embedding_batch(prej_phrases.clone()).await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; prej_phrases.len()])
        };
        let own = crate::utils::ai_utils::max_pool_sim(&query_emb, &a_embs);
        let prej = if p_embs.is_empty() { 0.0 } else {
            crate::utils::ai_utils::max_pool_sim(&query_emb, &p_embs)
        };
        let score = own - prej;
        emit_term(&format!(
            "  🎯 [EVENT NMS] '{}' | own: {:.4} | prej: {:.4} | score: {:+.4}",
            event_type, own, prej, score
        ));
        sds_own.push(own);
        sds_net.push(score);
        if score > 0.0 {
            event_types.push(event_type.to_string());
        }
    }
    // 🌟 [SDS 계측] 두 축의 감쇠 형상을 별도 축으로 남깁니다.
    if sds_own.len() >= 2 {
        crate::utils::score_dynamics::record_decay("analytic.event.bias_only", &sds_own);
        crate::utils::score_dynamics::record_decay("analytic.event.net", &sds_net);
    }
    // report 는 합성 문서이므로 항상 포함
    if !event_types.iter().any(|t| t == "report") {
        event_types.push("report".to_string());
    }
    // 전부 탈락하면 전체 타입을 스코프로 (리콜 보존)
    if event_types.iter().filter(|t| *t != "report").count() == 0 {
        event_types = ANALYTIC_SEARCH_TYPES.iter().map(|s| s.to_string()).collect();
        emit_term("  🛟 [EVENT FALLBACK] NMS 배틀에서 확정된 이벤트 타입이 없어 전체 타입을 스코프로 둡니다.");
    }
    emit_term(&format!("  ✅ [EVENT TYPES FINAL] {:?}", event_types));

    let keywords: Vec<String> = parsed
        .get("keywords")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_str().map(|s| s.trim().to_string()))
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let target = parsed
        .get("target")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| {
            if keywords.is_empty() { query.clone() } else { keywords.join(" ") }
        });
    // ── ③ 기간을 Rust 가 재확정 ──
    let mut started_at: i64 = 0;
    let mut expired_at: i64 = 0;

    if !season_intent.is_empty() {
        // 계절은 연도가 필요합니다. time_intent 가 과거를 가리키면 작년으로 내립니다.
        let (y, _, _) = ymd_of(now_ms);
        let year = if time_intent == "last_year" { y - 1 } else { y };
        if let Some((s, e)) = season_range(&season_intent, year) {
            started_at = s;
            expired_at = e;
        }
    }

    if started_at == 0 {
        if let Some((s, e)) = time_intent_range(&time_intent, now_ms) {
            started_at = s;
            expired_at = e;
        }
    }

    if started_at > 0 {
        emit_term(&format!(
            "  🗓️ [PERIOD CONFIRMED] time='{}' | season='{}' → {} ~ {} (epoch ms)",
            if time_intent.is_empty() { "-" } else { &time_intent },
            if season_intent.is_empty() { "-" } else { &season_intent },
            started_at, expired_at
        ));
    } else {
        emit_term("  🗓️ [PERIOD] 명시적 기간 표현이 없어 전체 구간을 검색합니다.");
    }

    // ── ④ 컨텍스트 조립 (lib.rs STAGE-3 계약과 동일) ──
    let mut condition = serde_json::Map::new();
    if started_at > 0 {
        condition.insert(
            "created_at".to_string(),
            json!({ "operator": "gte", "value": started_at })
        );
    }
    // 🌟 [상한 처리 위치] build_scope_filter 는 키 하나에 연산자 하나만 받으므로
    //    created_at 의 lte 는 여기서 넣지 않고, 아래에서 별도 컨텍스트로 분리합니다.
    //    (기존에는 이 자리에 본문 없는 if 블록만 남아 있었습니다)

    let mut contexts: Vec<Value> = Vec::new();

    contexts.push(json!({
        "text": target,
        "language": language,
        "type": event_types[0],
        "types": event_types,
        "condition": Value::Object(condition.clone()),
        "unassigned": keywords
    }));

    // 🌟 상한(lte)은 별도 컨텍스트로 분리해 build_scope_filter 가
    //    created_at <= X 를 SQL 로 함께 내려보내도록 합니다.
    if expired_at > 0 {
        let mut upper = serde_json::Map::new();
        upper.insert(
            "created_at".to_string(),
            json!({ "operator": "lte", "value": expired_at })
        );
        contexts.push(json!({
            "text": target,
            "language": language,
            "type": event_types[0],
            "types": event_types,
            "condition": Value::Object(upper),
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

    // 🌟 [SDS] 검색 경로는 enter_scope 만 있고 해제가 없었습니다.
    //
    //  ── 왜 위험한가 ──
    //   스코프가 살아 있는 채로 함수를 빠져나가면, 이어서 돌아가는
    //   백그라운드 인덱싱·구조화의 관측이 'analytic|query|' 로 잘못 귀속됩니다.
    //   Track 이 Analytic 으로 굳으면 Track::min_obs() 도 analytic 기준(40/15/0)이 적용되어
    //   커머스 통계가 영원히 발동 임계에 도달하지 못합니다.
    //
    //  ── 중간 `?` 조기 반환은 덮지 못합니다 ──
    //   secure_vram_relay 실패는 에러 경로라 여기에 도달하지 않지만,
    //   그 경우 다음 태스크의 enter_scope 가 스코프를 덮어쓰므로 오염이 1회로 끝납니다.
    emit_term(&format!("[ANALYTIC-QUERY] {}", crate::utils::score_dynamics::report()));
    crate::utils::score_dynamics::flush();
    crate::utils::score_dynamics::leave_scope();

    Ok(out)
}

// =====================================================================
// 🌟 [TASK ENTRY] 스케줄러에서 analytic_extraction 태스크로 들어오는 경로
// =====================================================================
pub async fn process_analytic_task(
    task: Task,
    store_mutex: &Arc<Mutex<Option<VectorStore>>>,
    model_mutex: &Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: &Arc<AtomicBool>,
    app_handle: &tauri::AppHandle,
    device_preference: Option<String>,
) -> Result<()> {
    let app_handle_clone = app_handle.clone();
    let tid_clone = task.id.clone();
    let emit_term = move |msg: &str| {
        println!("{}", msg);
        let _ = app_handle_clone.emit("task-console-log", json!({"task_id": tid_clone, "text": format!("{}\n", msg)}));
    };

    emit_term("[Scheduler] Starting Analytic Structuring Pipeline...");
    let list_log = json!({ "category": "Analytic Processing", "summary": "Analyzing user behavior logs...", "spinner": "⠋" });
    log_task_progress(app_handle, &task.id, &list_log);

    let store = {
        let store_guard = store_mutex.lock().await;
        store_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Store not initialized"))?.clone()
    };

    let model = {
        let mut model_lock = model_mutex.lock().await;
        if model_lock.is_none() {
            *model_lock = Some(LogisModel::new(app_handle.clone(), device_preference.as_deref()).await.map_err(|e| anyhow::anyhow!(e))?);
        }
        model_lock.as_ref().unwrap().clone()
    };

    let processed = run_analytic_structuring(
        &store,
        &model,
        cancellation_token,
        app_handle,
        &task.id,
        60,
    ).await.unwrap_or(0);

    model.deep_purge_resources().await;

    let store_guard = store_mutex.lock().await;
    if let Some(db) = store_guard.as_ref() {
        let _ = db.update_task_status(&task.id, 9).await;
        let _ = db.update_message_status(&task.id, 9, Some("Analytic Structuring Complete")).await;
    }
    drop(store_guard);

    let payload = json!({
        "task_id": task.id,
        "category": "Done",
        "summary": if processed > 0 {
            format!("{} behaviour event(s) structured.", processed)
        } else {
            "No pending analytic logs found.".to_string()
        },
        "spinner": "✅",
        "data": null
    });
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);

    if let Ok(mut w) = crate::ACTIVE_TASK_MEM.write() { *w = None; }
    Ok(())
}
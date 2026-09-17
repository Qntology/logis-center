use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use tokio::sync::Mutex;
use serde_json::{json, Value};
use crate::store::VectorStore;
use crate::model::LogisModel;
use crate::scheduler::{Task, PugMode, normalize_entity_key, entity_index, entity_id, entity_bcc, index_item_chunks};
use crate::scheduler::indexing::save_item;
use crate::utils::logger::log_task_progress;
use crate::parsing;
use tauri::Emitter;
use crate::logic::TRADE_DOC_TITLES;

const TRADE_TITLE_LABEL_ANCHOR: &str = "document type, kind of document, type of form, \
     name of this document, title of this document, document name, form name, \
     document code, form code, classification of this document";
const TRADE_REFERENCE_LABEL_ANCHOR: &str = "referenced document number, related document number, \
     reference number of another document, master document number, associated document, \
     payment terms, terms of payment, drawn under credit, issued under, \
     attached documents, required documents, enclosed documents, remark, note";

const TRADE_ITEM_ATTRIBUTE_ANCHOR: &str = "line item attribute, attribute of one product row, \
     item code, stock keeping unit, article number, product description, \
     quantity, unit of measure, unit price, line total, amount of this row, \
     table column header, row number in a list, subtotal, discount, total quantity";

const TRADE_ROW_MARKER_ANCHOR: &str = "row separator, table row marker, line item index, \
     item number in a list, section key, group key, metadata key, \
     continued from previous page, page break marker, list bullet";

const SITE_CHROME_ANCHOR: &str = "site name, shopping mall name, brand slogan, \
     administrator page, admin home, admin main menu, management menu, \
     dashboard, control panel, back office, console, \
     global navigation bar, breadcrumb, sidebar menu, footer, copyright notice, banner, \
     login, logout, sign in, sign out, my page, member management, \
     settings, configuration, preferences, \
     visitor counter, today visitors, yesterday visitors, total visitors, \
     software version number, welcome message, home, index page, \
     search form, filter form, page navigation, pagination";

#[derive(Debug, Clone)]
struct TitleCandidate {
    label: String,
    value: String,
    line: usize,
}

fn collect_title_candidates(pug: &str, band_ratio: f32) -> Vec<TitleCandidate> {
    let all: Vec<&str> = pug.lines().collect();
    if all.is_empty() { return Vec::new(); }
    // 거대 문서에서 O(n²) 페어 수집이 폭주하지 않도록 스캔 상한을 둡니다.
    let scan_cap = all.len().min(2000);
    let scan: Vec<&str> = all[..scan_cap].to_vec();
    let band = (((scan_cap as f32) * band_ratio).ceil() as usize)
        .max(1)
        .min(scan_cap);
    let pairs = crate::utils::ai_utils::collect_detail_label_value_pairs(&scan);
    let mut consumed: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for p in pairs.iter() {
        consumed.insert(p.primary_line);
        consumed.insert(p.label_line);
    }
    let mut out: Vec<TitleCandidate> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    // ① 라벨-값 페어
    for p in pairs.iter() {
        if p.label_line >= band && p.primary_line >= band { continue; }
        let v = p.value.trim().to_string();
        if v.chars().count() < 2 { continue; }
        if !seen.insert(v.clone()) { continue; }
        out.push(TitleCandidate {
            label: p.label.trim().to_string(),
            value: v,
            line: p.primary_line,
        });
    }
    // ② 라벨 없는 헤딩 라인
    for i in 0..band {
        if consumed.contains(&i) { continue; }
        let (_, tag, _, value) = crate::utils::ai_utils::pug_line_parts(scan[i]);
        if matches!(
            tag.as_str(),
            "tr" | "table" | "thead" | "tbody" | "tfoot" | "colgroup" | "col" | "form" | "button"
        ) {
            continue;
        }
        let v = value.trim().to_string();
        if v.chars().count() < 2 { continue; }
        if !seen.insert(v.clone()) { continue; }
        out.push(TitleCandidate { label: String::new(), value: v, line: i });
    }
    out.sort_by_key(|c| c.line);
    if out.len() > 24 { out.truncate(24); }
    out
}
pub(crate) async fn resolve_title_values_weighted(
    model: &LogisModel,
    light_pug: &str,
    band_ratio: f32,
    emit_term: &(dyn Fn(&str) + Send + Sync),
    verbose: bool,
) -> Vec<(String, f32)> {
    let cands = collect_title_candidates(light_pug, band_ratio);
    if verbose {
        emit_term(&format!("  🪪 [TITLE CANDIDATES] 상단 밴드 표제 후보 {}개 수집", cands.len()));
    }
    if cands.is_empty() { return Vec::new(); }
    let humanize = |raw: &str| -> String {
        let h = crate::utils::ai_utils::humanize_url_token(raw);
        if h.trim().is_empty() { raw.trim().to_string() } else { h }
    };
    let anchors = model
        .get_embedding_batch(vec![
            TRADE_TITLE_LABEL_ANCHOR.to_string(),        // 0 자기선언
            TRADE_REFERENCE_LABEL_ANCHOR.to_string(),    // 1 타문서참조
            TRADE_ITEM_ATTRIBUTE_ANCHOR.to_string(),     // 2 품목속성
            TRADE_ROW_MARKER_ANCHOR.to_string(),         // 3 행구분자
            SITE_CHROME_ANCHOR.to_string(),              // 4 사이트 껍데기
            crate::logic::UI_ACTION_ANCHOR.to_string(),  // 5 UI 액션
        ])
        .await
        .unwrap_or_else(|_| vec![vec![0.0; 384]; 6]);
    // ── 라벨 축 ──
    let mut label_texts: Vec<String> = Vec::new();
    for c in cands.iter() {
        if c.label.trim().is_empty() { continue; }
        let t = humanize(&c.label);
        if !label_texts.iter().any(|e| e == &t) { label_texts.push(t); }
    }
    let label_embs = if label_texts.is_empty() {
        Vec::new()
    } else {
        model.get_embedding_batch(label_texts.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; label_texts.len()])
    };
    let mut label_is_title: std::collections::HashMap<String, bool> =
        std::collections::HashMap::new();
    for (li, lt) in label_texts.iter().enumerate() {
        let ts = crate::utils::ai_utils::cosine_similarity(&label_embs[li], &anchors[0]);
        let rs = crate::utils::ai_utils::cosine_similarity(&label_embs[li], &anchors[1]);
        let is = crate::utils::ai_utils::cosine_similarity(&label_embs[li], &anchors[2]);
        let ch = crate::utils::ai_utils::cosine_similarity(&label_embs[li], &anchors[4]);
        let ac = crate::utils::ai_utils::cosine_similarity(&label_embs[li], &anchors[5]);
        // 🌟 [ALL-AXIS RIVAL] 2지선다가 아니라 '나머지 전 축의 최댓값' 과 겨룹니다.
        let rival = rs.max(is).max(ch).max(ac);
        let keep = ts > rival;
        label_is_title.insert(lt.clone(), keep);
        if verbose {
            let why = if keep {
                "표제 후보 유지"
            } else if ch >= ts && ch >= rs && ch >= is && ch >= ac {
                "사이트 껍데기 → 제외"
            } else if ac >= ts && ac >= rs && ac >= is {
                "UI 액션 → 제외"
            } else if is >= ts {
                "품목 속성 → 제외"
            } else {
                "참조 라벨 → 제외"
            };
            emit_term(&format!(
                "     🏷️ [TITLE LABEL GATE] '{}' | 자기선언 {:.4} vs 타문서참조 {:.4} vs 품목속성 {:.4} vs 껍데기 {:.4} vs 액션 {:.4} → {}",
                lt, ts, rs, is, ch, ac, why
            ));
        }
    }
    // ── 값 축 ──
    let mut headless: Vec<String> = Vec::new();
    // 🌟 [T-1] (값, 가중치) 로 승격합니다.
    //    라벨 축을 통과한 후보는 바닥 판정을 거치지 않으므로 항상 1.0 입니다.
    let mut kept: Vec<(String, f32)> = Vec::new();
    for c in cands.iter() {
        if c.label.trim().is_empty() {
            if !headless.iter().any(|e| e == &c.value) { headless.push(c.value.clone()); }
            continue;
        }
        if label_is_title.get(&humanize(&c.label)).copied().unwrap_or(false) {
            if !kept.iter().any(|(e, _)| e == &c.value) { kept.push((c.value.clone(), 1.0)); }
        }
    }
    if !headless.is_empty() {
        let he = model.get_embedding_batch(headless.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; headless.len()]);
        let mut ts_all: Vec<f32> = Vec::with_capacity(headless.len());
        for i in 0..headless.len() {
            if he[i].iter().all(|&x| x == 0.0) {
                ts_all.push(f32::MIN);
            } else {
                ts_all.push(crate::utils::ai_utils::cosine_similarity(&he[i], &anchors[0]));
            }
        }
        let floor = {
            let v: Vec<f32> = ts_all.iter().cloned().filter(|s| *s > f32::MIN).collect();
            if v.len() < 4 {
                0.0f32
            } else {
                v.iter().sum::<f32>() / (v.len() as f32)
            }
        };
        if verbose {
            emit_term(&format!(
                "     📏 [TITLE FLOOR] 라벨 없는 후보 {}개의 자기선언 분포에서 유도한 바닥 = 평균 = {:.4} (리콜 우선: 상대 5축 비교가 주 판정, 바닥은 하한선)",
                ts_all.iter().filter(|s| **s > f32::MIN).count(), floor
            ));
        }
        // 🌟 [SDS 계측] 표제 바닥과 자기선언 점수 분포를 관측으로 남깁니다.
        //
        //  ── 표적 사고 ──
        //   SA 문서에서 'document_type: Shipping Advice' 가 이 바닥(0.5457)에
        //   미달해 탈락했고, 그 결과 제목 축 정보가 0 으로 수렴했습니다.
        //   문제는 '점수가 낮다' 가 아니라 '탈락시켜 0 으로 만든다' 는 구조입니다.
        //   Phase 1 의 역분산 융합은 이 바닥을 '탈락 임계값' 에서
        //   '저가중 전환점' 으로 바꾸는데, 그러려면 바닥과 점수 분포가
        //   이 서식에서 통상 어느 대역인지가 먼저 관측되어야 합니다.
        //
        //  ── 감쇠 곡선도 함께 남기는 이유 ──
        //   후보가 [6.28, 2.42, 2.03, 2.02] 처럼 승자독식이면 제목 축을 신뢰해야 하고,
        //   [0.49, 0.52, 0.47] 처럼 평탄하면 감쇠시켜야 합니다.
        //   그 판정에 필요한 형상 통계가 record_decay 한 번에 압축됩니다.
        {
            crate::utils::score_dynamics::record_baseline("title_axis.floor", floor);
            let live: Vec<f32> = ts_all.iter().cloned().filter(|s| *s > f32::MIN).collect();
            if live.len() >= 2 {
                crate::utils::score_dynamics::record_decay("title_axis.self_declared", &live);
            }
        }
        // 🌟 [T-1] 바닥 미달 후보의 가중치를 유도할 분포 범위를 미리 구합니다.
        //    min-max 정규화이므로 이 문서의 분포에서만 유도되며 새 상수가 없습니다.
        let (ts_min, ts_max) = {
            let v: Vec<f32> = ts_all.iter().cloned().filter(|s| *s > f32::MIN).collect();
            if v.is_empty() {
                (0.0f32, 0.0f32)
            } else {
                let mn = v.iter().cloned().fold(f32::MAX, f32::min);
                let mx = v.iter().cloned().fold(f32::MIN, f32::max);
                (mn, mx)
            }
        };
        let mut weak_admitted = 0usize;
        for (i, v) in headless.iter().enumerate() {
            if ts_all[i] == f32::MIN { continue; }
            let ts = ts_all[i];
            let rw = crate::utils::ai_utils::cosine_similarity(&he[i], &anchors[3]);
            let is = crate::utils::ai_utils::cosine_similarity(&he[i], &anchors[2]);
            let ch = crate::utils::ai_utils::cosine_similarity(&he[i], &anchors[4]);
            let ac = crate::utils::ai_utils::cosine_similarity(&he[i], &anchors[5]);
            let rival = rw.max(is).max(ch).max(ac);

            // 🌟 [T-1 / 배타 판정은 유지] 경쟁 5축이 이기면 이것은 표제가 아닙니다.
            //    이 탈락은 정당하므로 완화하지 않습니다.
            if ts <= rival {
                if verbose {
                    let why = if ts <= ch {
                        "사이트 껍데기"
                    } else if ts <= ac {
                        "UI 액션"
                    } else if ts <= is {
                        "품목 속성"
                    } else {
                        "행 구분자"
                    };
                    emit_term(&format!(
                        "     🧹 [HEADLESS DROP] '{}' | 자기선언 {:.4} vs 행구분자 {:.4} vs 품목속성 {:.4} vs 껍데기 {:.4} vs 액션 {:.4} | 바닥 {:.4} → {} (표제 아님)",
                        v, ts, rw, is, ch, ac, floor, why
                    ));
                }
                continue;
            }

            // 🌟 [T-1 / 바닥은 저가중 전환점] 바닥 이상이면 가중치 1.0 으로
            //    기존 동작과 산술적으로 동일합니다. 미만이면 백분위로 감쇠해 생존합니다.
            let w = if ts >= floor {
                1.0f32
            } else {
                let span = (ts_max - ts_min).max(1e-6);
                ((ts - ts_min) / span).clamp(0.0, 1.0)
            };
            if w < 1.0 {
                weak_admitted += 1;
                if verbose {
                    emit_term(&format!(
                        "     🪶 [HEADLESS WEAK-KEEP] '{}' | 자기선언 {:.4} < 바닥 {:.4} 이지만 경쟁 5축({:.4})을 이겼습니다. 탈락 대신 가중치 {:.3} 로 생존시킵니다. (탈락시키면 이 축의 정보가 0 으로 수렴합니다)",
                        v, ts, floor, rival, w
                    ));
                }
            }
            if !kept.iter().any(|(e, _)| e == v) { kept.push((v.clone(), w)); }
        }
        if verbose && weak_admitted > 0 {
            emit_term(&format!(
                "     🪶 [WEAK ADMIT] 바닥 미달이지만 배타 판정을 통과한 후보 {}개를 저가중으로 편입했습니다.",
                weak_admitted
            ));
        }
    }
    if verbose {
        let shown: Vec<String> = kept
            .iter()
            .take(8)
            .map(|(v, w)| format!("{}(w{:.2})", v, w))
            .collect();
        emit_term(&format!(
            "  🪪 [TITLE AXIS] 표제 후보 값 {}개: {:?}",
            kept.len(), shown
        ));
    }
    kept
}

// =====================================================================
// 🌟 [BACK-COMPAT] 기존 시그니처를 그대로 유지하는 래퍼입니다.
// ---------------------------------------------------------------------
//  ── 왜 남기는가 ──
//   호출부가 Vec<String> 을 전제로 작성되어 있고, 가중치를 소비하지 않는
//   경로(페이지 연속성 판정 등)까지 시그니처를 바꾸면 회귀 범위가 넓어집니다.
//   이 래퍼는 가중치 1.0(바닥 이상)만 통과시키므로
//   기존 호출부는 T-1 이전과 바이트 단위로 동일한 결과를 받습니다.
//
//  ⚠️ 새 코드에서 표제 축 점수를 산출할 때는 반드시
//     resolve_title_values_weighted 를 쓰십시오.
//     이 래퍼를 쓰면 SA 형 사고가 그대로 재현됩니다.
// =====================================================================
pub(crate) async fn resolve_title_values(
    model: &LogisModel,
    light_pug: &str,
    band_ratio: f32,
    emit_term: &(dyn Fn(&str) + Send + Sync),
    verbose: bool,
) -> Vec<String> {
    resolve_title_values_weighted(model, light_pug, band_ratio, emit_term, verbose)
        .await
        .into_iter()
        .filter(|(_, w)| *w >= 1.0)
        .map(|(v, _)| v)
        .collect()
}

struct PageIdentityVerdict {
    title_top: f32,
    self_id_top: f32,
    is_standalone: bool,
}
const TRADE_SELF_ID_LABEL_ANCHOR: &str = "document number of this document, \
     number printed under the title of this form, own reference number of this form, \
     invoice number, order number, certificate number, declaration number, \
     bill of lading number, waybill number, policy number, claim number, \
     statement number, receipt number, licence number, booking number";

#[derive(Debug, Clone)]
pub struct TradeRerouteVerdict {
    pub code: String,
    pub title: String,
    /// 🌟 Gumbel 보정 후 점수입니다. (경쟁 키 개수 편향 제거됨)
    pub score: f32,
    pub rival: String,
    pub rival_score: f32,
    /// 🌟 이 코드를 지목한 실제 표제 값. 리라우트 감사 로그의 근거입니다.
    pub evidence_value: String,
    /// 🌟 그 값의 '서식 전문' 원시 코사인 (센터링 이전, 레벨 보존)
    pub title_cos: f32,
    /// 🌟 그 값의 '사이트 껍데기' 원시 코사인
    pub chrome_cos: f32,
    /// 🌟 trade_structural_evidence 가 찾은 국제 표준 포맷 증거
    pub markers: Vec<String>,
}
pub async fn probe_trade_document(
    model: &LogisModel,
    light_pug: &str,
    doc_lang: &str,
    emit_term: &(dyn Fn(&str) + Send + Sync),
) -> Option<TradeRerouteVerdict> {
    let (has_marker, markers) = trade_structural_evidence(light_pug);
    let code_marker = markers.iter().any(|m| m.starts_with("doccode:"));
    let strong_structure = code_marker || markers.len() >= 2;
    if has_marker {
        emit_term(&format!(
            "  🔩 [TRADE STRUCTURE] 국제 표준 포맷 증거 {}건 {:?} | 강한 증거: {}",
            markers.len(), markers, if strong_structure { "예(코드 접두어 또는 2종 이상)" } else { "아니오(1종)" }
        ));
    } else {
        emit_term("  ⚪ [TRADE STRUCTURE] 국제 표준 포맷 증거가 없습니다. (Incoterms / HS / 컨테이너 / AWB / B-L / 서식코드 문서번호)");
    }
    // 🌟 [T-1] 가중치를 보존한 채 표제 후보를 받습니다.
    //
    //  ── 왜 이 경로가 표적인가 ──
    //   SA-2026-0828 에서 'document_type: Shipping Advice' 가 바닥 미달로
    //   탈락한 결과, 남은 3개 후보만으로 채점되어
    //   마진 +0.0308 (잡음대 0.5334) 의 코인플립이 발생했습니다.
    //   여기서 약가중군을 합류시키면 그 후보가 서식 전문 'shipping advice' 와
    //   거의 완전 일치하므로 SA 코드의 점수가 정상 수준으로 회복됩니다.
    //
    //  ── 임베딩 비용 증가 없음 ──
    //   values 전체를 한 번에 배치 임베딩한 뒤 슬라이스로 나눕니다.
    //   bank_neutral_key_scores 를 두 번 호출하지만 그것은 순수 산술입니다.
    let weighted = resolve_title_values_weighted(model, light_pug, 0.30, emit_term, true).await;
    if weighted.is_empty() {
        emit_term("  ⚪ [MODE PROBE] 자기 종류를 선언하는 표제가 없습니다. 커머스 판정을 유지합니다.");
        return None;
    }
    let values: Vec<String> = weighted.iter().map(|(v, _)| v.clone()).collect();
    let title_weights: Vec<f32> = weighted.iter().map(|(_, w)| *w).collect();
    let weak_weight_max = title_weights
        .iter()
        .cloned()
        .filter(|w| *w < 1.0)
        .fold(0.0f32, f32::max);
    if weak_weight_max > 0.0 {
        emit_term(&format!(
            "  🪶 [TITLE WEAK AXIS] 저가중 표제 후보가 존재합니다. 최대 가중치 {:.3} 로 본 채점에 합류시킵니다. (탈락시키면 표제 축이 0 으로 수렴해 구조 증거 하나에만 의존하게 됩니다)",
            weak_weight_max
        ));
    }
    let val_embs = model.get_embedding_batch(values.clone()).await
        .unwrap_or_else(|_| vec![vec![0.0; 384]; values.len()]);
    // ── ① 무역 서식 전문 + 커머스 페이지 타입을 한 판에 올립니다 ──
    let mut bias_defs: Vec<(String, String, String)> = Vec::new();
    for (code, title) in TRADE_DOC_TITLES.iter() {
        bias_defs.push(("trade".to_string(), code.to_string(), title.to_string()));
    }
    const COMMERCE_TYPES: [&str; 6] = ["order", "goods", "tracking", "review", "coupon", "event"];
    let mut commerce_langs: Vec<String> = vec![doc_lang.to_string()];
    if doc_lang != "en" { commerce_langs.push("en".to_string()); }
    for c in COMMERCE_TYPES.iter() {
        for lg in commerce_langs.iter() {
            let anchor = crate::parsing::get_page_type_classification_bias(c, lg);
            for p in crate::utils::ai_utils::split_bias_phrases_full(&anchor) {
                if bias_defs.iter().any(|(_, k, e)| k == *c && e == &p) { continue; }
                bias_defs.push(("commerce".to_string(), c.to_string(), p));
            }
        }
    }
    // ── ② [CHROME PREJUDICE] 전 키 공통 편견 축 ──
    let chrome_phrases: Vec<String> = {
        let mut v = crate::utils::ai_utils::split_bias_phrases_full(SITE_CHROME_ANCHOR);
        for p in crate::utils::ai_utils::split_bias_phrases_full(crate::logic::UI_ACTION_ANCHOR) {
            if !v.iter().any(|e| e == &p) { v.push(p); }
        }
        v
    };
    let mut key_order: Vec<(String, String)> = Vec::new();
    for (c, k, _) in bias_defs.iter() {
        if !key_order.iter().any(|(_, kk)| kk == k) {
            key_order.push((c.clone(), k.clone()));
        }
    }
    let mut prej_defs: Vec<(String, String, String)> = Vec::new();
    for (c, k) in key_order.iter() {
        for p in chrome_phrases.iter() {
            prej_defs.push((c.clone(), k.clone(), p.clone()));
        }
    }
    let mut uniq: Vec<String> = Vec::new();
    for (_, _, p) in bias_defs.iter().chain(prej_defs.iter()) {
        if !uniq.iter().any(|e| e == p) { uniq.push(p.clone()); }
    }
    let uniq_embs = model.get_embedding_batch(uniq.clone()).await
        .unwrap_or_else(|_| vec![vec![0.0; 384]; uniq.len()]);
    let emb_of = |p: &str| -> Vec<f32> {
        match uniq.iter().position(|e| e == p) {
            Some(i) => uniq_embs[i].clone(),
            None => vec![0.0f32; 384],
        }
    };
    {
        let trade_embs: Vec<Vec<f32>> = bias_defs.iter()
            .filter(|(c, _, _)| c == "trade")
            .map(|(_, _, p)| emb_of(p))
            .collect();
        let mut rebuilt: Vec<(String, String, String)> = Vec::new();
        let mut dropped_total = 0usize;
        for c in COMMERCE_TYPES.iter() {
            let own_phrases: Vec<String> = bias_defs.iter()
                .filter(|(cat, k, _)| cat == "commerce" && k == *c)
                .map(|(_, _, p)| p.clone())
                .collect();
            if own_phrases.is_empty() { continue; }
            let own_embs: Vec<Vec<f32>> = own_phrases.iter().map(|p| emb_of(p)).collect();
            let mut kept: Vec<String> = Vec::new();
            let mut dropped: Vec<String> = Vec::new();
            for (pi, p) in own_phrases.iter().enumerate() {
                if own_embs[pi].iter().all(|&v| v == 0.0) { continue; }
                let mut own = 0.0f32;
                for pj in 0..own_phrases.len() {
                    if pj == pi { continue; }
                    if own_embs[pj].iter().all(|&v| v == 0.0) { continue; }
                    let s = crate::utils::ai_utils::cosine_similarity(&own_embs[pi], &own_embs[pj]);
                    if s > own { own = s; }
                }
                let mut rival = 0.0f32;
                for te in trade_embs.iter() {
                    if te.iter().all(|&v| v == 0.0) { continue; }
                    let s = crate::utils::ai_utils::cosine_similarity(&own_embs[pi], te);
                    if s > rival { rival = s; }
                }
                if rival >= own {
                    dropped.push(format!("{}(own {:.3} <= trade {:.3})", p, own, rival));
                } else {
                    kept.push(p.clone());
                }
            }
            if kept.is_empty() {
                emit_term(&format!(
                    "     ⚠️ [CROSS-MODE MASK] '{}' 뱅크의 모든 구가 실격되어 마스크를 적용하지 않습니다.",
                    c
                ));
                for p in own_phrases { rebuilt.push(("commerce".to_string(), c.to_string(), p)); }
                continue;
            }
            if !dropped.is_empty() {
                dropped_total += dropped.len();
                emit_term(&format!(
                    "     🧹 [CROSS-MODE MASK] 커머스 '{}' 에서 무역 전문을 더 잘 설명하는 구 {}개 제거 (잔존 {}개): {:?}",
                    c, dropped.len(), kept.len(), dropped.iter().take(6).collect::<Vec<_>>()
                ));
            }
            for p in kept { rebuilt.push(("commerce".to_string(), c.to_string(), p)); }
        }
        if dropped_total > 0 {
            let mut merged: Vec<(String, String, String)> = bias_defs.iter()
                .filter(|(c, _, _)| c == "trade").cloned().collect();
            merged.extend(rebuilt);
            bias_defs = merged;
            emit_term(&format!(
                "     🧹 [CROSS-MODE MASK] 총 {}개 공유 개념구를 커머스 뱅크에서 제거했습니다. (mode 간 유사도 충돌 차단)",
                dropped_total
            ));
        }
    }
    let bank: Vec<(String, String, Vec<f32>)> = bias_defs.iter()
        .map(|(c, k, p)| (c.clone(), k.clone(), emb_of(p))).collect();
    let prej_bank: Vec<(String, String, Vec<f32>)> = prej_defs.iter()
        .map(|(c, k, p)| (c.clone(), k.clone(), emb_of(p))).collect();
    let chrome_embs: Vec<Vec<f32>> = chrome_phrases.iter().map(|p| emb_of(p)).collect();
    let (keys, net, raw) = crate::utils::ai_utils::bank_neutral_key_matrix(
        &val_embs, &bank, &prej_bank,
    );
    if keys.is_empty() { return None; }
    let is_commerce = |k: &str| COMMERCE_TYPES.iter().any(|c| *c == k);
    let q = val_embs.len();
    // ── ③ [UNIT NORMALIZE] net 을 이 문서 안에서 다시 z 로 통일합니다 ★ ──
    //
    //  ── 무엇이 문제였나 (실측) ──
    //   bank_neutral_key_matrix 의 ⑤단계에는 SINGLE QUERY GUARD 가 있습니다.
    //       let single = n < 2;
    //       if single { net = raw_b - prej }     ← 코사인 차이 (z 아님)
    //       else      { net = z_b - z_p }        ← z
    //   질의가 1개면 net 은 z 가 아닌데, 여기에 z 공간의 √(2 ln N) 을 빼면
    //   반드시 큰 음수가 나옵니다.
    //     🚢 DGD  raw +0.0940 → 보정 -2.7433
    //     🛒 order raw +0.1482 → 보정 -1.7448
    //   두 값 모두 음수인 것은 판정이 아니라 '단위가 깨졌다' 는 신호입니다.
    //   게다가 무역 draw(56) > 커머스 draw(6) 이므로 무역만 더 깎여
    //   보정이 정확히 반대 방향으로 작동했습니다.
    //
    //   질의가 3개여도 안전하지 않습니다. 행 센터링 자유도가 2뿐이라
    //   max 가 구조적으로 작아지는데(관측 +1.6954) √(2 ln 168)=3.2012 은
    //   i.i.d. 표준정규 가정값이라 여전히 과대합니다.
    //
    //  ── 해결 ──
    //   보정을 빼기 '전에' net 값 전체의 표준편차로 나눠 단위를 z 로 통일합니다.
    //   (열 센터링 때문에 평균은 이미 0 근처이므로 척도만 맞추면 됩니다)
    //   이러면 single 분기든 아니든, 질의가 1개든 15개든 같은 척도가 됩니다.
    let net_sd = {
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        let mut cnt = 0usize;
        for ki in 0..keys.len() {
            for qi in 0..q {
                let v = net[ki][qi];
                if v == f32::MIN { continue; }
                sum += v as f64;
                sq += (v as f64) * (v as f64);
                cnt += 1;
            }
        }
        if cnt < 2 {
            1.0f32
        } else {
            let mu = sum / cnt as f64;
            let var = (sq / cnt as f64 - mu * mu).max(0.0);
            (var.sqrt() as f32).max(1e-6)
        }
    };
    let mut trade_keys = 0usize;
    let mut commerce_keys = 0usize;
    for k in keys.iter() {
        if is_commerce(k) { commerce_keys += 1; } else { trade_keys += 1; }
    }
    let trade_draws = trade_keys.saturating_mul(q);
    let commerce_draws = commerce_keys.saturating_mul(q);
    let trade_base = crate::utils::ai_utils::gumbel_expected_z(trade_draws);
    let commerce_base = crate::utils::ai_utils::gumbel_expected_z(commerce_draws);
    let mut best_trade: (String, f32, usize) = (String::new(), f32::MIN, 0);
    let mut best_commerce: (String, f32, usize) = (String::new(), f32::MIN, 0);
    for (ki, k) in keys.iter().enumerate() {
        for qi in 0..q {
            let v = net[ki][qi];
            if v == f32::MIN { continue; }
            if is_commerce(k) {
                if v > best_commerce.1 { best_commerce = (k.clone(), v, qi); }
            } else if v > best_trade.1 {
                best_trade = (k.clone(), v, qi);
            }
        }
    }
    let trade_score = if best_trade.1 == f32::MIN { f32::MIN } else { best_trade.1 / net_sd - trade_base };
    let commerce_score = if best_commerce.1 == f32::MIN { f32::MIN } else { best_commerce.1 / net_sd - commerce_base };
    emit_term(&format!(
        "     📐 [MODE PROBE / EVT] 질의 {}개 | net 표준편차 {:.4} (단위 통일 척도) | 무역 키 {}개(draw {} → 기대 최댓값 {:.4}) | 커머스 키 {}개(draw {} → {:.4})",
        q, net_sd, trade_keys, trade_draws, trade_base, commerce_keys, commerce_draws, commerce_base
    ));
    emit_term(&format!(
        "     📐 [MODE PROBE] 🚢 {} | net {:+.4} → z {:+.4} → 보정 {:+.4}  |  🛒 {} | net {:+.4} → z {:+.4} → 보정 {:+.4}",
        if best_trade.0.is_empty() { "-" } else { best_trade.0.as_str() },
        if best_trade.1 == f32::MIN { 0.0 } else { best_trade.1 },
        if best_trade.1 == f32::MIN { 0.0 } else { best_trade.1 / net_sd },
        if trade_score == f32::MIN { 0.0 } else { trade_score },
        if best_commerce.0.is_empty() { "-" } else { best_commerce.0.as_str() },
        if best_commerce.1 == f32::MIN { 0.0 } else { best_commerce.1 },
        if best_commerce.1 == f32::MIN { 0.0 } else { best_commerce.1 / net_sd },
        if commerce_score == f32::MIN { 0.0 } else { commerce_score }
    ));
    if trade_score == f32::MIN {
        emit_term("  🛒 [MODE KEEP] 무역 코드가 하나도 점수를 얻지 못했습니다. 커머스 판정을 유지합니다.");
        return None;
    }
    // ── ④ 게이트 1 : 진영 승패 ──
    //
    //  🌟 [STRUCTURE OVERRIDE] 강한 구조 증거가 있으면 진영 승부를 건너뜁니다.
    //     그 경우 코사인의 역할은 '무역인가' 가 아니라 '무역 중 어느 코드인가' 로 축소됩니다.
    //     커머스 웹페이지가 Incoterms 나 서식코드 문서번호를 갖는 일은 사실상 없습니다.
    if commerce_score >= trade_score {
        if strong_structure {
            emit_term(&format!(
                "  🔩 [STRUCTURE OVERRIDE] 코사인은 커머스 '{}'({:+.4}) 가 무역 '{}'({:+.4}) 이상이지만, 국제 표준 포맷 강한 증거 {:?} 가 있어 무역으로 확정합니다. (커머스 화면에는 이 포맷이 인쇄되지 않습니다)",
                best_commerce.0, commerce_score, best_trade.0, trade_score, markers
            ));
        } else {
            emit_term(&format!(
                "  🛒 [MODE KEEP] 보정 후 커머스 '{}'({:+.4}) 가 무역 '{}'({:+.4}) 이상이고 강한 구조 증거도 없습니다. 무역 리라우트를 하지 않습니다.",
                best_commerce.0, commerce_score, best_trade.0, trade_score
            ));
            return None;
        }
    }
    if trade_score <= 0.0 && !strong_structure {
        emit_term(&format!(
            "  ⚪ [MODE PROBE] 무역 코드 '{}' 의 보정 점수 {:+.4} 는 '무작위로 {}개를 뽑았을 때의 기대 최댓값' 이하이고 강한 구조 증거도 없습니다. 리라우트하지 않습니다.",
            best_trade.0, trade_score, trade_draws
        ));
        return None;
    }
    // ── ⑤ 게이트 2 : [TITLE CONFIRM] 원시 코사인으로 레벨을 되살려 확인 ──
    let win_ki = keys.iter().position(|k| k == &best_trade.0).unwrap_or(0);
    let title_cos = {
        let v = raw[win_ki][best_trade.2];
        if v == f32::MIN { 0.0 } else { v }
    };
    let chrome_cos = {
        let mut m = 0.0f32;
        for e in chrome_embs.iter() {
            if e.iter().all(|&x| x == 0.0) { continue; }
            let s = crate::utils::ai_utils::cosine_similarity(&val_embs[best_trade.2], e);
            if s > m { m = s; }
        }
        m
    };
    let evidence_value = values.get(best_trade.2).cloned().unwrap_or_default();
    if title_cos <= chrome_cos && !strong_structure {
        emit_term(&format!(
            "  🛒 [TITLE NOT CONFIRMED] 무역 1위 '{}' 를 지목한 표제 값 \"{}\" 은 서식 전문({:.4})보다 사이트 껍데기({:.4})에 더 가깝고 강한 구조 증거도 없습니다. 리라우트하지 않습니다.",
            best_trade.0, evidence_value, title_cos, chrome_cos
        ));
        return None;
    }
    // ── ⑥ 게이트 3 : 마진이 극값 잡음대 안이면 구조 증거를 요구 ──
    let margin = trade_score - commerce_score;
    let evt_sd = |n: usize| -> f32 {
        let z = crate::utils::ai_utils::gumbel_expected_z(n);
        if z <= 0.0 { 0.0 } else { (std::f32::consts::PI / 6.0f32.sqrt()) / z }
    };
    let noise_band = evt_sd(trade_draws).max(evt_sd(commerce_draws));
    crate::utils::score_dynamics::record_baseline("mode_probe.noise_band", noise_band);
    crate::utils::score_dynamics::record_baseline("mode_probe.margin", margin);
    crate::utils::score_dynamics::record_baseline("mode_probe.net_sd", net_sd);
    crate::utils::score_dynamics::record_baseline("mode_probe.title_cos", title_cos);
    crate::utils::score_dynamics::record_baseline("mode_probe.chrome_cos", chrome_cos);
    {
        let mut trade_row: Vec<f32> = Vec::new();
        for (ki, k) in keys.iter().enumerate() {
            if is_commerce(k) { continue; }
            let mut best = f32::MIN;
            for qi in 0..q {
                let v = net[ki][qi];
                if v == f32::MIN { continue; }
                if v > best { best = v; }
            }
            if best > f32::MIN { trade_row.push(best); }
        }
        if let Some(snr) = crate::utils::ai_utils::axis_snr(&trade_row) {
            emit_term(&format!(
                "  📐 [MODE PROBE / AXIS SNR] 무역 코드 축 SNR {:.4} | 저가중 표제 합류 {} | 마진 {:+.4} (잡음대 {:.4})",
                snr,
                if weak_weight_max > 0.0 { "예" } else { "아니오" },
                margin, noise_band
            ));
            crate::utils::score_dynamics::record_baseline("mode_probe.axis_snr", snr);
            crate::utils::score_dynamics::record_decay("mode_probe.trade_codes", &trade_row);
        }
    }

    if margin < noise_band && !has_marker {
        emit_term(&format!(
            "  🛒 [THIN MARGIN] 진영 마진 {:+.4} 가 극값 잡음대 {:.4} 미만이고 국제 표준 포맷 증거도 없습니다. 리라우트하지 않습니다.",
            margin, noise_band
        ));
        return None;
    }
    let title = TRADE_DOC_TITLES.iter()
        .find(|(c, _)| *c == best_trade.0.as_str())
        .map(|(_, t)| t.to_string())
        .unwrap_or_default();
    emit_term(&format!(
        "  ✅ [MODE PROBE CONFIRMED] '{}' | 보정 {:+.4} | 마진 {:+.4} (잡음대 {:.4}) | 근거 표제 \"{}\" (전문 {:.4} vs 껍데기 {:.4}) | 구조 증거 {:?}",
        best_trade.0, trade_score, margin, noise_band, evidence_value, title_cos, chrome_cos, markers
    ));
    // 🌟 [SDS 계측] 확정된 코드로 1차 스코프를 정밀화합니다.
    //    이 시점 이후의 모든 관측은 'trading|CI|' 처럼 서식별로 격리됩니다.
    crate::utils::score_dynamics::refine_primary(&best_trade.0);
    Some(TradeRerouteVerdict {
        code: best_trade.0,
        title,
        score: trade_score,
        rival: best_commerce.0,
        rival_score: if commerce_score == f32::MIN { 0.0 } else { commerce_score },
        evidence_value,
        title_cos,
        chrome_cos,
        markers,
    })
}

fn trade_structural_evidence(pug: &str) -> (bool, Vec<String>) {
    let upper = pug.to_uppercase();
    let mut found: Vec<String> = Vec::new();
    if let Ok(re) = regex::Regex::new(r"\b[A-Z]{4}\s?\d{7}\b") {
        if let Some(m) = re.find(&upper) {
            found.push(format!("container:{}", m.as_str().trim()));
        }
    }
    if let Ok(re) = regex::Regex::new(r"\b\d{3}-\d{8}\b") {
        if let Some(m) = re.find(&upper) {
            found.push(format!("awb:{}", m.as_str()));
        }
    }
    if let Ok(re) = regex::Regex::new(r"\b(\d{4})([.\-])(\d{2})[.\-](\d{2,4})\b") {
        for cap in re.captures_iter(&upper) {
            let sep = cap.get(2).map(|m| m.as_str()).unwrap_or("-");
            let g2: u32 = cap.get(3).and_then(|m| m.as_str().parse().ok()).unwrap_or(99);
            let g3: u32 = cap.get(4).and_then(|m| m.as_str().parse().ok()).unwrap_or(99);
            let looks_like_date = sep == "-" && (1..=12).contains(&g2) && (1..=31).contains(&g3);
            if looks_like_date { continue; }
            found.push(format!("hs:{}", cap.get(0).map(|m| m.as_str()).unwrap_or("")));
            break;
        }
    }
    const INCOTERMS: [&str; 11] = [
        "EXW", "FCA", "FAS", "FOB", "CFR", "CIF", "CPT", "CIP", "DAP", "DPU", "DDP",
    ];
    'inco: for t in INCOTERMS.iter() {
        for tok in upper.split(|c: char| !c.is_ascii_alphanumeric()) {
            if tok == *t {
                found.push(format!("incoterms:{}", t));
                break 'inco;
            }
        }
    }
    if upper.contains("B/L") {
        found.push("bl_label".to_string());
    }
    if let Ok(re) = regex::Regex::new(r"\b([A-Z]{2,8})-[A-Z0-9]{3,}\b") {
        let mut seen: Vec<String> = Vec::new();
        for cap in re.captures_iter(&upper) {
            let prefix = match cap.get(1) { Some(m) => m.as_str().to_string(), None => continue };
            if seen.iter().any(|e| e == &prefix) { continue; }
            let known = TRADE_DOC_TITLES.iter().any(|(c, _)| *c == prefix.as_str());
            if !known { continue; }
            seen.push(prefix.clone());
            found.push(format!("doccode:{}", prefix));
            if seen.len() >= 4 { break; }
        }
    }
    (!found.is_empty(), found)
}
fn is_row_index_section(section: &str) -> bool {
    let s: Vec<char> = section.trim().chars().collect();
    if s.is_empty() { return false; }
    let mut i = 0usize;
    while i < s.len() {
        if !s[i].is_ascii_digit() { i += 1; continue; }
        let start = i;
        while i < s.len() && s[i].is_ascii_digit() { i += 1; }
        let left_ok  = start == 0 || !s[start - 1].is_alphanumeric();
        let right_ok = i == s.len() || !s[i].is_alphanumeric();
        if left_ok && right_ok { return true; }
    }
    false
}
fn recover_pair_sections(anchors: &[(usize, usize)], pug_lines: &[String]) -> Vec<String> {
    let mut consumed: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for (label_line, primary_line) in anchors.iter() {
        consumed.insert(*label_line);
        consumed.insert(*primary_line);
    }
    let mut out: Vec<String> = Vec::with_capacity(anchors.len());
    for (label_line, _) in anchors.iter() {
        let li = (*label_line).min(pug_lines.len().saturating_sub(1));
        let mut sec = String::new();
        let mut i = li;
        while i > 0 {
            i -= 1;
            if consumed.contains(&i) { continue; }
            if pug_lines[i].trim().is_empty() { continue; }
            let (_, tag, _, txt) = crate::utils::ai_utils::pug_line_parts(&pug_lines[i]);
            if matches!(
                tag.as_str(),
                "table" | "thead" | "tbody" | "tfoot" | "tr" | "colgroup" | "col"
            ) {
                continue;
            }
            let t = txt.trim().trim_end_matches(':').trim().to_string();
            if t.chars().count() < 2 { continue; }
            sec = t;
            break;
        }
        out.push(sec);
    }
    out
}
fn prune_absent_keys(
    v: &mut Value,
    absent: &std::collections::HashSet<String>,
    dropped: &mut Vec<String>,
) {
    match v {
        Value::Object(map) => {
            let ks: Vec<String> = map.keys().cloned().collect();
            for k in ks {
                if absent.contains(&k) {
                    let was_filled = map.get(&k).map(|v| match v {
                        Value::Null => false,
                        Value::String(s) => {
                            let t = s.trim();
                            !(t.is_empty() || t == "null" || t == "N/A")
                        }
                        Value::Array(a) => !a.is_empty(),
                        Value::Object(o) => !o.is_empty(),
                        _ => true,
                    }).unwrap_or(false);
                    map.remove(&k);
                    if was_filled && !dropped.iter().any(|e| e == &k) { dropped.push(k); }
                    continue;
                }
                if let Some(child) = map.get_mut(&k) {
                    if child.is_object() || child.is_array() {
                        prune_absent_keys(child, absent, dropped);
                    }
                }
            }
        }
        Value::Array(arr) => {
            for it in arr.iter_mut() {
                prune_absent_keys(it, absent, dropped);
            }
        }
        _ => {}
    }
}
fn merge_trading_page_map(
    target: &mut serde_json::Map<String, Value>,
    source: &serde_json::Map<String, Value>,
) {
    fn is_empty_val(v: &Value) -> bool {
        match v {
            Value::Null => true,
            Value::String(s) => {
                let t = s.trim();
                t.is_empty() || t == "N/A" || t == "null"
            },
            Value::Array(a) => a.is_empty(),
            Value::Object(o) => o.is_empty(),
            _ => false,
        }
    }

    // 🌟 [ROW FRAGMENT MERGE] 축이 겹치지 않는 인접 조각 행을 한 행으로 접습니다.
    //
    //  ── 실측 사고 (SA-2026-0828) ──
    //   line_items: [
    //     { "description": "Cotton T-Shirts ..." },      ← 설명만 있는 행
    //     { "quantity": 1 }                              ← 수치만 있는 행
    //   ]
    //   PLINKO 가 arr[0] 에 한 축을 넣고, 그 뒤 LLM 이 다른 축만 담은 배열을 돌려주면
    //   merge_json_manual 이 그것을 원소로 '추가' 하기 때문에 한 품목이 두 행이 됩니다.
    //
    //  ── 판정 규칙 (전부 구조) ──
    //   R1 두 행의 '값이 있는 키 집합' 이 교집합 0 이어야 합니다.
    //      (같은 축을 둘 다 갖고 있으면 서로 다른 품목이므로 절대 합치지 않습니다)
    //   R2 두 행 모두 '행 정체성 축'(description / item_code / container_number)을
    //      동시에 갖고 있으면 합치지 않습니다. 정체성이 둘이면 품목도 둘입니다.
    //   R3 인접한 행끼리만 봅니다. 표의 행 순서가 곧 품목 순서이기 때문입니다.
    fn row_identity_keys() -> [&'static str; 4] {
        ["description", "item_code", "container_number", "charge_code"]
    }
    fn filled_keys(v: &Value) -> Vec<String> {
        match v.as_object() {
            Some(o) => o
                .iter()
                .filter(|(_, val)| !is_empty_val(val))
                .map(|(k, _)| k.clone())
                .collect(),
            None => Vec::new(),
        }
    }
    fn try_fold_fragment_rows(arr: &mut Vec<Value>) -> usize {
        if arr.len() < 2 { return 0; }
        let ident = row_identity_keys();
        let mut folded = 0usize;
        let mut i = 0usize;
        while i + 1 < arr.len() {
            let a_keys = filled_keys(&arr[i]);
            let b_keys = filled_keys(&arr[i + 1]);
            if a_keys.is_empty() || b_keys.is_empty() { i += 1; continue; }
            let overlap = a_keys.iter().any(|k| b_keys.iter().any(|x| x == k));
            if overlap { i += 1; continue; }
            let a_ident = a_keys.iter().any(|k| ident.iter().any(|x| x == k));
            let b_ident = b_keys.iter().any(|k| ident.iter().any(|x| x == k));
            if a_ident && b_ident { i += 1; continue; }
            let src = arr[i + 1].clone();
            if let (Some(tgt_obj), Some(src_obj)) = (arr[i].as_object_mut(), src.as_object()) {
                for (k, v) in src_obj {
                    if is_empty_val(v) { continue; }
                    tgt_obj.insert(k.clone(), v.clone());
                }
            }
            arr.remove(i + 1);
            folded += 1;
        }
        folded
    }

    for (cat, src_val) in source {
        
        if let Some(src_arr) = src_val.as_array() {
            let entry = target.entry(cat.clone()).or_insert_with(|| json!([]));
            if !entry.is_array() { *entry = json!([]); }
            if let Some(tgt_arr) = entry.as_array_mut() {
                for item in src_arr {
                    if is_empty_val(item) { continue; }
                    if tgt_arr.iter().any(|ex| ex == item) { continue; }
                    tgt_arr.push(item.clone());
                }
                let folded = try_fold_fragment_rows(tgt_arr);
                if folded > 0 {
                    println!(
                        "  🧬 [ROW FRAGMENT MERGE] '{}' 배열에서 축이 겹치지 않는 조각 행 {}개를 접었습니다. (잔존 {}행)",
                        cat, folded, tgt_arr.len()
                    );
                }
            }
            continue;
        }

        
        if let Some(src_obj) = src_val.as_object() {
            let entry = target.entry(cat.clone()).or_insert_with(|| json!({}));
            if !entry.is_object() { *entry = json!({}); }
            if let Some(tgt_obj) = entry.as_object_mut() {
                for (k, v) in src_obj {
                    if is_empty_val(v) { continue; }
                    let need = match tgt_obj.get(k) {
                        None => true,
                        Some(cur) => is_empty_val(cur),
                    };
                    if need { tgt_obj.insert(k.clone(), v.clone()); }
                }
            }
            continue;
        }

        
        if is_empty_val(src_val) { continue; }
        let need = match target.get(cat) {
            None => true,
            Some(cur) => is_empty_val(cur),
        };
        if need { target.insert(cat.clone(), src_val.clone()); }
    }
}

pub(crate) fn normalize_trading_data(item: &mut Value, doc_lang: &str) {
    // 🌟 [NUMERIC KEY / RULE-BASED] 이름 화이트리스트를 규칙 판정으로 대체합니다.
    //
    //  ── 실측 사고 ──
    //   SR 결과에 container_gross_weight: "1000.0", item_gross_weight: "1500.0" 이
    //   문자열로 남았습니다. 아래 배열이 15개 이름만 손으로 나열하고 있어
    //   container_* / item_* 계열이 전부 누락되었기 때문입니다.
    //   문자열로 저장되면 canonicalize_data 의 CanonKind::Numeric 판정에는 걸리지만,
    //   그 전에 json_to_natural_language 가 "Its container gross weight is 1000.0" 으로
    //   문장을 만들고, Dexie 의 belowOrEqual 비교가 문자열 비교로 떨어집니다.
    //
    //  ── 왜 canonical::kind_of 를 재사용하는가 ──
    //   utils/canonical.rs 는 NUM_SUFFIX 에 "_weight" / "_volume" / "_count" 를,
    //   NUM_CONTAINS 에 "price" / "amount" / "measurement" / "tare_weight" 를 이미 갖고 있어
    //   container_gross_weight / item_net_weight / item_package_count 를 전부 잡습니다.
    //   같은 규칙을 두 벌 유지하면 반드시 어긋나므로 진실의 원천 하나에 위임합니다.
    //
    //  ── DATE 도 동일 ──
    //   trade_schema 는 declaration_date / clearance_date / inspection_date /
    //   treatment_date / weighing_date / claim_date / effective_date / transaction_date /
    //   maturity_date / valid_until / departure_date / arrival_date /
    //   latest_shipment_date / cargo_closing_date 등 20축 이상의 날짜를 갖는데
    //   아래 배열은 6개뿐이었습니다.
    //   canonical 의 NUM_SUFFIX 는 "_at" 만 날짜로 보므로, 여기서는
    //   '_date 로 끝나거나 date 를 포함' 이라는 구조 규칙을 별도로 적용합니다.
    const DATE_EXACT: [&str; 8] = [
        "etd", "eta", "valid_until", "due_date",
        "maturity_date", "expiry_date", "issue_date", "registration_date",
    ];

    fn is_numeric_key(k: &str) -> bool {
        use crate::utils::canonical::{kind_of, CanonKind};
        // 날짜가 우선입니다. registration_date 는 canonical 규칙상 Free 이지만 날짜입니다.
        if is_date_key(k) { return false; }
        matches!(kind_of(k), CanonKind::Numeric)
    }

    fn is_date_key(k: &str) -> bool {
        let lower = k.to_lowercase();
        if DATE_EXACT.iter().any(|d| *d == lower) { return true; }
        lower.ends_with("_date") || lower.starts_with("date_") || lower.ends_with("_at")
    }

    fn to_number(v: &Value) -> Option<f64> {
        match v {
            Value::Number(n) => n.as_f64(),
            Value::String(s) => {
                let mut buf = String::new();
                let mut seen_digit = false;
                for c in s.chars() {
                    if c.is_ascii_digit() {
                        buf.push(c);
                        seen_digit = true;
                    } else if c == ',' && seen_digit {
                        continue;
                    } else if c == '.' && seen_digit && !buf.contains('.') {
                        buf.push(c);
                    } else if seen_digit {
                        break;
                    }
                }
                if !seen_digit { return None; }
                buf.trim_end_matches('.').parse::<f64>().ok()
            },
            _ => None,
        }
    }

    /// 🌟 [MONTH TOKEN] 영문 월 약어를 숫자로 환원합니다.
    ///
    ///  ── 왜 필요한가 ──
    ///   기존 to_iso_date 는 정규식 \d+ 로 숫자만 뽑습니다.
    ///   "Apr-19-2022" 에서는 [19, 2022] 두 개만 얻어 nums.len() < 3 으로 즉시 포기하고,
    ///   CI 이미지의 issue_date 가 원문 그대로 남았습니다.
    ///   무역 서식은 숫자 날짜의 월/일 순서 모호성(03/04)을 피하려고
    ///   영문 월 약어를 쓰는 것이 국제 관행이므로 이 경로가 오히려 다수입니다.
    ///
    ///  ── 왜 어휘 하드코딩이 아닌가 ──
    ///   12개월 약어는 ISO 8601 / IATA / SWIFT 가 공유하는 국제 표준 표기이며
    ///   언어별 사전이 아니라 서식 규약입니다.
    ///   (컨테이너 번호가 '영문 4자 + 숫자 7자' 인 것과 같은 성격입니다)
    fn month_token_to_num(t: &str) -> Option<u32> {
        let s: String = t.chars().filter(|c| c.is_ascii_alphabetic()).collect();
        if s.chars().count() < 3 { return None; }
        let head: String = s.chars().take(3).map(|c| c.to_ascii_uppercase()).collect();
        let n = match head.as_str() {
            "JAN" => 1, "FEB" => 2, "MAR" => 3, "APR" => 4,
            "MAY" => 5, "JUN" => 6, "JUL" => 7, "AUG" => 8,
            "SEP" => 9, "OCT" => 10, "NOV" => 11, "DEC" => 12,
            _ => return None,
        };
        Some(n)
    }

    fn to_iso_date(v: &Value) -> Option<String> {
        let s = match v {
            Value::String(s) => s.trim().to_string(),
            Value::Number(n) => n.to_string(),
            _ => return None,
        };
        if s.is_empty() || s == "N/A" || s == "null" { return None; }
        if s.contains('T') && s.chars().count() >= 19 { return Some(s); }

        // 🌟 [ALPHA MONTH PATH] 영문 월 약어가 있으면 그것을 월로 확정하고,
        //    남은 숫자에서 일/연을 크기로 가릅니다. (일 <= 31 < 연)
        //    'Apr-19-2022' / '19 Apr 2022' / 'APRIL 19, 2022' 를 한 경로로 처리합니다.
        {
            let mut alpha_month: Option<u32> = None;
            for tok in s.split(|c: char| !c.is_alphanumeric()) {
                if tok.is_empty() { continue; }
                if let Some(m) = month_token_to_num(tok) {
                    alpha_month = Some(m);
                    break;
                }
            }
            if let Some(month) = alpha_month {
                let re_n = regex::Regex::new(r"\d+").ok()?;
                let nums: Vec<u32> = re_n
                    .find_iter(&s)
                    .filter_map(|m| m.as_str().parse().ok())
                    .collect();
                let mut year: Option<u32> = None;
                let mut day: Option<u32> = None;
                for n in nums.iter() {
                    if *n > 31 {
                        if year.is_none() { year = Some(*n); }
                    } else if day.is_none() {
                        day = Some(*n);
                    }
                }
                // 두 자리 연도('22')만 있는 경우: 일이 이미 잡혔으면 남은 값을 연으로 봅니다.
                if year.is_none() {
                    let leftover: Vec<u32> = nums
                        .iter()
                        .cloned()
                        .filter(|n| Some(*n) != day)
                        .collect();
                    if let Some(y) = leftover.first() { year = Some(*y); }
                }
                if let (Some(mut y), Some(d)) = (year, day) {
                    if y < 100 { y += if y > 50 { 1900 } else { 2000 }; }
                    let dd = d.clamp(1, 31);
                    return Some(format!("{:04}-{:02}-{:02}T00:00:00", y, month, dd));
                }
            }
        }

        let re = regex::Regex::new(r"\d+").ok()?;
        let nums: Vec<u32> = re.find_iter(&s).filter_map(|m| m.as_str().parse().ok()).collect();
        if nums.len() < 3 { return None; }
        let (mut year, mut month, mut day) = (nums[0], nums[1], nums[2]);
        
        if day > 31 && year <= 31 {
            year = nums[2];
            day = nums[1];
            month = nums[0];
        }
        if year < 100 { year += if year > 50 { 1900 } else { 2000 }; }
        
        if month > 12 && day <= 12 { std::mem::swap(&mut month, &mut day); }
        month = month.clamp(1, 12);
        day = day.clamp(1, 31);
        let hour   = if nums.len() > 3 { nums[3].clamp(0, 23) } else { 0 };
        let minute = if nums.len() > 4 { nums[4].clamp(0, 59) } else { 0 };
        let second = if nums.len() > 5 { nums[5].clamp(0, 59) } else { 0 };
        Some(format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", year, month, day, hour, minute, second))
    }

    fn walk(v: &mut Value) {
        match v {
            Value::Object(map) => {
                let keys: Vec<String> = map.keys().cloned().collect();
                for k in keys {
                    if is_date_key(&k) {
                        let converted = map.get(&k).and_then(to_iso_date);
                        if let Some(iso) = converted {
                            map.insert(k.clone(), json!(iso));
                        }
                        continue;
                    }
                    if is_numeric_key(&k) {
                        let converted = map.get(&k).and_then(to_number);
                        if let Some(num) = converted {
                            map.insert(k.clone(), json!(num));
                        }
                        continue;
                    }
                    if let Some(child) = map.get_mut(&k) {
                        if child.is_object() || child.is_array() {
                            walk(child);
                        }
                    }
                }
            },
            Value::Array(arr) => {
                for it in arr.iter_mut() { walk(it); }
            },
            _ => {}
        }
    }
    walk(item);

    if let Some(obj) = item.as_object_mut() {
        let cur = obj.get("currency").and_then(|v| v.as_str()).unwrap_or("").trim().to_string();
        if cur.is_empty() || cur == "N/A" || cur == "null" {
            let def = match doc_lang {
                "ko" => "KRW",
                "ja" => "JPY",
                "zh" | "zh-tw" | "zh-hk" | "zh-hans" => "CNY",
                "de" | "fr" | "it" | "es" | "nl" | "pt" | "el" => "EUR",
                "ru" => "RUB",
                "th" => "THB",
                "vi" => "VND",
                "hi" | "bn" => "INR",
                _ => "USD",
            };
            obj.insert("currency".to_string(), json!(def));
        } else {
            obj.insert("currency".to_string(), json!(cur.to_uppercase()));
        }

        if obj.get("started_at").is_none() {
            if let Some(v) = obj.get("etd").cloned() {
                obj.insert("started_at".to_string(), v);
            }
        }
        if obj.get("expired_at").is_none() {
            let v = obj.get("eta").cloned().or_else(|| obj.get("expiry_date").cloned());
            if let Some(v) = v {
                obj.insert("expired_at".to_string(), v);
            }
        }
    }
}

fn pdf_page_to_structured_html(page_text: &str) -> (String, usize) {
    fn esc(s: &str) -> String {
        s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
    }

    fn is_label_like(s: &str) -> bool {
        let t = s.trim();
        if t.is_empty() { return false; }
        if t.chars().count() > 40 { return false; }
        if !t.chars().any(|c| c.is_alphabetic()) { return false; }
        
        let digits = t.chars().filter(|c| c.is_ascii_digit()).count();
        let alnum = t.chars().filter(|c| c.is_alphanumeric()).count().max(1);
        digits * 2 < alnum
    }

    
    fn split_by_colon(line: &str) -> Option<(String, String)> {
        let chars: Vec<char> = line.chars().collect();
        for (i, c) in chars.iter().enumerate() {
            if *c != ':' && *c != '：' { continue; }
            if i == 0 { continue; }
            let head: String = chars[..i].iter().collect();
            let tail: String = chars[i + 1..].iter().collect();

            
            if tail.starts_with("//") { continue; }
            
            let prev_digit = chars[i - 1].is_ascii_digit();
            let next_digit = chars.get(i + 1).map_or(false, |c| c.is_ascii_digit());
            if prev_digit && next_digit { continue; }

            if !is_label_like(&head) { continue; }
            return Some((head.trim().to_string(), tail.trim().to_string()));
        }
        None
    }

    
    fn split_by_gap(line: &str) -> Option<(String, String)> {
        let chars: Vec<char> = line.chars().collect();
        let mut run = 0usize;
        for i in 0..chars.len() {
            if chars[i] == '\t' {
                run = 2;
            } else if chars[i] == ' ' {
                run += 1;
            } else {
                if run >= 2 {
                    let head: String = chars[..i].iter().collect();
                    let tail: String = chars[i..].iter().collect();
                    if is_label_like(&head) && !tail.trim().is_empty() {
                        return Some((head.trim().to_string(), tail.trim().to_string()));
                    }
                }
                run = 0;
            }
        }
        None
    }

    let mut rows = String::new();
    let mut pair_cnt = 0usize;

    for raw in page_text.lines() {
        let line = raw.trim_end();
        if line.trim().is_empty() { continue; }

        let pair = split_by_colon(line).or_else(|| split_by_gap(line));

        match pair {
            Some((label, value)) if !value.is_empty() => {
                rows.push_str(&format!(
                    "<tr><th scope=\"row\">{}</th><td>{}</td></tr>\n",
                    esc(&label), esc(&value)
                ));
                pair_cnt += 1;
            },
            _ => {
                
                rows.push_str(&format!(
                    "<tr><td colspan=\"2\">{}</td></tr>\n",
                    esc(line.trim())
                ));
            }
        }
    }

    (format!("<table>\n{}</table>", rows), pair_cnt)
}

async fn extract_continuation_page(
    model: &LogisModel,
    clean_html: &str,
    url: &str,
    doc_type: &str,
    doc_lang: &str,
    task_id: &str,
    page_idx: usize,
    cancellation_token: &Arc<AtomicBool>,
    app_handle: &tauri::AppHandle,
    emit_term: &(dyn Fn(&str) + Send + Sync),
) -> anyhow::Result<serde_json::Map<String, Value>> {
    let _ = (task_id, app_handle);
    let mut out = serde_json::Map::new();
    out.insert("header".to_string(), json!({}));
    out.insert("parties".to_string(), json!({}));
    out.insert("logistics".to_string(), json!({}));
    out.insert("conditions".to_string(), json!({}));
    out.insert("financials".to_string(), json!({}));
    out.insert("cargo".to_string(), json!({}));
    out.insert("line_items".to_string(), json!([]));
    out.insert("containers".to_string(), json!([]));
    if cancellation_token.load(Ordering::Relaxed) {
        return Ok(out);
    }
    let content_pug = {
        let full_pug =
            parsing::convert_to_clean_pug(clean_html, PugMode::ListMode, Some(url));
        model.truncate_pug_context(&full_pug, true, 2000, None).await
    };
    let pug_lines: Vec<String> = content_pug.lines().map(|s| s.to_string()).collect();
    let pug_lines_ref: Vec<&str> = pug_lines.iter().map(|s| s.as_str()).collect();
    let mut detail_pairs = crate::utils::ai_utils::collect_detail_label_value_pairs(&pug_lines_ref);
    {
        let anchors: Vec<(usize, usize)> = detail_pairs.iter()
            .map(|p| (p.label_line, p.primary_line)).collect();
        let recovered_secs = recover_pair_sections(&anchors, &pug_lines);
        let mut recovered = 0usize;
        for (i, p) in detail_pairs.iter_mut().enumerate() {
            if !p.section.trim().is_empty() { continue; }
            let s = match recovered_secs.get(i) { Some(s) => s, None => continue };
            if s.is_empty() { continue; }
            p.section = s.clone();
            recovered += 1;
        }
        if recovered > 0 {
            emit_term(&format!(
                "  🧭 [SECTION RECOVERY / CONTINUATION] {}페이지 섹션 {}개 복원",
                page_idx + 1, recovered
            ));
        }
    }
    emit_term(&format!(
        "  🧷 [CONTINUATION PAIR] {}페이지 구조적 라벨-값 페어 {}개 확보 (doc_type='{}')",
        page_idx + 1, detail_pairs.len(), doc_type
    ));
    if detail_pairs.is_empty() {
        return Ok(out);
    }
    let trade_fields = crate::parsing::get_detail_schema_fields(doc_type, url, doc_lang);
    let mut f_names: Vec<String> = Vec::new();
    let mut f_label: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut f_weight: Vec<Vec<f32>> = Vec::new();
    for (fname, _, _, _) in &trade_fields {
        let (lp, lw) = crate::utils::ai_utils::label_phrase_bank(doc_lang, doc_type, fname);
        if lp.is_empty() { continue; }
        let le = model.get_embedding_batch(lp.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; lp.len()]);
        f_names.push(fname.clone());
        f_label.push(le);
        f_weight.push(lw);
    }
    if f_names.is_empty() {
        return Ok(out);
    }
    // 라벨/값 준비
    let mut labels: Vec<String> = Vec::new();
    let mut leafs: Vec<String> = Vec::new();
    let mut vals: Vec<String> = Vec::new();
    let mut lines: Vec<usize> = Vec::new();
    let mut sections: Vec<String> = Vec::new();
    for p in detail_pairs.iter() {
        if p.value.trim().is_empty() { continue; }
        let key = format!("{}\u{1}{}", p.label, p.value);
        if labels.iter().zip(vals.iter()).any(|(l, v)| format!("{}\u{1}{}", l, v) == key) {
            continue;
        }
        labels.push(p.label.clone());
        leafs.push(p.label.clone());
        vals.push(p.value.clone());
        lines.push(p.primary_line);
        sections.push(p.section.trim().to_string());
    }
    if labels.is_empty() {
        return Ok(out);
    }
    let leaf_embs = model.get_embedding_batch(leafs.clone()).await
        .unwrap_or_else(|_| vec![vec![0.0; 384]; leafs.len()]);
    let row_index_section: Vec<bool> = sections
        .iter()
        .map(|s| is_row_index_section(s))
        .collect();
    {
        let hits: Vec<String> = (0..labels.len())
            .filter(|&h| row_index_section[h])
            .map(|h| format!("{}('{}')", labels[h], sections[h]))
            .collect();
        if !hits.is_empty() {
            emit_term(&format!(
                "  🧾 [ROW SECTION SCOPE / CONTINUATION] 행 인덱스 섹션 라벨 {}개는 items/containers 에만 배정합니다: {:?}",
                hits.len(), hits.iter().take(8).collect::<Vec<_>>()
            ));
        }
    }
    let mut matrix: Vec<Vec<f32>> = vec![vec![-1.0f32; labels.len()]; f_names.len()];
    for f in 0..f_names.len() {
        let fmt = crate::utils::ai_utils::detect_field_format(&f_names[f]);
        let f_row_cat = {
            let c = crate::logic::trade_field_category(&f_names[f]);
            c == "items" || c == "containers"
        };
        for h in 0..labels.len() {
            if leaf_embs[h].iter().all(|&v| v == 0.0) { continue; }
            if row_index_section[h] && !f_row_cat { continue; }
            if !crate::utils::ai_utils::value_matches_format(fmt, &vals[h]) {
                continue;
            }
            matrix[f][h] = crate::utils::ai_utils::weighted_max_pool_sim(
                &leaf_embs[h], &f_label[f], &f_weight[f],
            );
        }
    }
    let centered = crate::utils::ai_utils::double_center_matrix(&matrix);
    // 🌟 [T-2] 배열 행 배정에도 잡음 마진 게이트를 적용합니다.
    //
    //  ── 실측 표적 ──
    //   SR-2026-0820 의 line_items 에서
    //     unit: "6.0"        ← B200 의 volume 6.0 이 유출
    //     total_price: 15    ← 전체 volume 15.0 이 유출
    //   두 사고 모두 '순수 수치 셀' 이 여러 수치 필드와 비슷한 점수를 내는
    //   평탄 분포에서 발생했습니다. 즉 Flat 동률이며,
    //   지금은 마진 임계 0.0 이라 그대로 확정됩니다.
    //
    //  ── 배열 행에서는 혼동 사전을 쓰지 않습니다 ──
    //   행 내부 필드는 unit / quantity / total_price 처럼 수치 축이 몰려 있어,
    //   같은 쌍이 반복 충돌하는 것이 '역사적 승자가 있다' 는 뜻이 아니라
    //   '구조적으로 구분 불가' 라는 뜻입니다.
    //   따라서 동률은 종류를 불문하고 보류하고 LLM 에 맡깁니다.
    let (assign, arr_diags) = crate::utils::ai_utils::exclusive_assign_by_score_adaptive(
        &centered,
        0.0,
        0.0,
        |_f: usize, _rival: usize, _kind: crate::utils::ai_utils::TieKind| -> bool { false },
    );
    for d in arr_diags.iter() {
        if d.tie == crate::utils::ai_utils::TieKind::Decisive { continue; }
        let fname = f_names.get(d.field).cloned().unwrap_or_default();
        let rname = f_names.get(d.rival).cloned().unwrap_or_default();
        emit_term(&format!(
            "    ⚖️ [ARRAY TIE HOLD] '{}' vs '{}' | 마진 {:+.4} < 잡음 마진 {:.4} ({:?}) → 보류. 수치 축이 몰린 행에서는 결정론으로 가릴 근거가 없습니다.",
            fname, rname, d.margin, d.noise_margin, d.tie
        ));
        crate::utils::score_dynamics::record_confusion(&fname, &rname, d.margin);
    }
    use crate::logic::trade_field_category;
    let mut assigned = 0usize;
    let mut item_row = serde_json::Map::new();
    let mut container_row = serde_json::Map::new();
    for (f, a) in assign.iter().enumerate() {
        let (h, own, margin) = match a { Some(v) => *v, None => continue };
        let fname = f_names[f].clone();
        if crate::utils::ai_utils::is_id_link_field(&fname) { continue; }
        {
            let mut best_f = usize::MAX;
            let mut best_raw = f32::MIN;
            let mut sum = 0.0f64;
            let mut sq = 0.0f64;
            let mut cnt = 0usize;
            for ff in 0..f_names.len() {
                let v = matrix[ff][h];
                if v < 0.0 { continue; }
                sum += v as f64;
                sq += (v as f64) * (v as f64);
                cnt += 1;
                if v > best_raw { best_raw = v; best_f = ff; }
            }
            if cnt >= 2 && best_f != f && matrix[f][h] >= 0.0 {
                let mu = sum / cnt as f64;
                let sd = ((sq / cnt as f64 - mu * mu).max(0.0)).sqrt() as f32;
                let gap = best_raw - matrix[f][h];
                if gap > sd {
                    emit_term(&format!(
                        "    🚫 [CONTINUATION DRIFT] Label '{}' 의 원시 argmax 는 '{}'({:.4}) 인데 '{}'({:.4}) 로 밀렸습니다. 격차 {:+.4} > 분포 표준편차 {:.4} → 배정을 폐기합니다.",
                        labels[h], f_names[best_f], best_raw, fname, matrix[f][h], gap, sd
                    ));
                    continue;
                }
            }
        }
        let cat = trade_field_category(&fname);
        if cat.is_empty() { continue; }
        let wrote = match cat {
            "items" => { item_row.insert(fname.clone(), json!(vals[h].clone())); true }
            "containers" => { container_row.insert(fname.clone(), json!(vals[h].clone())); true }
            _ => match out.get_mut(cat).and_then(|v| v.as_object_mut()) {
                Some(slot) => { slot.insert(fname.clone(), json!(vals[h].clone())); true }
                None => false,
            },
        };
        if !wrote {
            // 🌟 구버전은 이 경로에서도 ASSIGN 로그를 찍어 소실을 감췄습니다.
            emit_term(&format!(
                "    ⚠️ [CONTINUATION WRITE MISS] Label '{}' → Field '{}' (cat: '{}') 를 기록할 슬롯이 없어 폐기합니다.",
                labels[h], fname, cat
            ));
            continue;
        }
        assigned += 1;
        emit_term(&format!(
            "    ✨ [CONTINUATION ASSIGN] Label '{}' → Field '{}' (cat: {}) | Score: {:+.4} | Margin: {:+.4} | Line {} | Value: \"{}\"",
            labels[h], fname, cat, own, margin, lines[h] + 1, vals[h]
        ));
        crate::utils::score_dynamics::record_field_seen(&fname);
        crate::utils::score_dynamics::record_field_assigned(&fname, margin);
        crate::utils::score_dynamics::record_baseline("trading.continuation_margin", margin);
    }
    // 🌟 [COUNT] 행 필드는 위 루프에서 이미 assigned 에 반영되었습니다. 여기서 다시 더하지 않습니다.
    if !item_row.is_empty() {
        let n = item_row.len();
        if let Some(arr) = out.get_mut("line_items").and_then(|v| v.as_array_mut()) {
            arr.push(Value::Object(item_row));
            emit_term(&format!(
                "    🧾 [CONTINUATION ROW] items 행 1개(필드 {}개)를 line_items 에 추가했습니다.", n
            ));
        }
    }
    if !container_row.is_empty() {
        let n = container_row.len();
        if let Some(arr) = out.get_mut("containers").and_then(|v| v.as_array_mut()) {
            arr.push(Value::Object(container_row));
            emit_term(&format!(
                "    🧾 [CONTINUATION ROW] containers 행 1개(필드 {}개)를 추가했습니다.", n
            ));
        }
    }
    emit_term(&format!(
        "  ✅ [CONTINUATION] {}페이지에서 {}개 필드를 앞 문서 '{}' 에 병합합니다. (LLM 호출 0회)",
        page_idx + 1, assigned, doc_type
    ));
    Ok(out)
}
pub async fn process_trading_task(
    task: Task,
    store_mutex: &Arc<Mutex<Option<VectorStore>>>,
    model_mutex: &Arc<Mutex<Option<LogisModel>>>,
    cancellation_token: &Arc<AtomicBool>,
    app_handle: &tauri::AppHandle,
    device_preference: Option<String>,
) -> anyhow::Result<()> {
    let app_handle_clone = app_handle.clone();
    let tid_clone = task.id.clone();
    let emit_term = move |msg: &str| {
        println!("{}", msg);
        use tauri::Emitter;
        let _ = app_handle_clone.emit("task-console-log", serde_json::json!({"task_id": tid_clone, "text": format!("{}\n", msg)}));
    };

    let zero_addr = "0x0000000000000000000000000000000000000000";
    let from_addr = if task.from.is_empty() { zero_addr.to_string() } else { task.from.clone() };
    let team_id = if task.to.is_empty() || task.to == zero_addr {
        crate::utils::hash::hash_id(&from_addr)
    } else {
        task.to.clone()
    };

    emit_term("\n=======================================");
    emit_term(&format!("[TRADING] ⚙️ Task {} started trading extraction.", task.id));
    // 🌟 [SDS SCOPE / 필수] 계측 스코프를 여기서 세웁니다.
    //
    //  ── 왜 여기여야 하는가 (실측 사고) ──
    //   Phase 0 에서는 scheduler.rs::process_task 의 resolve_absolute_url 직후에
    //   enter_scope 를 두었습니다. 그런데 process_task 는 그보다 훨씬 앞에서
    //     if search_mode == "shipping" && (html_extraction || document_extraction) {
    //         return process_trading_task(...).await;
    //     }
    //   로 이탈합니다. 즉 무역 경로는 enter_scope 에 영원히 도달하지 못합니다.
    //   그 결과 current_scope() 가 None 이 되어 모든 record_* 가 즉시 반환하고,
    //   DIRTY 가 false 로 남아 flush() 가 파일을 쓰지 않았습니다.
    //   score_dynamics.json 이 생성되지 않은 유일한 원인입니다.
    //
    //  ── 1차 키를 'shipping' 으로 시작하는 이유 ──
    //   이 시점에는 doc_type 을 모릅니다. STEP A 가 코드를 확정한 뒤
    //   refine_primary('SA') 가 키를 교체하므로, 그 전 관측만 잠시 'shipping' 에
    //   들어갔다가 이후 관측이 정확한 서식 스코프로 흘러갑니다.
    // 🌟 [2차 스코프] task.cc 는 이 시점에 비어 있습니다.
    //
    //  ── 실측 ──
    //   score_dynamics.json 의 키가 'trading|sa|' 로 secondary 가 공백이었습니다.
    //   로그의 cc='0xa276...' 는 파이프라인 후반의 [TRADING ENVELOPE] 에서
    //   산출되는 값이라 진입 시점에는 존재하지 않습니다.
    //   그 결과 key_secondary() 와 key_primary() 가 같아져
    //   폴백 3단이 1단으로 붕괴하고, ASE 가 1차 임계(12건)가 아니라
    //   2차 임계(30건)를 요구하게 됩니다.
    //
    //  ── 대체 축 ──
    //   문서의 출처(파일명 또는 도메인)를 2차 키로 씁니다.
    //   같은 벤더가 보낸 서식은 레이아웃이 같으므로
    //   '발행처별 격리' 라는 기획 2-3 의 의도와도 일치합니다.
    //   추출할 수 없으면 빈 문자열로 두고 1차 스코프만 씁니다.
    {
        // 🌟 [위치 의존성 제거] task_data 대신 task.data_json 을 그 자리에서 파싱합니다.
        //
        //  ── 왜 이렇게 바꾸는가 ──
        //   task_data 는 이 함수의 더 아래에서
        //     let mut task_data: Value = serde_json::from_str(&task.data_json)...
        //   로 선언됩니다. 그보다 앞에서 참조하면 E0425 가 납니다.
        //   블록을 아래로 옮겨도 되지만, 그러면 이 코드가 다시 '특정 줄 아래' 라는
        //   위치 조건에 묶입니다. 실제로 그 조건을 세 번 연속으로 틀렸습니다.
        //   여기서 직접 파싱하면 함수 어느 지점에 두어도 컴파일되고 동작합니다.
        //
        //  ── 비용 ──
        //   태스크당 JSON 파싱 1회입니다. data_json 은 수 KB 수준이고
        //   이 함수는 문서 1건에 한 번만 실행되므로 무시할 수 있습니다.
        //
        //  ── 2차 스코프를 origin 으로 두는 이유 ──
        //   task.cc 는 이 시점에 비어 있습니다.
        //   (로그의 cc='0xa276...' 는 후반의 [TRADING ENVELOPE] 산출값입니다)
        //   그대로 두면 key_secondary() 와 key_primary() 가 같아져
        //   폴백 3단이 1단으로 붕괴합니다.
        //   같은 벤더가 보낸 서식은 레이아웃이 같으므로 발행처를 2차 키로 씁니다.
        let scope_src: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));
        let origin_key = scope_src
            .get("origin")
            .or_else(|| scope_src.get("domain"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                scope_src
                    .get("image_path")
                    .and_then(|v| v.as_str())
                    .and_then(|p| {
                        std::path::Path::new(p)
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .map(|s| s.to_string())
                    })
            })
            .map(|s| {
                // 파일명 뒤의 페이지/일련 번호는 제거해 같은 벤더가 한 스코프로 모이게 합니다.
                // 예) "Shipping_Advice_Part14" → "Shipping_Advice_Part"
                s.chars()
                    .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-' || *c == '.')
                    .collect::<String>()
                    .trim_end_matches(|c: char| c.is_ascii_digit() || c == '_' || c == '-')
                    .to_string()
            })
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| task.cc.clone());
        crate::utils::score_dynamics::enter_scope(
            &team_id,
            crate::utils::score_dynamics::Track::Trading,
            "shipping",
            &origin_key,
        );
    }
    let payload = json!({
        "task_id": task.id,
        "task_type": task.r#type,
        "category": "Processing", "summary": "Starting trading extraction...", "spinner": "⠋"
    });
    let _ = app_handle.emit("extraction-progress", &payload);
    log_task_progress(app_handle, &task.id, &payload);

    if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

    let mut task_data: Value = serde_json::from_str(&task.data_json).unwrap_or(json!({}));
    let language = "english";
    let mut doc_lang = "en".to_string();

    
    let model = {
        println!("[TRADING] 🛡️ Attempting to acquire Model Lock...");
        let mut model_lock = model_mutex.lock().await;
        println!("[TRADING] ✅ Model Lock acquired.");
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
        if let Some(m) = model_lock.as_ref() {
            let wants_cpu = device_preference.as_deref() == Some("cpu");
            if m.is_cpu_mode != wants_cpu {
                println!("[TRADING] Device preference mismatch. Reloading model...");
                m.deep_purge_resources().await;
                *model_lock = None;
            }
        }
        if model_lock.is_none() {
            println!("[TRADING] Model not initialized. Starting LogisModel::new...");
            log_task_progress(app_handle, &task.id, &json!({ "category": "Loading Model", "summary": "Initializing AI Core..." }));
            match LogisModel::new(app_handle.clone(), device_preference.as_deref()).await {
                Ok(m) => {
                    println!("[TRADING] LogisModel::new successful.");
                    *model_lock = Some(m);
                },
                Err(e) => {
                    println!("[TRADING] ❌ LogisModel::new failed: {}", e);
                    return Err(anyhow::anyhow!("Model Load Failed: {}", e));
                }
            }
        }
        model_lock.as_ref().unwrap().clone()
    };

    let page_htmls: Vec<String> = if let Some(raw_html) = task_data.get("html").and_then(|s| s.as_str()) {
        let content = raw_html.to_string();
        if let Some(obj) = task_data.as_object_mut() {
            obj.remove("html");
        }
        vec![content]
    } else if task.r#type == "document_extraction" {
        let file_path = task_data.get("image_path").and_then(|s| s.as_str()).unwrap_or("");
        let ext = task_data.get("document_ext").and_then(|s| s.as_str()).unwrap_or("");
        let payload = json!({
            "task_id": task.id,
            "category": "Document Parsing",
            "summary": format!("Splitting {} file into pages...", ext.to_uppercase()),
            "spinner": "📄"
        });
        let _ = app_handle.emit("extraction-progress", &payload);
        log_task_progress(app_handle, &task.id, &payload);

        let pages = crate::parsers::extract_document_pages(file_path)
            .map_err(|e| anyhow::anyhow!("Trading document parsing failed: {}", e))?;

        let mut out: Vec<String> = Vec::with_capacity(pages.len());
        let mut total_pairs = 0usize;
        for (pi, page_text) in pages.iter().enumerate() {
            if page_text.trim().is_empty() {
                emit_term(&format!("[TRADING] ⚪ {}페이지는 추출 가능한 텍스트가 없어 건너뜁니다.", pi + 1));
                continue;
            }
            let (fake_html, pair_cnt) = pdf_page_to_structured_html(page_text);
            total_pairs += pair_cnt;
            out.push(format!("<html><body>{}</body></html>", fake_html));
        }
        if out.is_empty() {
            return Err(anyhow::anyhow!(
                "Trading document '{}' produced no usable page after splitting.",
                file_path
            ));
        }
        emit_term(&format!(
            "[TRADING] 📄 문서를 {}개 페이지로 분해했습니다. 라벨-값 행 {}개를 구조로 복원했습니다.",
            out.len(), total_pairs
        ));
        out
    } else {
        return Err(anyhow::anyhow!(
            "Trading extraction requires HTML content or a document file in task data"
        ));
    };

    let (url, _origin_candidate) = crate::utils::url_utils::resolve_absolute_url(&task_data).await;

    let total_pages = page_htmls.len();
    let mut page_results: Vec<(String, String, serde_json::Map<String, Value>)> = Vec::new();

    
    
    for (page_idx, page_html) in page_htmls.iter().enumerate() {
    let raw_html_content: &str = page_html.as_str();
    let page_label = format!("p{}", page_idx + 1);

    emit_term(&format!("\n[TRADING PAGE {}/{}] ▶ 페이지 단위 추출 시작", page_idx + 1, total_pages));
    let payload_page = json!({
        "task_id": task.id,
        "category": format!("Page {}/{}", page_idx + 1, total_pages),
        "summary": "Classifying and extracting this page...",
        "spinner": "📄"
    });
    let _ = app_handle.emit("extraction-progress", &payload_page);
    log_task_progress(app_handle, &task.id, &payload_page);

    if cancellation_token.load(Ordering::Relaxed) {
        return Err(anyhow::anyhow!("Task cancelled"));
    }

    let clean_html_content = parsing::pre_clean_html(&raw_html_content);

    
    
    let raw_pug =
        parsing::convert_to_clean_pug(&clean_html_content, PugMode::NoAttributesMode, Some(&url));
    let light_pug = model
        .truncate_pug_context(&raw_pug, false, 2000, None)
        .await;

    
    doc_lang = crate::utils::lang_utils::detect_document_language(&light_pug);
    println!("[TRADING] Detected document language (page {}): {}", page_idx + 1, doc_lang);
    emit_term("[TRADING STEP A] Classifying trade document type (2-depth)...");
    log_task_progress(app_handle, &task.id, &json!({
        "category": "Classification", "summary": "Identifying trade document group...", "spinner": "⠋"
    }));
    
    model.check_embedding_downloaded().await?;
    model.ensure_embedding().await?;
    if page_idx > 0 && !page_results.is_empty() {
        let humanize_c = |raw: &str| -> String {
            let h = crate::utils::ai_utils::humanize_url_token(raw);
            if h.trim().is_empty() { raw.trim().to_string() } else { h }
        };
        let mut title_top = f32::MIN;
        {
            let vals = resolve_title_values(&model, &light_pug, 0.30, &emit_term, false).await;
            if !vals.is_empty() {
                // 🌟 [DUPLICATE EMBED FIX] 같은 배치를 두 번 부르던 중복 호출을 제거합니다.
                let ve = model.get_embedding_batch(vals.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; vals.len()]);
                let mut tb: Vec<(String, String, String)> = Vec::new();
                let mut tp: Vec<(String, String, String)> = Vec::new();
                for (code, title) in TRADE_DOC_TITLES.iter() {
                    tb.push(("title".to_string(), code.to_string(), title.to_string()));
                    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
                    for (other, ot) in TRADE_DOC_TITLES.iter() {
                        if other == code { continue; }
                        if seen.insert(ot) {
                            tp.push(("title".to_string(), code.to_string(), ot.to_string()));
                        }
                    }
                }
                let mut uq: Vec<String> = Vec::new();
                for (_, _, p) in tb.iter().chain(tp.iter()) {
                    if !uq.iter().any(|e| e == p) { uq.push(p.clone()); }
                }
                let ue = model.get_embedding_batch(uq.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; uq.len()]);
                let te = |p: &str| -> Vec<f32> {
                    match uq.iter().position(|e| e == p) {
                        Some(i) => ue[i].clone(),
                        None => vec![0.0f32; 384],
                    }
                };
                let tbb: Vec<(String, String, Vec<f32>)> =
                    tb.iter().map(|(c, k, p)| (c.clone(), k.clone(), te(p))).collect();
                let tpb: Vec<(String, String, Vec<f32>)> =
                    tp.iter().map(|(c, k, p)| (c.clone(), k.clone(), te(p))).collect();
                let (t_keys, t_net, _) = crate::utils::ai_utils::bank_neutral_key_matrix(
                    &ve, &tbb, &tpb,
                );
                if !t_keys.is_empty() {
                    let q = ve.len();
                    // ── ① net 전체의 표준편차로 단위를 z 로 통일 ──
                    let net_sd = {
                        let mut sum = 0.0f64;
                        let mut sq = 0.0f64;
                        let mut cnt = 0usize;
                        for ki in 0..t_keys.len() {
                            for qi in 0..q {
                                let v = t_net[ki][qi];
                                if v == f32::MIN { continue; }
                                sum += v as f64;
                                sq += (v as f64) * (v as f64);
                                cnt += 1;
                            }
                        }
                        if cnt < 2 {
                            1.0f32
                        } else {
                            let mu = sum / cnt as f64;
                            let var = (sq / cnt as f64 - mu * mu).max(0.0);
                            (var.sqrt() as f32).max(1e-6)
                        }
                    };
                    // ── ② 극값 기대치 차감 ──
                    let draws = t_keys.len().saturating_mul(q);
                    let base = crate::utils::ai_utils::gumbel_expected_z(draws);
                    let mut best = f32::MIN;
                    for ki in 0..t_keys.len() {
                        for qi in 0..q {
                            let v = t_net[ki][qi];
                            if v == f32::MIN { continue; }
                            if v > best { best = v; }
                        }
                    }
                    if best != f32::MIN {
                        title_top = best / net_sd - base;
                        emit_term(&format!(
                            "     📐 [PAGE CONTINUITY / EVT] 표제 후보 {}개 | 키 {}개(draw {} → 기대 최댓값 {:.4}) | net {:+.4} → z {:+.4} → 보정 {:+.4}",
                            q, t_keys.len(), draws, base, best, best / net_sd, title_top
                        ));
                    }
                }
            }
        }
        // ── ② 자기 문서번호 라벨 축 ──
        let mut self_id_top = f32::MIN;
        {
            let pairs_all: Vec<&str> = light_pug.lines().collect();
            let dp = crate::utils::ai_utils::collect_detail_label_value_pairs(&pairs_all);
            let mut labels: Vec<String> = Vec::new();
            for p in dp.iter() {
                if p.label.trim().is_empty() { continue; }
                let t = humanize_c(&p.label);
                if !labels.iter().any(|e| e == &t) { labels.push(t); }
            }
            if !labels.is_empty() {
                let sid = model
                    .get_embedding_batch(vec![
                        TRADE_SELF_ID_LABEL_ANCHOR.to_string(),
                        TRADE_REFERENCE_LABEL_ANCHOR.to_string(),
                    ])
                    .await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; 2]);
                let le = model.get_embedding_batch(labels.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; labels.len()]);
                for (i, l) in labels.iter().enumerate() {
                    let ss = crate::utils::ai_utils::cosine_similarity(&le[i], &sid[0]);
                    let rs = crate::utils::ai_utils::cosine_similarity(&le[i], &sid[1]);
                    if ss > rs && ss > self_id_top {
                        self_id_top = ss;
                        let _ = l;
                    }
                }
            }
        }
        let verdict = PageIdentityVerdict {
            title_top,
            self_id_top,
            is_standalone: title_top > 0.0 || self_id_top > f32::MIN,
        };
        emit_term(&format!(
            "  🧭 [PAGE CONTINUITY] 표제 축: {} | 자기번호 축: {} → {}",
            if verdict.title_top == f32::MIN { "없음".to_string() } else { format!("{:+.4}", verdict.title_top) },
            if verdict.self_id_top == f32::MIN { "없음".to_string() } else { format!("{:+.4}", verdict.self_id_top) },
            if verdict.is_standalone { "독립 문서" } else { "앞 문서의 연속" }
        ));
        {
            if verdict.title_top != f32::MIN {
                crate::utils::score_dynamics::record_baseline("trading.continuation_title_top", verdict.title_top);
            }
            if verdict.self_id_top != f32::MIN {
                crate::utils::score_dynamics::record_baseline("trading.continuation_selfid_top", verdict.self_id_top);
            }
            crate::utils::score_dynamics::record_baseline(
                "trading.continuation_standalone",
                if verdict.is_standalone { 1.0 } else { 0.0 },
            );
            let a = if verdict.title_top == f32::MIN { 0.0 } else { verdict.title_top };
            let b = if verdict.self_id_top == f32::MIN { 0.0 } else { verdict.self_id_top };
            crate::utils::score_dynamics::record_decay(
                "trading.page_identity",
                &[a.max(b), a.min(b)],
            );
        }
        if !verdict.is_standalone {
            let prev_type = page_results.last().map(|(t, _, _)| t.clone()).unwrap_or_default();
            let prev_lang = page_results.last().map(|(_, l, _)| l.clone()).unwrap_or_else(|| doc_lang.clone());
            emit_term(&format!(
                "  🔗 [PAGE CONTINUATION] {}페이지는 표제도 자기 문서번호도 없어 독립 문서가 될 수 없습니다. 앞 문서 '{}' 의 연속으로 처리하며 STEP A(문서분류)를 건너뜁니다.",
                page_idx + 1, prev_type
            ));
            let carry_type = prev_type.clone();
            let carry_lang = prev_lang.clone();
            let cont_map = extract_continuation_page(
                &model, &clean_html_content, &url, &carry_type, &carry_lang,
                &task.id, page_idx, cancellation_token, app_handle, &emit_term,
            ).await?;
            page_results.push((carry_type, carry_lang, cont_map));
            continue;
        }
    }
    
    use crate::logic::{TRADE_GROUPS, TRADE_GROUP_CODES as GROUP_CODES, TRADE_DOC_TITLES, trade_code_anchor};

    let doc_lines: Vec<String> = {
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut out: Vec<String> = Vec::new();
        for line in light_pug.lines() {
            let t = match line.find('|') {
                Some(p) => line[p + 1..].trim(),
                None => continue,
            };
            if t.chars().count() < 2 { continue; }
            let key = t.to_string();
            if !seen.insert(key.clone()) { continue; }
            out.push(key);
            if out.len() >= 200 { break; }
        }
        if out.is_empty() {
            out.push(light_pug.chars().take(2000).collect::<String>());
        }
        out
    };
    emit_term(&format!("  🧱 [TRADE QUERY LINES] 판정 대상 라인 {}개", doc_lines.len()));

    let line_embs = model.get_embedding_batch(doc_lines.clone()).await
        .unwrap_or_else(|_| vec![vec![0.0; 384]; doc_lines.len()]);

    
    let (has_trade_marker, trade_markers) = trade_structural_evidence(&light_pug);
    if has_trade_marker {
        emit_term(&format!("  🔩 [TRADE STRUCTURE] 국제 표준 포맷 증거 발견: {:?}", trade_markers));
    } else {
        emit_term("  ⚪ [TRADE STRUCTURE] 국제 표준 포맷 증거가 없습니다. (택배 라벨 가능성 열림)");
    }

    let mut g_bias_defs: Vec<(String, String, String)> = Vec::new();
    let mut g_prej_defs: Vec<(String, String, String)> = Vec::new();
    for (gname, raw) in TRADE_GROUPS.iter() {
        for p in crate::utils::ai_utils::split_bias_phrases_full(raw) {
            g_bias_defs.push(("group".to_string(), gname.to_string(), p));
        }
        for (other, other_raw) in TRADE_GROUPS.iter() {
            if other == gname { continue; }
            for p in crate::utils::ai_utils::split_bias_phrases_full(other_raw) {
                g_prej_defs.push(("group".to_string(), gname.to_string(), p));
            }
            let _ = other_raw;
        }
    }

    let mut uniq_group_phrases: Vec<String> = Vec::new();
    for (_, _, p) in g_bias_defs.iter().chain(g_prej_defs.iter()) {
        if !uniq_group_phrases.iter().any(|e| e == p) { uniq_group_phrases.push(p.clone()); }
    }
    let uniq_group_embs = model.get_embedding_batch(uniq_group_phrases.clone()).await
        .unwrap_or_else(|_| vec![vec![0.0; 384]; uniq_group_phrases.len()]);
    let group_phrase_emb = |p: &str| -> Vec<f32> {
        match uniq_group_phrases.iter().position(|e| e == p) {
            Some(i) => uniq_group_embs[i].clone(),
            None => vec![0.0f32; 384],
        }
    };

    let g_bias_bank: Vec<(String, String, Vec<f32>)> = g_bias_defs.iter()
        .map(|(c, k, p)| (c.clone(), k.clone(), group_phrase_emb(p))).collect();
    let g_prej_bank: Vec<(String, String, Vec<f32>)> = g_prej_defs.iter()
        .map(|(c, k, p)| (c.clone(), k.clone(), group_phrase_emb(p))).collect();
    let group_scores_raw = crate::utils::ai_utils::bank_neutral_key_scores(
        &line_embs, &g_bias_bank, &g_prej_bank,
    );
    let mut group_scores: Vec<(String, f32)> = group_scores_raw;
    if group_scores.is_empty() {
        group_scores.push(("shipping".to_string(), 0.0));
    }
    for (g, s) in group_scores.iter() {
        emit_term(&format!("  📐 [TRADE GROUP] {} | Score(bank-neutral): {:+.4}", g, s));
    }

    let mut best_group = group_scores[0].0.clone();
    let group_margin = group_scores[0].1
        - group_scores.get(1).map(|x| x.1).unwrap_or(group_scores[0].1);

    
    
    
    if best_group == "parcel" && has_trade_marker {
        if let Some((alt, alt_s)) = group_scores.iter().find(|(g, _)| g != "parcel").cloned() {
            emit_term(&format!(
                "  🚫 [TRACKING VETO] 구조 증거 {:?} 가 존재하므로 parcel 을 거부하고 '{}'({:+.4}) 로 교체합니다.",
                trade_markers, alt, alt_s
            ));
            best_group = alt;
        }
    }

    emit_term(&format!("  👑 [TRADE GROUP SELECTED] '{}' | Top: {:+.4} | Margin: {:+.4}",
        best_group, group_scores[0].1, group_margin));

    let mut codes: Vec<&str> = GROUP_CODES.iter()
        .find(|(g, _)| *g == best_group)
        .map(|(_, c)| c.to_vec())
        .unwrap_or_else(|| vec!["Unknown"]);
    let expand_gate = {
        let pos: Vec<f32> = group_scores.iter().map(|(_, s)| *s).filter(|s| *s > 0.0).collect();
        if pos.len() < 4 {
            0.0f32
        } else {
            let n = pos.len() as f32;
            let mean = pos.iter().sum::<f32>() / n;
            let var = pos.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n;
            mean + var.sqrt()
        }
    };
    emit_term(&format!(
        "  🚧 [GROUP EXPANSION GATE] 확장 임계 = 양수 그룹 점수의 (평균 + 표준편차) = {:+.4}",
        expand_gate
    ));
    for (g, s) in group_scores.iter() {
        if g == &best_group { continue; }
        if *s <= expand_gate {
            emit_term(&format!(
                "    ⚪ [GROUP EXPANSION SKIP] '{}' ({:+.4}) 는 임계 이하라 코드 후보를 열지 않습니다.",
                g, s
            ));
            continue;
        }
        if g == "parcel" && has_trade_marker { continue; }
        if let Some((_, extra)) = GROUP_CODES.iter().find(|(gn, _)| gn == g) {
            for c in extra.iter() {
                if !codes.iter().any(|x| x == c) { codes.push(c); }
            }
        }
    }

    let mut title_scores: Vec<(String, f32)> = Vec::new();
    {
        let cands = collect_title_candidates(&light_pug, 0.30);
        emit_term(&format!("  🪪 [TITLE CANDIDATES] 상단 밴드 표제 후보 {}개 수집", cands.len()));
        if !cands.is_empty() {
            let humanize = |raw: &str| -> String {
                let h = crate::utils::ai_utils::humanize_url_token(raw);
                if h.trim().is_empty() { raw.trim().to_string() } else { h }
            };
            let mut label_texts: Vec<String> = Vec::new();
            for c in cands.iter() {
                if c.label.trim().is_empty() { continue; }
                let t = humanize(&c.label);
                if !label_texts.iter().any(|e| e == &t) { label_texts.push(t); }
            }
            let anchor_embs = model
                .get_embedding_batch(vec![
                    TRADE_TITLE_LABEL_ANCHOR.to_string(),
                    TRADE_REFERENCE_LABEL_ANCHOR.to_string(),
                ])
                .await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; 2]);
            let label_embs = if label_texts.is_empty() {
                Vec::new()
            } else {
                model.get_embedding_batch(label_texts.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; label_texts.len()])
            };
            let mut label_is_title: std::collections::HashMap<String, bool> =
                std::collections::HashMap::new();
            for (li, lt) in label_texts.iter().enumerate() {
                let ts = crate::utils::ai_utils::cosine_similarity(&label_embs[li], &anchor_embs[0]);
                let rs = crate::utils::ai_utils::cosine_similarity(&label_embs[li], &anchor_embs[1]);
                let keep = ts > rs;
                label_is_title.insert(lt.clone(), keep);
                emit_term(&format!(
                    "     🏷️ [TITLE LABEL GATE] '{}' | 자기선언 {:.4} vs 타문서참조 {:.4} → {}",
                    lt, ts, rs,
                    if keep { "표제 후보 유지" } else { "참조 라벨 → 제외" }
                ));
            }
            // 🌟 [T-1] 값 축 후보를 강가중군과 약가중군으로 나눕니다.
            //
            //  ── 왜 나누는가 ──
            //   bank_neutral_key_scores 는 뱅크 중립화(행/열 센터링)를 내부에서
            //   수행합니다. 강한 값과 약한 값을 한 배열에 섞으면 약한 값이
            //   기준선을 끌어내려, 정작 정확한 후보의 대비까지 깎입니다.
            //   따라서 각각 독립적으로 채점한 뒤 가중 합성해야
            //   '약한 후보를 살리되 강한 후보를 해치지 않는다' 가 성립합니다.
            //
            //  ── 이 함수의 로컬 판정을 그대로 두는 이유 ──
            //   여기의 label_is_title / cands 는 resolve_title_values_weighted 와
            //   별개로 이 스코프에서 이미 산출되어 있습니다.
            //   그 판정을 버리고 함수를 다시 부르면 임베딩이 중복되므로,
            //   가중치만 동일 규칙으로 여기서 유도합니다.
            let mut value_texts: Vec<String> = Vec::new();
            let mut value_is_headless: Vec<bool> = Vec::new();
            for c in cands.iter() {
                let keep = if c.label.trim().is_empty() {
                    true
                } else {
                    label_is_title.get(&humanize(&c.label)).copied().unwrap_or(false)
                };
                if !keep { continue; }
                if !value_texts.iter().any(|e| e == &c.value) {
                    value_texts.push(c.value.clone());
                    value_is_headless.push(c.label.trim().is_empty());
                }
            }
            if value_texts.is_empty() {
                emit_term("  ⚪ [TITLE AXIS] 표제 후보가 전부 참조 라벨로 판정되어 제목 축을 사용하지 않습니다.");
            } else {
                emit_term(&format!(
                    "  🪪 [TITLE AXIS] 표제 후보 값 {}개: {:?}",
                    value_texts.len(),
                    value_texts.iter().take(8).collect::<Vec<_>>()
                ));
                let val_embs = model.get_embedding_batch(value_texts.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; value_texts.len()]);
                let mut t_bias: Vec<(String, String, String)> = Vec::new();
                let mut t_prej: Vec<(String, String, String)> = Vec::new();
                for (code, title) in TRADE_DOC_TITLES.iter() {
                    t_bias.push(("title".to_string(), code.to_string(), title.to_string()));
                    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
                    for (other, other_title) in TRADE_DOC_TITLES.iter() {
                        if other == code { continue; }
                        if seen.insert(other_title) {
                            t_prej.push(("title".to_string(), code.to_string(), other_title.to_string()));
                        }
                    }
                }
                let mut uniq_t: Vec<String> = Vec::new();
                for (_, _, p) in t_bias.iter().chain(t_prej.iter()) {
                    if !uniq_t.iter().any(|e| e == p) { uniq_t.push(p.clone()); }
                }
                let uniq_t_embs = model.get_embedding_batch(uniq_t.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; uniq_t.len()]);
                let t_emb = |p: &str| -> Vec<f32> {
                    match uniq_t.iter().position(|e| e == p) {
                        Some(i) => uniq_t_embs[i].clone(),
                        None => vec![0.0f32; 384],
                    }
                };
                let t_bias_bank: Vec<(String, String, Vec<f32>)> = t_bias.iter()
                    .map(|(c, k, p)| (c.clone(), k.clone(), t_emb(p))).collect();
                let t_prej_bank: Vec<(String, String, Vec<f32>)> = t_prej.iter()
                    .map(|(c, k, p)| (c.clone(), k.clone(), t_emb(p))).collect();
                // 🌟 [T-1] 라벨 없는 후보의 바닥 판정을 여기서도 가중치로 전환합니다.
                //
                //  ── 판정 근거는 resolve_title_values_weighted 와 동일합니다 ──
                //   자기선언 앵커와의 코사인 분포에서 평균을 바닥으로 삼고,
                //   미달분은 min-max 백분위를 가중치로 부여합니다.
                //   새 상수를 만들지 않기 위해 두 곳이 같은 규칙을 씁니다.
                let title_anchor_emb = model
                    .get_embedding(TRADE_TITLE_LABEL_ANCHOR.to_string())
                    .await
                    .unwrap_or_else(|_| vec![0.0f32; 384]);
                let mut self_decl: Vec<f32> = Vec::with_capacity(value_texts.len());
                for (vi, _) in value_texts.iter().enumerate() {
                    if val_embs[vi].iter().all(|&x| x == 0.0) {
                        self_decl.push(f32::MIN);
                    } else {
                        self_decl.push(crate::utils::ai_utils::cosine_similarity(
                            &val_embs[vi], &title_anchor_emb,
                        ));
                    }
                }
                let live: Vec<f32> = self_decl.iter().cloned().filter(|s| *s > f32::MIN).collect();
                let t_floor = if live.len() < 4 {
                    0.0f32
                } else {
                    live.iter().sum::<f32>() / (live.len() as f32)
                };
                let (t_min, t_max) = if live.is_empty() {
                    (0.0f32, 0.0f32)
                } else {
                    (
                        live.iter().cloned().fold(f32::MAX, f32::min),
                        live.iter().cloned().fold(f32::MIN, f32::max),
                    )
                };

                let mut strong_embs: Vec<Vec<f32>> = Vec::new();
                let mut weak_embs: Vec<Vec<f32>> = Vec::new();
                let mut weak_w_max = 0.0f32;
                let mut weak_names: Vec<String> = Vec::new();
                for (vi, v) in value_texts.iter().enumerate() {
                    // 라벨이 표제로 확정된 후보는 바닥 판정 대상이 아니므로 항상 강가중입니다.
                    if !value_is_headless[vi] || self_decl[vi] == f32::MIN || self_decl[vi] >= t_floor {
                        strong_embs.push(val_embs[vi].clone());
                        continue;
                    }
                    let span = (t_max - t_min).max(1e-6);
                    let w = ((self_decl[vi] - t_min) / span).clamp(0.0, 1.0);
                    if w > weak_w_max { weak_w_max = w; }
                    weak_embs.push(val_embs[vi].clone());
                    weak_names.push(format!("{}(w{:.2})", v, w));
                }

                if !strong_embs.is_empty() {
                    title_scores = crate::utils::ai_utils::bank_neutral_key_scores(
                        &strong_embs, &t_bias_bank, &t_prej_bank,
                    );
                }
                // 🌟 [T-1 / 약가중 합성] 약가중군을 독립 채점한 뒤 가중 합산합니다.
                //
                //  ── 실측 표적 ──
                //   SA 문서에서 'Shipping Advice' 가 이 경로로 되살아납니다.
                //   그 값은 TRADE_DOC_TITLES 의 ("SA", "shipping advice") 와
                //   사실상 동일 문자열이므로 약가중이어도 SA 코드 점수를
                //   본문 축이 뒤집을 수 없는 수준으로 끌어올립니다.
                if !weak_embs.is_empty() && weak_w_max > 0.0 {
                    let weak_scores = crate::utils::ai_utils::bank_neutral_key_scores(
                        &weak_embs, &t_bias_bank, &t_prej_bank,
                    );
                    emit_term(&format!(
                        "  🪶 [TITLE WEAK MERGE] 바닥({:.4}) 미달 후보 {}개를 가중치 {:.3} 로 표제 축에 합류시킵니다: {:?}",
                        t_floor, weak_embs.len(), weak_w_max,
                        weak_names.iter().take(6).collect::<Vec<_>>()
                    ));
                    if title_scores.is_empty() {
                        title_scores = weak_scores
                            .into_iter()
                            .map(|(c, s)| (c, s * weak_w_max))
                            .collect();
                    } else {
                        for (c, s) in weak_scores.into_iter() {
                            match title_scores.iter_mut().find(|(tc, _)| *tc == c) {
                                Some(slot) => slot.1 += s * weak_w_max,
                                None => title_scores.push((c, s * weak_w_max)),
                            }
                        }
                    }
                    title_scores.sort_by(|a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
                for (c, s) in title_scores.iter().take(6) {
                    emit_term(&format!("     📐 [TITLE AXIS] {} | Score: {:+.4}", c, s));
                }
                // 🌟 [SDS 계측] 표제 축의 감쇠 곡선과 바닥을 남깁니다.
                {
                    let arr: Vec<f32> = title_scores.iter().map(|(_, s)| *s).collect();
                    if arr.len() >= 2 {
                        crate::utils::score_dynamics::record_decay("trading.title_axis", &arr);
                    }
                    crate::utils::score_dynamics::record_baseline("trading.title_floor", t_floor);
                }
            }
        }
    }
    // 🌟 [TITLE RESCUE] 제목 축 승자가 그룹 후보 밖이면 후보로 편입합니다.
    //    그룹 판정이 틀렸을 때의 유일한 복구 경로입니다.
    if let Some((tc, ts)) = title_scores.first().cloned() {
        let second = title_scores.get(1).map(|x| x.1).unwrap_or(ts);
        if ts > 0.0 && ts > second && !codes.iter().any(|x| *x == tc.as_str()) {
            if let Some(entry) = TRADE_DOC_TITLES.iter().find(|(c, _)| *c == tc.as_str()) {
                emit_term(&format!(
                    "  🛟 [TITLE RESCUE] 제목 축 승자 '{}' ({:+.4}, 2위 대비 {:+.4}) 가 그룹 후보에 없어 편입합니다.",
                    tc, ts, ts - second
                ));
                codes.push(entry.0);
            }
        }
    }
    emit_term(&format!("  🎯 [TRADE CODE CANDIDATES] {}개 {:?}", codes.len(), codes));

    
    let mut c_bias_defs: Vec<(String, String, String)> = Vec::new();
    let mut c_prej_defs: Vec<(String, String, String)> = Vec::new();
    for c in codes.iter() {
        for p in crate::utils::ai_utils::split_bias_phrases_full(trade_code_anchor(c)) {
            c_bias_defs.push(("code".to_string(), c.to_string(), p));
        }
        for other in codes.iter() {
            if other == c { continue; }
            for p in crate::utils::ai_utils::split_bias_phrases_full(trade_code_anchor(other)) {
                c_prej_defs.push(("code".to_string(), c.to_string(), p));
            }
        }
    }

    let mut uniq_code_phrases: Vec<String> = Vec::new();
    for (_, _, p) in c_bias_defs.iter().chain(c_prej_defs.iter()) {
        if !uniq_code_phrases.iter().any(|e| e == p) { uniq_code_phrases.push(p.clone()); }
    }
    let uniq_code_embs = model.get_embedding_batch(uniq_code_phrases.clone()).await
        .unwrap_or_else(|_| vec![vec![0.0; 384]; uniq_code_phrases.len()]);
    let code_phrase_emb = |p: &str| -> Vec<f32> {
        match uniq_code_phrases.iter().position(|e| e == p) {
            Some(i) => uniq_code_embs[i].clone(),
            None => vec![0.0f32; 384],
        }
    };

    let c_bias_bank: Vec<(String, String, Vec<f32>)> = c_bias_defs.iter()
        .map(|(c, k, p)| (c.clone(), k.clone(), code_phrase_emb(p))).collect();
    let c_prej_bank: Vec<(String, String, Vec<f32>)> = c_prej_defs.iter()
        .map(|(c, k, p)| (c.clone(), k.clone(), code_phrase_emb(p))).collect();
    let mut code_scores: Vec<(String, f32)> = crate::utils::ai_utils::bank_neutral_key_scores(
        &line_embs, &c_bias_bank, &c_prej_bank,
    );
    if code_scores.is_empty() {
        code_scores.push((codes[0].to_string(), 0.0));
    }
    if !title_scores.is_empty() {
        // 🌟 [T-1 / 역분산 융합] 고정 가중치 2.0 을 축 SNR 비율로 스케일링합니다.
        //
        //  ── 2.0 의 출처와 한계 ──
        //   기존 주석: "실측 제목 축 마진(1.3076)이 바디 축 최대 왜곡(약 1.1)보다
        //   커지도록 한 값". 즉 특정 문서 한 건에서 유도한 상수입니다.
        //   제목 축이 그 문서보다 약한 문서에서는 과대 가중이 되고,
        //   그 결과 약한 표제가 본문 축의 정당한 판정을 뒤엎을 수 있습니다.
        //
        //  ── 왜 SNR 비율인가 ──
        //   두 축의 점수는 척도가 다릅니다(본문은 bank-neutral surprisal,
        //   제목은 bank-neutral title). 절대값 비교가 불가능하므로
        //   각 축 내부에서 무차원량을 뽑아야 합니다.
        //     신호 = 1위 − 꼬리 중앙값
        //     잡음 = 꼬리(2위 이하) 표준편차
        //   두 SNR 의 비율은 척도 불변이며, 제목 축이 본문 축만큼 변별력이
        //   있을 때 1.0 이 됩니다.
        //
        //  ── 왜 상한을 1.0 으로 두는가 ──
        //   이것은 튜닝 상수가 아니라 안전 규칙입니다.
        //   '기존 동작(2.0)을 초과하지 않는다' 를 보장하면
        //   제목 축이 강한 문서(SR, CI)에서 회귀가 구조적으로 불가능해집니다.
        //   즉 이 변경은 '제목 축이 약할 때만 감쇠' 라는 단방향입니다.
        //
        //  ── SNR 을 구할 수 없으면 ──
        //   axis_snr 이 None(후보 3개 미만 또는 꼬리 균일)이면
        //   기존 상수 2.0 을 그대로 씁니다. 냉간 시작 원칙과 동일한 폴백입니다.
        let title_weight = {
            let t_arr: Vec<f32> = title_scores.iter().map(|(_, s)| *s).collect();
            let b_arr: Vec<f32> = code_scores.iter().map(|(_, s)| *s).collect();
            match (
                crate::utils::ai_utils::axis_snr(&t_arr),
                crate::utils::ai_utils::axis_snr(&b_arr),
            ) {
                (Some(t_snr), Some(b_snr)) if b_snr > 1e-6 => {
                    let ratio = (t_snr / b_snr).clamp(0.0, 1.0);
                    emit_term(&format!(
                        "  📐 [TITLE AXIS CONFIDENCE] 제목 축 SNR {:.4} / 본문 축 SNR {:.4} = 비율 {:.4} → 융합 가중치 {:.4} (상한 2.0)",
                        t_snr, b_snr, ratio, 2.0 * ratio
                    ));
                    crate::utils::score_dynamics::record_baseline("trading.title_snr", t_snr);
                    crate::utils::score_dynamics::record_baseline("trading.body_snr", b_snr);
                    crate::utils::score_dynamics::record_baseline("trading.title_weight", 2.0 * ratio);
                    2.0f32 * ratio
                }
                _ => {
                    emit_term("  📐 [TITLE AXIS CONFIDENCE] 축 SNR 을 산출할 수 없어 기존 가중치 2.0 을 유지합니다.");
                    2.0f32
                }
            }
        };
        let mut merged = 0usize;
        for (cname, cs) in code_scores.iter_mut() {
            if let Some((_, ts)) = title_scores.iter().find(|(t, _)| t == cname) {
                *cs += title_weight * ts;
                merged += 1;
            }
        }
        code_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        emit_term(&format!(
            "  🪪 [TITLE AXIS MERGE] 본문 축에 제목 축(가중치 {:.4})을 합산했습니다. (코드 {}개)",
            title_weight, merged
        ));
    }
    for (c, s) in code_scores.iter() {
        emit_term(&format!("    📐 [TRADE CODE] {} | Score(bank-neutral + title): {:+.4}", c, s));
    }

    
    
    if has_trade_marker {
        let before = code_scores.len();
        code_scores.retain(|(c, _)| c != "TRACKING");
        if code_scores.len() != before {
            emit_term("    🚫 [TRACKING VETO / CODE] 구조 증거가 있어 TRACKING 을 코드 후보에서 제거했습니다.");
        }
        if code_scores.is_empty() {
            code_scores.push(("CI".to_string(), 0.0));
        }
    }

    let cosine_code = code_scores[0].0.clone();
    let code_margin = code_scores[0].1
        - code_scores.get(1).map(|x| x.1).unwrap_or(code_scores[0].1);
    emit_term(&format!("  👑 [TRADE CODE COSINE] '{}' | Top: {:+.4} | Margin: {:+.4}",
        cosine_code, code_scores[0].1, code_margin));

    
    
    
    let doc_type = if codes.len() == 1 {
        emit_term(&format!("  ⚡ [TRADE CODE DETERMINISTIC] 그룹 '{}' 의 코드가 1개뿐이라 LLM 호출을 생략합니다.", best_group));
        cosine_code
    } else if code_margin > 0.01 {
        emit_term(&format!("  ⚡ [TRADE CODE DETERMINISTIC] 코사인 마진 {:+.4} 로 '{}' 확정. LLM 호출을 생략합니다.", code_margin, cosine_code));
        cosine_code
    } else {
        emit_term(&format!("  ⚠️ [TRADE CODE AMBIGUOUS] 코사인 마진 {:+.4} 부족. 그룹 '{}' 내 {}개 코드로 LLM 재판정합니다.",
            code_margin, best_group, codes.len()));
        model.secure_vram_relay(crate::model::ModelSize::Qwen3_5, None, Some(cancellation_token.clone()), false, None).await?;
        
        
        let base_prompt = crate::prompts::page_type_prompt("shipping");
        let scoped_prompt = {
            let mut s = String::from("[VECTOR EVIDENCE]
    The vector engine scored this document against candidate codes:
    ");
            for (c, sc) in &code_scores {
                s.push_str(&format!("- {} (vector score {:.4})
    ", c, sc));
            }
            s.push_str(&format!("
    {}", base_prompt));
            s
        };
        let picked = if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
            let params = crate::openai_types::ChatCompletionParameters {
                messages: vec![
                    crate::openai_types::ChatCompletionRequestMessage::System(
                        crate::openai_types::ChatCompletionRequestSystemMessage {
                            content: format!("[PUG CONTENT]
    {}", light_pug),
                            name: None,
                        },
                    ),
                    crate::openai_types::ChatCompletionRequestMessage::User(
                        crate::openai_types::ChatCompletionRequestUserMessage {
                            content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(
                                scoped_prompt,
                            ),
                            name: None,
                        },
                    ),
                ],
                model: "qwen3.5".to_string(),
                max_tokens: Some(1024),
                temperature: Some(0.0),
                top_p: Some(0.95),
                ..Default::default()
            };

            let res = gen
                .generate(
                    params,
                    Some(cancellation_token.clone()),
                    Some(format!("{}_{}_doctype", task.id, page_label)),
                    None,
                    None,
                    None,
                )
                .await?;
            let parsed = crate::parsing::parse_json_from_llm(&res);
            parsed
                .get("type")
                .or_else(|| parsed.get("doc_type"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string()
        } else {
            String::new()
        };
        if !picked.is_empty() && codes.iter().any(|c| *c == picked.as_str()) {
            emit_term(&format!("  🤖 [TRADE CODE LLM] LLM 이 '{}' 로 확정했습니다.", picked));
            picked
        } else {
            if !picked.is_empty() {
                emit_term(&format!("  🚫 [TRADE CODE LLM REJECT] LLM 이 반환한 '{}' 는 그룹 '{}' 후보에 없어 폐기합니다.", picked, best_group));
            }
            cosine_code
        }
    };

    emit_term(&format!("[TRADING STEP A] ✅ Document classified as: {} (group: {})", doc_type, best_group));
    crate::utils::score_dynamics::refine_primary(&doc_type);
    emit_term("[TRADING STEP B] Running PLINKO field assignment before LLM...");

    // 🌟 [OTHER PARTIES SLOT] 스키마는 9개 카테고리인데 여기만 8개였습니다.
    //    실측: ⚠️ [PLINKO WRITE MISS] 'party_address' (cat: 'other_parties') 를 기록할 루트 슬롯을 찾지 못했습니다.
    //    forwarder / carrier / bank / agent 등 sender·recipient 가 아닌 당사자가 전부 이 버킷입니다.
    // 🌟 [CATEGORY / SINGLE SOURCE] 카테고리 목록을 logic.rs 하나로 접습니다.
    //
    //  ── 무엇이 문제였나 ──
    //   trade_field_category 는 20갈래(customs / inspection / insurance / settlement /
    //   hazmat / origin / compliance / charges / test_results / findings_and_damage /
    //   account_ledger 포함)를 돌려주는데, 이 배열은 10개만 적혀 있었습니다.
    //   그래서 그 11개 카테고리로 라우팅된 필드는
    //     ⚠️ [PLINKO WRITE MISS] '...' (cat: 'customs') 를 기록할 루트 슬롯을 찾지 못했습니다.
    //   로 전량 폐기되고, LLM 카테고리 순회 대상에서도 빠져 두 번 소실되었습니다.
    //
    //  ── insurance 버킷 오염의 정체 ──
    //   CI overlay 의 financials.insurance 는 '보험료 라인' 입니다.
    //   그런데 trade_field_category 의 명시 매핑에 'insurance' 필드명이 없어
    //   규칙 폴백의 f.contains("insur") 에 걸려 'insurance' 카테고리가 됩니다.
    //   그 카테고리가 순회 대상이 아니었으므로 값이 엉뚱한 곳으로 흘렀고,
    //   검수에서 "insurance 버킷에 cargo 값(4/20)이 중복 복제" 로 관측되었습니다.
    //   목록을 단일화하면 그 필드는 정상적으로 insurance 슬롯을 갖게 되고,
    //   아래 [FINANCIAL ALIAS GUARD] 가 이름 충돌 자체를 없앱니다.
    //
    //  ── 배열 여부 판정 ──
    //   logic::is_trade_array_category 가 이미 소유한 사실이므로 여기에 복제하지 않습니다.
    let categories: Vec<&'static str> = crate::logic::TRADE_EXTRACTION_CATEGORIES.to_vec();
    let mut final_data_map = serde_json::Map::new();
    for c in categories.iter() {
        if crate::logic::is_trade_array_category(c) {
            // items 만 레거시 소비처(merge_trading_page_map / hoist_array_identifiers)가
            // line_items 라는 이름을 읽습니다. 나머지 배열은 카테고리명을 그대로 씁니다.
            let key = if *c == "items" { "line_items" } else { *c };
            final_data_map.insert(key.to_string(), json!([]));
        } else {
            final_data_map.insert(c.to_string(), json!({}));
        }
    }
    emit_term(&format!(
        "  🗂️ [CATEGORY SLOTS] logic::TRADE_EXTRACTION_CATEGORIES 기준 {}개 슬롯 생성 (배열 {}개)",
        categories.len(),
        categories.iter().filter(|c| crate::logic::is_trade_array_category(c)).count()
    ));
    final_data_map.insert("header".to_string(), json!({"doc_type": doc_type.clone()}));

    let content_pug = {
        let full_pug =
            parsing::convert_to_clean_pug(&clean_html_content, PugMode::ListMode, Some(&url));
        model
            .truncate_pug_context(&full_pug, true, 2000, None)
            .await
    };
    let pug_lines: Vec<String> = content_pug.lines().map(|s| s.to_string()).collect();
    let pug_lines_ref: Vec<&str> = pug_lines.iter().map(|s| s.as_str()).collect();

    
    let mut detail_pairs = crate::utils::ai_utils::collect_detail_label_value_pairs(&pug_lines_ref);
    {
        let anchors: Vec<(usize, usize)> = detail_pairs.iter()
            .map(|p| (p.label_line, p.primary_line)).collect();
        let recovered_secs = recover_pair_sections(&anchors, &pug_lines);
        let mut recovered = 0usize;
        for (i, p) in detail_pairs.iter_mut().enumerate() {
            if !p.section.trim().is_empty() { continue; }
            let s = match recovered_secs.get(i) { Some(s) => s, None => continue };
            if s.is_empty() { continue; }
            p.section = s.clone();
            recovered += 1;
        }
        if recovered > 0 {
            emit_term(&format!(
                "  🧭 [SECTION RECOVERY v2] 직전 미소비 텍스트 행으로 섹션 {}개 복원 (동일 라벨 중복 구분 가능)",
                recovered
            ));
        }
    }
    emit_term(&format!("  🧷 [TRADING PAIR] 구조적 라벨-값 페어 {}개 확보", detail_pairs.len()));
    for p in &detail_pairs {
        emit_term(&format!(
            "    Line {} | Section: '{}' | Label: '{}' | Value: '{}'",
            p.primary_line + 1, p.section, p.label, p.value
        ));
    }

    
    
    let trade_fields = crate::parsing::get_detail_schema_fields(&doc_type, &url, &doc_lang);
    emit_term(&format!("  📐 [TRADING SCHEMA] doc_type '{}' 에 대응하는 스키마 필드 {}개 로드", doc_type, trade_fields.len()));

    let mut t_field_names: Vec<String> = Vec::new();
    let mut t_label_embs: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut t_label_weights: Vec<Vec<f32>> = Vec::new();
    let mut t_prej_raw: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut t_prej_texts: Vec<Vec<String>> = Vec::new();

    for (fname, _, _, _) in &trade_fields {
        let (lp, lw) = crate::utils::ai_utils::label_phrase_bank(&doc_lang, &doc_type, fname);
        if lp.is_empty() { continue; }
        let pp = crate::utils::ai_utils::prejudice_phrase_bank(&doc_lang, &doc_type, fname);
        let le = model.get_embedding_batch(lp.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; lp.len()]);
        let pe = if pp.is_empty() {
            Vec::new()
        } else {
            model.get_embedding_batch(pp.clone()).await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; pp.len()])
        };
        t_field_names.push(fname.clone());
        t_label_embs.push(le);
        t_label_weights.push(lw);
        t_prej_raw.push(pe);
        t_prej_texts.push(pp);
    }

    
    let mut t_prej_embs: Vec<Vec<Vec<f32>>> = Vec::with_capacity(t_field_names.len());
    for f in 0..t_field_names.len() {
        let mask = crate::utils::ai_utils::self_poisoned_prejudice_mask(
            &t_label_embs[f], &t_prej_raw[f], &t_label_embs, f
        );
        let mut kept: Vec<Vec<f32>> = Vec::new();
        let mut dropped = 0usize;
        for (pi, poisoned) in mask.iter().enumerate() {
            if *poisoned {
                dropped += 1;
                if dropped <= 4 {
                    emit_term(&format!("    🧪 [SELF-POISON DROP] '{}' 의 편견 구 '{}' 박탈",
                        t_field_names[f], t_prej_texts[f].get(pi).cloned().unwrap_or_default()));
                }
            } else {
                kept.push(t_prej_raw[f][pi].clone());
            }
        }
        emit_term(&format!("  🏷️ [TRADING LABEL BANK] '{}' | 라벨 구 {}개 | 편견 구 {}개 (자기오염 {}개 제거)",
            t_field_names[f], t_label_embs[f].len(), kept.len(), dropped));
        t_prej_embs.push(kept);
    }

    
    let mut unique_labels: Vec<String> = Vec::new();
    let mut unique_leaf: Vec<String> = Vec::new();
    let mut unique_section: Vec<String> = Vec::new();
    let mut unique_qualified: Vec<bool> = Vec::new();
    let mut label_count: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for p in &detail_pairs { *label_count.entry(p.label.clone()).or_insert(0) += 1; }

    let mut pair_phrases: Vec<String> = Vec::with_capacity(detail_pairs.len());
    let mut qualified = 0usize;
    for p in &detail_pairs {
        let dup = label_count.get(&p.label).copied().unwrap_or(0) > 1;
        if dup && !p.section.trim().is_empty() {
            pair_phrases.push(format!("{} {}", p.section.trim(), p.label));
            qualified += 1;
        } else {
            pair_phrases.push(p.label.clone());
        }
    }
    if qualified > 0 {
        emit_term(&format!(
            "  🏷️ [SECTION-QUALIFIED] 중복 라벨 {}개를 '섹션 + 라벨' 로 한정했습니다. (동일 라벨 접합 방지)",
            qualified
        ));
    }
    for (pi, ph) in pair_phrases.iter().enumerate() {
        if unique_labels.iter().any(|e| e == ph) { continue; }
        unique_labels.push(ph.clone());
        unique_leaf.push(detail_pairs[pi].label.clone());
        unique_section.push(detail_pairs[pi].section.trim().to_string());
        unique_qualified.push(
            label_count.get(&detail_pairs[pi].label).copied().unwrap_or(0) > 1
        );
    }

    let mut assigned_fields: std::collections::HashMap<String, String> = std::collections::HashMap::new();

    if !unique_labels.is_empty() && !t_field_names.is_empty() {
        let leaf_embs = model.get_embedding_batch(unique_leaf.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; unique_leaf.len()]);
        let section_texts: Vec<String> = unique_section.iter()
            .map(|s| if s.is_empty() { " ".to_string() } else { s.clone() })
            .collect();
        let section_embs = model.get_embedding_batch(section_texts.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; section_texts.len()]);

        
        let mut phrase_single: Vec<String> = vec![String::new(); unique_labels.len()];
        let mut phrase_multi: Vec<String> = vec![String::new(); unique_labels.len()];
        let mut phrase_line: Vec<usize> = vec![0usize; unique_labels.len()];
        let mut multi_section: Vec<String> = vec![String::new(); unique_labels.len()];
        for (pi, ph) in pair_phrases.iter().enumerate() {
            let h = match unique_labels.iter().position(|u| u == ph) { Some(v) => v, None => continue };
            let p = &detail_pairs[pi];
            if phrase_single[h].is_empty() && !p.value.trim().is_empty() {
                phrase_single[h] = p.value.clone();
                phrase_line[h] = p.primary_line;
                multi_section[h] = p.section.trim().to_string();
            }
            let av = p.value_all.trim();
            if av.is_empty() { continue; }
            if phrase_multi[h].contains(av) { continue; }
            // 섹션이 다르면 접합하지 않습니다.
            let same_scope = multi_section[h].is_empty()
                || p.section.trim().is_empty()
                || multi_section[h] == p.section.trim();
            if !same_scope {
                continue;
            }
            if phrase_multi[h].is_empty() {
                phrase_multi[h] = av.to_string();
            } else {
                phrase_multi[h].push(' ');
                phrase_multi[h].push_str(av);
            }
        }

        let (label_self_cos, label_ref_cos) = {
            let self_phrases = crate::utils::ai_utils::split_bias_phrases_full(TRADE_SELF_ID_LABEL_ANCHOR);
            let ref_phrases = crate::utils::ai_utils::split_bias_phrases_full(TRADE_REFERENCE_LABEL_ANCHOR);
            let mut uniq: Vec<String> = Vec::new();
            for p in self_phrases.iter().chain(ref_phrases.iter()) {
                if !uniq.iter().any(|e| e == p) { uniq.push(p.clone()); }
            }
            let uniq_embs = model.get_embedding_batch(uniq.clone()).await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; uniq.len()]);
            let emb_of = |p: &str| -> Vec<f32> {
                match uniq.iter().position(|e| e == p) {
                    Some(i) => uniq_embs[i].clone(),
                    None => vec![0.0f32; 384],
                }
            };
            let self_embs: Vec<Vec<f32>> = self_phrases.iter().map(|p| emb_of(p)).collect();
            let ref_embs: Vec<Vec<f32>> = ref_phrases.iter().map(|p| emb_of(p)).collect();
            let gate_surface: Vec<String> = unique_labels.iter().map(|l| {
                let s: String = l.chars()
                    .map(|c| if c == '_' || c == '-' || c == '.' || c == '/' { ' ' } else { c })
                    .collect();
                s.split_whitespace().collect::<Vec<_>>().join(" ").trim().to_lowercase()
            }).collect();
            let gate_embs = model.get_embedding_batch(gate_surface.clone()).await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; gate_surface.len()]);
            let mut sc: Vec<f32> = vec![f32::MIN; unique_labels.len()];
            let mut rc: Vec<f32> = vec![f32::MIN; unique_labels.len()];
            for h in 0..unique_labels.len() {
                // 원 라벨 임베딩과 정규화 표기 임베딩 중 '참조 신호가 살아 있는 쪽' 을 씁니다.
                // 둘 다 보되 max 를 취하므로, 표기 차이로 신호가 죽는 경우만 구제됩니다.
                let mut s_best = f32::MIN;
                let mut r_best = f32::MIN;
                if !leaf_embs[h].iter().all(|&v| v == 0.0) {
                    s_best = crate::utils::ai_utils::max_pool_sim(&leaf_embs[h], &self_embs);
                    r_best = crate::utils::ai_utils::max_pool_sim(&leaf_embs[h], &ref_embs);
                }
                if !gate_embs[h].iter().all(|&v| v == 0.0) {
                    let s2 = crate::utils::ai_utils::max_pool_sim(&gate_embs[h], &self_embs);
                    let r2 = crate::utils::ai_utils::max_pool_sim(&gate_embs[h], &ref_embs);
                    // 두 표기 중 '참조 - 자기' 격차가 큰 쪽을 채택합니다.
                    // 격차가 큰 표기가 그 라벨의 방향성을 더 선명하게 담고 있습니다.
                    let d1 = if s_best == f32::MIN { f32::MIN } else { r_best - s_best };
                    let d2 = r2 - s2;
                    if d1 == f32::MIN || d2.abs() > d1.abs() {
                        s_best = s2;
                        r_best = r2;
                    }
                }
                sc[h] = s_best;
                rc[h] = r_best;
            }
            (sc, rc)
        };
        let is_self_id_field = |name: &str| -> bool {
            let n = name.trim().to_lowercase();
            n == "doc_number" || n == "document_number"
        };
        let mut self_id_gate_logged = 0usize;
        // 🌟 [DOC-CODE PREFIX] 값의 선두 알파벳을 국제 문서코드로 해석합니다.
        //    SELF-ID ANCHOR 와 FOREIGN REF SKIP 이 같은 판정을 공유합니다.
        let leading_code = |v: &str| {
            let head: String = v.trim().chars().take_while(|c| c.is_alphabetic()).collect();
            if head.is_empty() { return None; }
            crate::utils::bias_schema::canonical_trade_doc_code(&head)
        };
        let self_id_anchor_label: Option<usize> = {
            // ── 1단계: 문서코드 접두어 일치 (결정론) ──
            let mut prefix_hits: Vec<usize> = Vec::new();
            for h in 0..unique_labels.len() {
                let v = phrase_single[h].trim();
                if v.is_empty() { continue; }
                if !crate::utils::ai_utils::value_matches_format(
                    crate::utils::ai_utils::FieldFormat::Identifier, v,
                ) { continue; }
                if let Some(code) = leading_code(v) {
                    if code.eq_ignore_ascii_case(doc_type.as_str()) {
                        prefix_hits.push(h);
                    }
                }
            }
            // ── 2단계: 접두어 후보가 여럿이거나 없을 때만 코사인 격차로 가림 ──
            let pick_by_gap = |cands: &[usize]| -> Option<usize> {
                let mut best: Option<(usize, f32)> = None;
                for &h in cands {
                    if label_self_cos[h] == f32::MIN { continue; }
                    let gap = label_self_cos[h] - label_ref_cos[h];
                    match best {
                        Some((_, g)) if g >= gap => {}
                        _ => best = Some((h, gap)),
                    }
                }
                best.map(|(h, _)| h)
            };
            if prefix_hits.len() == 1 {
                let h = prefix_hits[0];
                emit_term(&format!(
                    "  🪪 [SELF-ID ANCHOR / DOC-CODE] '{}' 를 자기 문서번호로 확정합니다. | 값 \"{}\" 의 접두어가 이 문서의 doc_type '{}' 와 일치합니다.",
                    unique_labels[h], phrase_single[h], doc_type
                ));
                Some(h)
            } else if prefix_hits.len() > 1 {
                let h = pick_by_gap(&prefix_hits).unwrap_or(prefix_hits[0]);
                emit_term(&format!(
                    "  🪪 [SELF-ID ANCHOR / DOC-CODE TIE] 접두어가 '{}' 인 라벨이 {}개입니다. 자기선언-타문서참조 격차가 최대인 '{}' 를 선택합니다. | Value: \"{}\"",
                    doc_type, prefix_hits.len(), unique_labels[h], phrase_single[h]
                ));
                Some(h)
            } else {
                // 접두어 규약을 따르지 않는 서식(순수 숫자 번호 등)은 코사인으로 폴백합니다.
                let mut cands: Vec<usize> = Vec::new();
                for h in 0..unique_labels.len() {
                    if label_self_cos[h] == f32::MIN { continue; }
                    if label_self_cos[h] <= label_ref_cos[h] { continue; }
                    let v = phrase_single[h].trim();
                    if v.is_empty() { continue; }
                    if !crate::utils::ai_utils::value_matches_format(
                        crate::utils::ai_utils::FieldFormat::Identifier, v,
                    ) { continue; }
                    // 타 문서코드 접두어를 가진 값은 명시적으로 배제합니다.
                    // (CI-, PO-, LC- 가 자기 번호로 승격되는 경로를 코사인 단계에서 차단)
                    if let Some(code) = leading_code(v) {
                        if !code.eq_ignore_ascii_case(doc_type.as_str()) { continue; }
                    }
                    cands.push(h);
                }
                match pick_by_gap(&cands) {
                    Some(h) => {
                        emit_term(&format!(
                            "  🪪 [SELF-ID ANCHOR / COSINE] 접두어 근거가 없어 코사인으로 판정합니다. '{}' | 자기선언 {:.4} > 타문서참조 {:.4} | Value: \"{}\"",
                            unique_labels[h], label_self_cos[h], label_ref_cos[h], phrase_single[h]
                        ));
                        Some(h)
                    }
                    None => {
                        emit_term("  ⚪ [SELF-ID ANCHOR] 자기 문서번호로 볼 라벨이 없어 doc_number 를 비워 둡니다.");
                        None
                    }
                }
            }
        };
        let foreign_ref_label: Vec<bool> = (0..unique_labels.len()).map(|h| {
            if Some(h) == self_id_anchor_label { return false; }
            let v = phrase_single[h].trim();
            if v.is_empty() { return false; }
            if !crate::utils::ai_utils::value_matches_format(
                crate::utils::ai_utils::FieldFormat::Identifier, v,
            ) { return false; }
            match leading_code(v) {
                Some(code) => !code.eq_ignore_ascii_case(doc_type.as_str()),
                None => false,
            }
        }).collect();
        for h in 0..unique_labels.len() {
            if foreign_ref_label[h] {
                emit_term(&format!(
                    "    🚫 [FOREIGN REF SKIP] '{}' 값 \"{}\" 은 타 문서코드 접두어를 가집니다. header.reference_* 는 LLM 이 채우므로 PLINKO 배정에서 제외합니다.",
                    unique_labels[h], phrase_single[h]
                ));
            }
        }
        let row_index_section: Vec<bool> = unique_section
            .iter()
            .map(|s| is_row_index_section(s))
            .collect();
        {
            let hits: Vec<String> = (0..unique_labels.len())
                .filter(|&h| row_index_section[h])
                .map(|h| format!("{}('{}')", unique_labels[h], unique_section[h]))
                .collect();
            if !hits.is_empty() {
                emit_term(&format!(
                    "  🧾 [ROW SECTION SCOPE] 행 인덱스 섹션의 라벨 {}개는 items/containers 에만 배정합니다: {:?}",
                    hits.len(), hits.iter().take(8).collect::<Vec<_>>()
                ));
            }
        }

        // 🌟 [ROLE PARTY ANCHOR] '역할 당사자' 와 '주요 당사자' 를 가르는 두 축을 준비합니다.
        //
        //  ── 앵커 출처 ──
        //   bias.json 의 trade_schema.base.other_parties.party_role 설명문과
        //   parties 의 sender/recipient/notify 설명문을 그대로 씁니다.
        //   어휘 목록을 코드에 복제하지 않으므로 역할이 늘어도 수정 대상이 아닙니다.
        let (role_party_embs, main_party_embs, role_party_anchor_ready) = {
            let triples = crate::utils::bias_schema::canonical_trade_triples(&doc_type);
            let mut role_phrases: Vec<String> = Vec::new();
            let mut main_phrases: Vec<String> = Vec::new();
            for (category, field, desc) in triples.iter() {
                if category == "other_parties" && field == "party_role" {
                    for p in crate::utils::ai_utils::split_bias_phrases_full(desc) {
                        if !role_phrases.iter().any(|e| e == &p) { role_phrases.push(p); }
                    }
                }
                if category == "parties"
                    && (field == "sender_name" || field == "recipient_name" || field == "notify_party_name")
                {
                    for p in crate::utils::ai_utils::split_bias_phrases_full(desc) {
                        if !main_phrases.iter().any(|e| e == &p) { main_phrases.push(p); }
                    }
                }
            }
            // 스키마에서 얻지 못한 극단 상황의 최소 방어선입니다.
            if role_phrases.is_empty() {
                for p in [
                    "carrier", "insurer", "issuing bank", "advising bank", "customs broker",
                    "warehouse operator", "surveying agency", "shipping agent", "freight forwarder",
                    "drawer", "drawee", "payee", "claimant", "applicant",
                ] { role_phrases.push(p.to_string()); }
            }
            if main_phrases.is_empty() {
                for p in [
                    "shipper", "exporter", "seller", "consignor",
                    "consignee", "importer", "buyer", "receiver", "notify party",
                ] { main_phrases.push(p.to_string()); }
            }
            let re = model.get_embedding_batch(role_phrases.clone()).await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; role_phrases.len()]);
            let me = model.get_embedding_batch(main_phrases.clone()).await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; main_phrases.len()]);
            let ready = !re.is_empty() && !me.is_empty();
            emit_term(&format!(
                "  🧑‍💼 [ROLE PARTY ANCHOR] 역할 당사자 구 {}개 | 주요 당사자 구 {}개 준비 (활성: {})",
                re.len(), me.len(), if ready { "예" } else { "아니오" }
            ));
            (re, me, ready)
        };

        let pair_abs_floor = 0.50f32;
        let mut leaf_raw: Vec<Vec<f32>> = vec![vec![-1.0f32; unique_labels.len()]; t_field_names.len()];
        let mut sec_raw:  Vec<Vec<f32>> = vec![vec![-1.0f32; unique_labels.len()]; t_field_names.len()];
        for f in 0..t_field_names.len() {
            let f_fmt = crate::utils::ai_utils::detect_field_format(&t_field_names[f]);
            let f_multi = crate::utils::ai_utils::is_multi_value_field(&t_field_names[f]);
            // 🌟 [ROW SECTION SCOPE] 이 필드가 '행 속성' 축인가.
            let f_row_cat = {
                let c = crate::logic::trade_field_category(&t_field_names[f]);
                c == "items" || c == "containers"
            };
            let f_strict = matches!(
                f_fmt,
                crate::utils::ai_utils::FieldFormat::Date
                    | crate::utils::ai_utils::FieldFormat::TrackingCode
                    | crate::utils::ai_utils::FieldFormat::Numeric
                    | crate::utils::ai_utils::FieldFormat::Phone
                    | crate::utils::ai_utils::FieldFormat::Address
                    | crate::utils::ai_utils::FieldFormat::Text
            );
            let f_self_id = is_self_id_field(&t_field_names[f]);
            let f_cohesion = crate::utils::ai_utils::bank_internal_cohesion(&t_label_embs[f]);
            for h in 0..unique_labels.len() {
                if leaf_embs[h].iter().all(|&v| v == 0.0) { continue; }
                // 🌟 [ROW SECTION SCOPE] '[ Item N ]' 안의 라벨은 문서 수준 축과 겨루지 않습니다.
                if row_index_section[h] && !f_row_cat { continue; }
                let own = crate::utils::ai_utils::weighted_max_pool_sim(
                    &leaf_embs[h], &t_label_embs[f], &t_label_weights[f]
                );
                if own < pair_abs_floor { continue; }
                // 🌟 [SELF-ID GATE] 자기 문서번호 축에는 '남을 가리키는 라벨' 을 올리지 않습니다.
                if f_self_id
                    && label_self_cos[h] != f32::MIN
                    && label_ref_cos[h] > label_self_cos[h]
                {
                    if self_id_gate_logged < 8 {
                        self_id_gate_logged += 1;
                        emit_term(&format!(
                            "    🚫 [SELF-ID GATE] '{}' → '{}' | 자기번호 {:.4} < 타문서참조 {:.4} → 이 라벨은 남의 문서를 가리키므로 자기 문서번호 축에서 제외합니다.",
                            unique_labels[h], t_field_names[f], label_self_cos[h], label_ref_cos[h]
                        ));
                    }
                    // 🌟 [SDS 계측] 로그 상한(8건)과 무관하게 관측은 전량 기록합니다.
                    //    로그는 사람이 읽는 상한이 필요하지만 통계는 표본이 많을수록 정확합니다.
                    crate::utils::score_dynamics::record_field_reject(
                        &t_field_names[f],
                        crate::utils::score_dynamics::GateKind::SelfId,
                    );
                    continue;
                }
                // 🌟 [SDS 계측] 이 필드가 후보로 검토되었다는 사실 자체를 남깁니다.
                //    거절률의 분모가 되므로 게이트보다 먼저 기록해야 합니다.
                crate::utils::score_dynamics::record_field_seen(&t_field_names[f]);

                let prej = if t_prej_embs[f].is_empty() {
                    0.0
                } else {
                    crate::utils::ai_utils::max_pool_sim(&leaf_embs[h], &t_prej_embs[f])
                };
                if crate::utils::ai_utils::prejudice_dominates(own, prej, f_cohesion) {
                    emit_term(&format!("    🚫 [TRADING PREJUDICE GATE] '{}' → '{}' | Label: {:.4} | Prej: {:.4} | Cohesion: {:.4} (상대 우위 초과)",
                        unique_labels[h], t_field_names[f], own, prej, f_cohesion));
                    // 🌟 [SDS 계측] 게이트 거절은 '판정 결과' 가 아니라
                    //    '형식·구조 사실' 이므로 학습해도 자기강화 피드백(리스크 R1)이
                    //    발생하지 않습니다. Phase 2 학습형 prejudice 의 유일한 입력입니다.
                    crate::utils::score_dynamics::record_field_reject(
                        &t_field_names[f],
                        crate::utils::score_dynamics::GateKind::Prejudice,
                    );
                    continue;
                }
                let pair_val = if f_multi { &phrase_multi[h] } else { &phrase_single[h] };
                if pair_val.trim().is_empty()
                    || !crate::utils::ai_utils::value_matches_format(f_fmt, pair_val) {
                    emit_term(&format!("    🚫 [TRADING VALUE FORMAT GATE] '{}' → '{}' ({:?}) | 값 \"{}\" 형식 불일치",
                        unique_labels[h], t_field_names[f], f_fmt, pair_val));
                    crate::utils::score_dynamics::record_field_reject(
                        &t_field_names[f],
                        crate::utils::score_dynamics::GateKind::Format,
                    );
                    continue;
                }
                let _ = f_strict;
                if f_fmt == crate::utils::ai_utils::FieldFormat::Enum
                    && crate::utils::ai_utils::is_pure_numeric_value(pair_val) {
                    emit_term(&format!("    🚫 [TRADING ENUM NUMERIC GATE] '{}' → '{}' | 값 \"{}\" 은 순수 수치",
                        unique_labels[h], t_field_names[f], pair_val));
                    crate::utils::score_dynamics::record_field_reject(
                        &t_field_names[f],
                        crate::utils::score_dynamics::GateKind::Enum,
                    );
                    continue;
                }
                leaf_raw[f][h] = own;
                if unique_section[h].is_empty() { continue; }
                if section_embs[h].iter().all(|&v| v == 0.0) { continue; }
                sec_raw[f][h] = crate::utils::ai_utils::weighted_max_pool_sim(
                    &section_embs[h], &t_label_embs[f], &t_label_weights[f]
                );
            }
        }

        const SECTION_WEIGHT: f32 = 0.5f32;
        let mut t_matrix: Vec<Vec<f32>> = vec![vec![-1.0f32; unique_labels.len()]; t_field_names.len()];
        let mut sec_applied = 0usize;
        for h in 0..unique_labels.len() {
            let qualified = unique_qualified.get(h).copied().unwrap_or(false);
            let mut sec_sum = 0.0f32;
            let mut sec_cnt = 0usize;
            for f in 0..t_field_names.len() {
                if leaf_raw[f][h] < 0.0 { continue; }
                if sec_raw[f][h] < 0.0 { continue; }
                sec_sum += sec_raw[f][h];
                sec_cnt += 1;
            }
            let sec_mean = if sec_cnt > 0 { sec_sum / (sec_cnt as f32) } else { 0.0 };
            if qualified && sec_cnt > 1 { sec_applied += 1; }
            for f in 0..t_field_names.len() {
                if leaf_raw[f][h] < 0.0 { continue; }
                let sec_term = if qualified && sec_cnt > 1 && sec_raw[f][h] >= 0.0 {
                    sec_raw[f][h] - sec_mean
                } else {
                    0.0
                };
                t_matrix[f][h] = leaf_raw[f][h] + SECTION_WEIGHT * sec_term;
            }
        }
        emit_term(&format!(
            "  🧭 [SECTION SCOPE] 섹션 보정 적용 라벨 {}개 / 전체 {}개 (고유 라벨에는 라벨 축만 사용)",
            sec_applied, unique_labels.len()
        ));
        if let Some(anchor) = self_id_anchor_label {
            for f in 0..t_field_names.len() {
                if is_self_id_field(&t_field_names[f]) {
                    // doc_number 열은 앵커 라벨만 남깁니다.
                    for h in 0..unique_labels.len() {
                        if h != anchor { t_matrix[f][h] = -1.0; }
                    }
                    // 앵커 라벨이 게이트에 걸려 -1 이 되어 있었다면 되살립니다.
                    if t_matrix[f][anchor] < 0.0 {
                        t_matrix[f][anchor] = leaf_raw[f][anchor].max(1.0);
                    }
                } else {
                    // 앵커 라벨은 doc_number 이외의 필드로 새지 않습니다.
                    t_matrix[f][anchor] = -1.0;
                }
            }
        }

        // 🌟 [FOREIGN REF PIN] 타 문서 참조 라벨의 열을 전 필드에서 닫습니다.
        for h in 0..unique_labels.len() {
            if !foreign_ref_label[h] { continue; }
            for f in 0..t_field_names.len() {
                t_matrix[f][h] = -1.0;
            }
        }

        // 🌟 [T-2] 잡음 마진 게이트를 적용한 배타 배정으로 교체합니다.
        //
        //  ── 실측 표적 ──
        //   ✨ [PLINKO ASSIGN] 'related_po_number' → 'marks_numbers' | Margin: +0.0001
        //   이 경로의 마진 임계값은 0.0 이라 사실상 게이트가 없습니다.
        //   그 결과 동점에 가까운 배정이 아무 저항 없이 통과했습니다.
        //
        //  ── allow_tie 의 판단 근거 ──
        //   PairConfusion(1·2위만 붙음)이면 혼동 사전을 조회해
        //   과거에 같은 쌍이 반복 충돌했고 승자가 일관되었는지 봅니다.
        //   Phase 0 관측이 부족하면 confusion_winner 가 None 을 돌려주므로
        //   보류가 되고, 그 필드는 아래 LLM 카테고리 패스가 다시 채웁니다.
        //   Flat(전체 평탄)은 다중 속성 값이거나 변별력 없음이므로
        //   역사와 무관하게 항상 보류합니다. 결정론으로 고를 근거가 없기 때문입니다.
        //
        //  ── 보류가 손실이 아닌 이유 ──
        //   PLINKO 는 'LLM 없이 확정' 하는 선행 단계일 뿐이고,
        //   확정되지 않은 필드는 PRESENCE GATE 를 거쳐 LLM 이 채웁니다.
        //   즉 잘못 확정하면 되돌릴 수 없지만, 보류하면 복구 경로가 있습니다.
        let (t_assign, t_diags) = {
            let names = t_field_names.clone();
            crate::utils::ai_utils::exclusive_assign_by_score_adaptive(
                &t_matrix,
                0.0,
                0.0,
                |f: usize, rival: usize, kind: crate::utils::ai_utils::TieKind| -> bool {
                    if kind == crate::utils::ai_utils::TieKind::Flat { return false; }
                    let (a, b) = match (names.get(f), names.get(rival)) {
                        (Some(x), Some(y)) => (x.as_str(), y.as_str()),
                        _ => return false,
                    };
                    match crate::utils::score_dynamics::confusion_winner(a, b) {
                        Some((winner, rate)) => winner == a && rate > 0.5,
                        None => false,
                    }
                },
            )
        };
        // 🌟 [T-2 진단] 동률 판정 결과를 로그와 관측으로 남깁니다.
        for d in t_diags.iter() {
            let fname = t_field_names.get(d.field).cloned().unwrap_or_default();
            let rname = t_field_names.get(d.rival).cloned().unwrap_or_default();
            match d.tie {
                crate::utils::ai_utils::TieKind::Decisive => {}
                crate::utils::ai_utils::TieKind::PairConfusion => {
                    emit_term(&format!(
                        "    ⚖️ [TIE / PAIR CONFUSION] Label '{}' | '{}' vs '{}' | 마진 {:+.4} < 잡음 마진 {:.4} | 1·2위만 붙고 3위 이하는 떨어짐 → {}",
                        unique_labels.get(d.line).cloned().unwrap_or_default(),
                        fname, rname, d.margin, d.noise_margin,
                        if d.accepted { "혼동 사전의 역사 승자로 확정" } else { "보류 (LLM 패스로 이관)" }
                    ));
                }
                crate::utils::ai_utils::TieKind::Flat => {
                    emit_term(&format!(
                        "    ⚖️ [TIE / FLAT] Label '{}' | '{}' vs '{}' | 마진 {:+.4} < 잡음 마진 {:.4} | 경쟁 분포 전체가 평탄 → 보류 (다중 속성 값이거나 변별력 없음)",
                        unique_labels.get(d.line).cloned().unwrap_or_default(),
                        fname, rname, d.margin, d.noise_margin
                    ));
                }
            }
            if d.tie != crate::utils::ai_utils::TieKind::Decisive {
                crate::utils::score_dynamics::record_confusion(&fname, &rname, d.margin);
                crate::utils::score_dynamics::record_baseline("plinko.noise_margin", d.noise_margin);
            }
        }
        {
            let held = t_diags.iter().filter(|d| !d.accepted).count();
            if held > 0 {
                emit_term(&format!(
                    "  ⚖️ [T-2 TIE GATE] 잡음 마진을 넘지 못한 배정 {}건을 보류했습니다. (잘못 확정하면 되돌릴 수 없지만, 보류는 LLM 패스가 복구합니다)",
                    held
                ));
            }
        }
        use crate::logic::trade_field_category;
        let mut drift_dropped = 0usize;
        for (f, a) in t_assign.iter().enumerate() {
            let (h, score, margin) = match a { Some(v) => *v, None => continue };
            let fname = t_field_names[f].clone();
            if crate::utils::ai_utils::is_id_link_field(&fname) { continue; }
            let pinned = match self_id_anchor_label {
                Some(a) => a == h && is_self_id_field(&fname),
                None => false,
            };
            {
                let mut best_f = usize::MAX;
                let mut best_v = f32::MIN;
                let mut sum = 0.0f64;
                let mut sq = 0.0f64;
                let mut cnt = 0usize;
                for ff in 0..t_field_names.len() {
                    let v = t_matrix[ff][h];
                    if v < 0.0 { continue; }
                    sum += v as f64;
                    sq += (v as f64) * (v as f64);
                    cnt += 1;
                    if v > best_v { best_v = v; best_f = ff; }
                }
                if !pinned && cnt >= 2 && best_f != f && t_matrix[f][h] >= 0.0 {
                    let mu = sum / cnt as f64;
                    let sd = ((sq / cnt as f64 - mu * mu).max(0.0)).sqrt() as f32;
                    let gap = best_v - t_matrix[f][h];
                    if gap > sd {
                        drift_dropped += 1;
                        emit_term(&format!(
                            "    🚫 [PREEMPTION DRIFT] Label '{}' 의 최적 필드는 '{}'({:+.4}) 인데 선점당해 '{}'({:+.4}) 로 밀렸습니다. 격차 {:+.4} > 분포 표준편차 {:.4} → 배정을 폐기합니다.",
                            unique_labels[h], t_field_names[best_f], best_v, fname, t_matrix[f][h], gap, sd
                        ));
                        continue;
                    }
                }
            }
            let f_multi = crate::utils::ai_utils::is_multi_value_field(&fname);
            let mut val = if f_multi { phrase_multi[h].clone() } else { phrase_single[h].clone() };
            if val.trim().is_empty() { continue; }
            let mut split_pair: Option<(String, String, String)> = None; // (수량필드, 수량, 단위)
            if let Some((cnt_f, unit_f)) = crate::utils::ai_utils::count_unit_pair_of(&fname) {
                if let Some((num, unit)) = crate::utils::ai_utils::split_count_and_unit(&val) {
                    if fname == unit_f && !unit_f.is_empty() {
                        // 단위 축으로 배정되었는데 수량이 붙어 있는 경우
                        emit_term(&format!(
                            "    ✂️ [COUNT-UNIT SPLIT] '{}' 값 \"{}\" 을 분해합니다 → {}={} / {}={}",
                            fname, val, cnt_f, num, unit_f, unit
                        ));
                        split_pair = Some((cnt_f.to_string(), num, unit.clone()));
                        val = unit;
                    } else if fname == cnt_f {
                        // 수량 축으로 배정되었는데 단위가 붙어 있는 경우
                        emit_term(&format!(
                            "    ✂️ [COUNT-UNIT SPLIT] '{}' 값 \"{}\" 을 분해합니다 → {}={} / {}={}",
                            fname, val, cnt_f, num, if unit_f.is_empty() { "(단위축 없음)" } else { unit_f }, unit
                        ));
                        if !unit_f.is_empty() {
                            split_pair = Some((unit_f.to_string(), unit, num.clone()));
                        }
                        val = num;
                    }
                }
            }

            let cat = trade_field_category(&fname);

            // 🌟 [ROLE PARTY DIVERT] 당사자 축인데 라벨이 '역할 당사자' 를 가리키면
            //    parties 대신 other_parties 배열로 흡수합니다.
            //
            //  ── 실측 사고 (SR-2026-0820) ──
            //   recipient_name: "Pacific Ocean Lines" — 포워더가 수하인으로 배정되었습니다.
            //   bias.json 의 편견 강화(Fix J)로 recipient_name 후보에서는 탈락하지만,
            //   탈락한 값이 갈 곳이 없으면 그대로 소실되어 '운송인을 못 뽑는' 상태가 됩니다.
            //   trade_schema.base.other_parties 가 party_role / party_name 축을 이미 갖고 있으므로
            //   그쪽으로 흘려보내면 값을 잃지 않으면서 수하인 축도 오염되지 않습니다.
            //
            //  ── 판정 근거 ──
            //   라벨 임베딩과 '역할 당사자 앵커' / 'sender·recipient 앵커' 를 비교합니다.
            //   어휘를 코드에 적지 않고, bias.json 의 other_parties.party_role 설명문을
            //   그대로 앵커로 씁니다. 역할이 늘어도 이 코드는 수정 대상이 아닙니다.
            let mut cat = cat;
            let mut fname = fname;
            if (fname == "recipient_name" || fname == "sender_name" || fname == "notify_party_name")
                && role_party_anchor_ready
            {
                let li = match unique_labels.iter().position(|u| u == &unique_labels[h]) {
                    Some(v) => v,
                    None => h,
                };
                if !leaf_embs[li].iter().all(|&v| v == 0.0) {
                    let role_sim = crate::utils::ai_utils::max_pool_sim(&leaf_embs[li], &role_party_embs);
                    let main_sim = crate::utils::ai_utils::max_pool_sim(&leaf_embs[li], &main_party_embs);
                    if role_sim > main_sim {
                        emit_term(&format!(
                            "    🔀 [ROLE PARTY DIVERT] Label '{}' 는 역할 당사자({:.4}) 가 주요 당사자({:.4}) 보다 우세합니다. '{}' 대신 other_parties.party_name 으로 흡수합니다. | Value: \"{}\"",
                            unique_labels[h], role_sim, main_sim, fname, val
                        ));
                        // 역할 라벨 자체를 party_role 로 함께 기록해 두면 소비처가 구분할 수 있습니다.
                        let slot = final_data_map
                            .entry("other_parties".to_string())
                            .or_insert_with(|| Value::Array(Vec::new()));
                        if let Some(arr) = slot.as_array_mut() {
                            arr.push(json!({
                                "party_role": unique_labels[h].clone(),
                                "party_name": val.clone()
                            }));
                        }
                        assigned_fields.insert(format!("party_name::{}", unique_labels[h]), val.clone());
                        continue;
                    }
                }
                let _ = (&mut cat, &mut fname);
            }

            if cat.is_empty() {
                emit_term(&format!("    ⚪ [TRADING CATEGORY UNMAPPED] '{}' 는 카테고리에 매핑되지 않아 루트에만 주입합니다.", fname));
            } else if let Some(slot) = final_data_map.get_mut(cat).and_then(|v| v.as_object_mut()) {
                slot.insert(fname.clone(), json!(val.clone()));
            } else {
                let arr_key: Option<&str> = if crate::logic::is_trade_array_category(cat) {
                    Some(if cat == "items" { "line_items" } else { cat })
                } else {
                    None
                };
                if let Some(ak) = arr_key {
                    let slot = final_data_map
                        .entry(ak.to_string())
                        .or_insert_with(|| Value::Array(Vec::new()));
                    if let Some(arr) = slot.as_array_mut() {
                        if arr.is_empty() {
                            arr.push(Value::Object(serde_json::Map::new()));
                        }
                        if let Some(row) = arr[0].as_object_mut() {
                            row.insert(fname.clone(), json!(val.clone()));
                        }
                    }
                } else {
                    emit_term(&format!(
                        "    ⚠️ [PLINKO WRITE MISS] '{}' (cat: '{}') 를 기록할 루트 슬롯을 찾지 못했습니다.",
                        fname, cat
                    ));
                }
            }
            assigned_fields.insert(fname.clone(), val.clone());
            emit_term(&format!("    ✨ [TRADING PLINKO ASSIGN] Label '{}' → Field '{}' (cat: {}) | Score: {:+.4} | Margin: {:+.4} | Line {} | Value: \"{}\"",
                unique_labels[h], fname, if cat.is_empty() { "-" } else { cat }, score, margin, phrase_line[h] + 1, val));

            // 🌟 [SDS 계측] 확정 마진과 그 라벨의 전체 감쇠 곡선을 남깁니다.
            //
            //  ── 표적 사고 ──
            //   'related_po_number' → 'marks_numbers' | Margin: +0.0001
            //   현재는 이 값이 로그로만 흘러가고 아무 데도 축적되지 않습니다.
            //   그런데 이런 동률에는 두 종류가 있습니다.
            //     우연한 근접  — 다음에는 다른 필드가 이길 수 있음
            //     구조적 혼동  — 같은 쌍이 반복해서 붙음
            //   혼동 카운트를 쌓으면 Phase 1 에서 역사 승자로 결정론 해소가 됩니다.
            //
            //  ── 감쇠 곡선을 함께 남기는 이유 ──
            //   [0.85, 0.20, 0.15] 는 낮은 마진이어도 1위가 이상치이고,
            //   [0.60, 0.59, 0.58] 은 높은 마진이어도 변별력이 없습니다.
            //   두 상황을 마진만으로는 영원히 구분할 수 없습니다.
            //
            //  ── 값은 저장하지 않습니다 ──
            //   프라이버시 정책상 필드명과 수치만 남기고 val 은 기록하지 않습니다.
            crate::utils::score_dynamics::record_field_assigned(&fname, margin);
            {
                // 🌟 [정정] 이 스코프의 라벨×필드 행렬은 t_matrix 입니다.
                //    d_matrix 는 scheduler.rs 상세 경로의 변수명이며 여기에는 존재하지 않습니다.
                //    또한 무효 표식이 f32::MIN 이 아니라 -1.0 이므로
                //    필터 조건도 그에 맞춰야 합니다. (FOREIGN REF PIN 이 -1.0 으로 열을 닫습니다)
                let row: Vec<f32> = (0..t_field_names.len())
                    .map(|fi| t_matrix[fi][h])
                    .filter(|v| *v >= 0.0 && v.is_finite())
                    .collect();
                if row.len() >= 2 {
                    crate::utils::score_dynamics::record_decay("plinko.label_row", &row);
                }
                // 2위 필드를 찾아 혼동 쌍으로 남깁니다.
                let mut best2: (usize, f32) = (usize::MAX, f32::MIN);
                for fi in 0..t_field_names.len() {
                    if t_field_names[fi] == fname { continue; }
                    let v = t_matrix[fi][h];
                    if v < 0.0 || !v.is_finite() { continue; }
                    if v > best2.1 { best2 = (fi, v); }
                }
                if best2.0 != usize::MAX {
                    crate::utils::score_dynamics::record_confusion(
                        &fname,
                        &t_field_names[best2.0],
                        margin,
                    );
                    // 마진이 극소하면 니어미스로도 별도 계상합니다.
                    // '얼마나 작아야 니어미스인가' 는 Phase 1 에서 분포로 유도하므로
                    // 여기서는 판정하지 않고 마진 분포에 그대로 남깁니다.
                    crate::utils::score_dynamics::record_near_miss(&t_field_names[best2.0]);
                }
            }
        }
        emit_term(&format!(
            "  ✅ [TRADING PLINKO] LLM 없이 {}개 필드 확정 완료. (표류 폐기 {}건)",
            assigned_fields.len(), drift_dropped
        ));
    }

    let mut absent_fields: std::collections::HashSet<String> = std::collections::HashSet::new();
    {
        let mut evidence: Vec<String> = Vec::new();
        for l in unique_leaf.iter() {
            let t = l.trim();
            if t.chars().count() < 2 { continue; }
            if !evidence.iter().any(|e| e == t) { evidence.push(t.to_string()); }
        }
        {
            let mut consumed: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for p in detail_pairs.iter() {
                consumed.insert(p.primary_line);
                consumed.insert(p.label_line);
            }
            for (i, line) in pug_lines.iter().enumerate() {
                if consumed.contains(&i) { continue; }
                let (_, _, _, txt) = crate::utils::ai_utils::pug_line_parts(line);
                let t = txt.trim();
                if t.chars().count() < 3 { continue; }
                if evidence.iter().any(|e| e == t) { continue; }
                evidence.push(t.to_string());
                if evidence.len() >= 200 { break; }
            }
        }
        if evidence.is_empty() || t_field_names.is_empty() {
            emit_term("  ⚪ [PRESENCE GATE] 증거 라인 또는 라벨 뱅크가 없어 전 필드를 존재로 간주합니다. (fail-open)");
        } else {
            let ev_embs = model.get_embedding_batch(evidence.clone()).await
                .unwrap_or_else(|_| vec![vec![0.0; 384]; evidence.len()]);

            let mut bias_bank: Vec<(String, String, Vec<f32>)> = Vec::new();
            for f in 0..t_field_names.len() {
                let c = crate::logic::trade_field_category(&t_field_names[f]);
                let c = if c.is_empty() { "-".to_string() } else { c.to_string() };
                for e in t_label_embs[f].iter() {
                    if e.iter().all(|&v| v == 0.0) { continue; }
                    bias_bank.push((c.clone(), t_field_names[f].clone(), e.clone()));
                }
            }
            let no_prej: Vec<(String, String, Vec<f32>)> = Vec::new();
            let (keys, net, _) = crate::utils::ai_utils::bank_neutral_key_matrix(
                &ev_embs, &bias_bank, &no_prej,
            );
            let key_cat: Vec<String> = keys.iter().map(|k| {
                let c = crate::logic::trade_field_category(k);
                if c.is_empty() { "-".to_string() } else { c.to_string() }
            }).collect();
            let mut present: std::collections::HashSet<String> = std::collections::HashSet::new();
            let mut cat_hits: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
            for ei in 0..evidence.len() {
                // ① 카테고리 argmax
                let mut best_cat = String::new();
                let mut best_score = f32::MIN;
                for ki in 0..keys.len() {
                    let v = net[ki][ei];
                    if v == f32::MIN { continue; }
                    if v > best_score { best_score = v; best_cat = key_cat[ki].clone(); }
                }
                if best_cat.is_empty() || best_score <= 0.0 { continue; }
                *cat_hits.entry(best_cat.clone()).or_insert(0) += 1;
                // ② 승리 카테고리 내부 평균 이상 필드만 존재로 인정
                let mut sum = 0.0f32;
                let mut cnt = 0usize;
                for ki in 0..keys.len() {
                    if key_cat[ki] != best_cat { continue; }
                    let v = net[ki][ei];
                    if v == f32::MIN { continue; }
                    sum += v;
                    cnt += 1;
                }
                if cnt == 0 { continue; }
                let mean = sum / cnt as f32;
                for ki in 0..keys.len() {
                    if key_cat[ki] != best_cat { continue; }
                    let v = net[ki][ei];
                    if v == f32::MIN { continue; }
                    if v >= mean { present.insert(keys[ki].clone()); }
                }
            }

            for k in assigned_fields.keys() { present.insert(k.clone()); }
            present.insert("doc_number".to_string());
            present.insert("doc_type".to_string());
            for f in t_field_names.iter() {
                if !present.contains(f) { absent_fields.insert(f.clone()); }
            }
            // 🌟 [SDS / T-1 → 기획 T-4 입력] 카테고리별 실현 최댓값과 실효 필드 수
            //
            //  ── 무엇이 문제였나 (실측) ──
            //   이 게이트는 (필드 × 증거) 행렬 net 을 이미 만들어 두고
            //   카테고리 argmax 분포를 세는데, 그 값이 전부 emit_term 문자열로만
            //   소비되고 통계로 가지 않았습니다.
            //   그래서 trading 스코프의 category 가 {} 로 비어 있고,
            //   기획 T-4 의 N_eff 캘리브레이션은 입력이 없어 착수 자체가 불가능했습니다.
            //   같은 관측을 비전 경로(vision_encoder.rs)는 이미 남기고 있습니다.
            //
            //  ── 왜 '실현 최댓값' 인가 ──
            //   CATEGORY-NEUTRAL 은 gumbel_expected_z(N) 으로 카테고리 크기 편향을
            //   보정하는데, 이때 N 을 '스키마상 필드 수' 로 가정합니다.
            //   그런데 같은 카테고리의 필드들은 동의어로 상관되어 있어
            //   실효 드로잉 수 N_eff < N 입니다.
            //   실현 최댓값의 분포를 쌓아 두면 그 분포에서 N_eff 를 역산할 수 있고,
            //   구조가 아니라 실측이 보정치를 정하게 됩니다.
            //
            //  ── n_fields 를 keys 기준으로 세는 이유 ──
            //   t_field_names 에는 라벨 뱅크가 비어 임베딩이 만들어지지 않은 필드도 있습니다.
            //   그런 필드는 이 행렬에 열로 존재하지 않으므로 드로잉에 참여하지 않았습니다.
            //   실제로 경쟁한 키만 세야 N 과 실현 최댓값의 짝이 맞습니다.
            //
            //  ── Phase 0 원칙 ──
            //   present / absent_fields 판정에는 전혀 손대지 않습니다. 관측만 추가합니다.
            {
                let mut sds_cat_fields: std::collections::HashMap<String, std::collections::HashSet<String>> =
                    std::collections::HashMap::new();
                let mut sds_cat_max: std::collections::HashMap<String, f32> =
                    std::collections::HashMap::new();
                for ki in 0..keys.len() {
                    let c = key_cat[ki].clone();
                    if c == "-" { continue; }
                    sds_cat_fields.entry(c.clone()).or_default().insert(keys[ki].clone());
                    let mut m = f32::MIN;
                    for ei in 0..evidence.len() {
                        let v = net[ki][ei];
                        if v == f32::MIN { continue; }
                        if v > m { m = v; }
                    }
                    if m == f32::MIN { continue; }
                    let e = sds_cat_max.entry(c).or_insert(f32::MIN);
                    if m > *e { *e = m; }
                }
                for (c, m) in sds_cat_max.iter() {
                    let n = sds_cat_fields.get(c).map(|s| s.len()).unwrap_or(0);
                    if n == 0 { continue; }
                    crate::utils::score_dynamics::record_category_max(c, n, *m);
                }
                // 게이트 자체의 규모도 남깁니다.
                // 증거가 몇 개일 때 부재 판정이 얼마나 나오는지가
                // '이 게이트가 과도하게 필드를 잘라내는가' 의 유일한 근거입니다.
                crate::utils::score_dynamics::record_baseline(
                    "trading.presence_evidence",
                    evidence.len() as f32,
                );
                if !t_field_names.is_empty() {
                    crate::utils::score_dynamics::record_baseline(
                        "trading.presence_absent_ratio",
                        absent_fields.len() as f32 / t_field_names.len() as f32,
                    );
                }
            }
            let mut hit_summary: Vec<String> = cat_hits.iter()
                .map(|(c, n)| format!("{}({})", c, n)).collect();
            hit_summary.sort();
            emit_term(&format!(
                "  🧭 [PRESENCE GATE] 증거 {}개 | 카테고리 argmax 분포: {} | 존재 {}필드 / 부재 {}필드",
                evidence.len(),
                if hit_summary.is_empty() { "-".to_string() } else { hit_summary.join(" ") },
                t_field_names.len().saturating_sub(absent_fields.len()),
                absent_fields.len()
            ));
            {
                let mut by_cat: std::collections::HashMap<String, Vec<String>> =
                    std::collections::HashMap::new();
                for f in absent_fields.iter() {
                    let c = crate::logic::trade_field_category(f);
                    let c = if c.is_empty() { "-".to_string() } else { c.to_string() };
                    by_cat.entry(c).or_default().push(f.clone());
                }
                let mut cats: Vec<&String> = by_cat.keys().collect();
                cats.sort();
                for c in cats {
                    let mut fs = by_cat.get(c).cloned().unwrap_or_default();
                    fs.sort();
                    emit_term(&format!(
                        "    ⚪ [NOT PRESENT / {}] {}개: {:?}",
                        c, fs.len(), fs.iter().take(12).collect::<Vec<_>>()
                    ));
                }
            }
        }
    }

    model.secure_vram_relay(crate::model::ModelSize::Qwen3_5, None, Some(cancellation_token.clone()), false, None).await?;
    for cat in &categories {
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }
        let cat_schema_fields: Vec<String> = trade_fields.iter()
            .map(|(f, _, _, _)| f.clone())
            .filter(|f| crate::logic::trade_field_category(f) == *cat)
            .collect();
        let absent_in_cat: Vec<String> = cat_schema_fields.iter()
            .filter(|f| absent_fields.contains(*f))
            .cloned()
            .collect();
        // 🌟 [PRESENCE-AWARE SCHEMA] 부재 필드를 뺀 스키마로 프롬프트를 만듭니다.
        //    (PRESENCE GATE 결과를 사후 필터가 아니라 프롬프트 생성에 직접 연결)
        let absent_set: std::collections::HashSet<String> = absent_in_cat.iter().cloned().collect();
        let schema_prompt = crate::parsing::get_trade_category_schema_present(cat, &doc_type, &absent_set);
        if schema_prompt.contains("SCHEMA:\n{}") || schema_prompt.contains("SCHEMA:\n[ {} ]") {
            emit_term(&format!("[TRADING STEP B] Category '{}' has no fields for {}. Skipping.", cat.to_uppercase(), doc_type));
            continue;
        }
        if !cat_schema_fields.is_empty() && absent_in_cat.len() == cat_schema_fields.len() {
            emit_term(&format!(
                "  ⚪ [PRESENCE SKIP] Category '{}' 의 필드 {}개가 이 문서에 하나도 인쇄되어 있지 않습니다. LLM 호출을 생략합니다. (빈 슬롯 창작 차단)",
                cat.to_uppercase(), cat_schema_fields.len()
            ));
            continue;
        }
        if !crate::logic::is_trade_array_category(cat) {
            let filled = final_data_map.get(*cat)
                .and_then(|v| v.as_object())
                .map(|o| o.iter().filter(|(k, _)| *k != "doc_type").count())
                .unwrap_or(0);
            
            let schema_field_count = schema_prompt.lines()
                .filter(|l| l.trim_start().starts_with('"'))
                .count();
            if schema_field_count > 0 && filled >= schema_field_count {
                emit_term(&format!("  ⚡ [TRADING LLM SKIP] Category '{}' 는 PLINKO 가 {}/{} 필드를 전부 확정하여 LLM 호출을 생략합니다.",
                    cat.to_uppercase(), filled, schema_field_count));
                continue;
            }

            let present_cnt = cat_schema_fields.len().saturating_sub(absent_in_cat.len());
            if present_cnt > 0 && filled >= present_cnt {
                emit_term(&format!(
                    "  ⚡ [PRESENCE LLM SKIP] Category '{}' 는 존재 필드 {}개를 PLINKO 가 전부 확정했습니다. (부재 {}개 제외) LLM 호출을 생략합니다.",
                    cat.to_uppercase(), present_cnt, absent_in_cat.len()
                ));
                continue;
            }
        }

        
        let claimed_ctx = if assigned_fields.is_empty() {
            String::new()
        } else {
            let list: Vec<serde_json::Value> = assigned_fields.iter()
                .map(|(k, v)| json!({ "target_column": k, "extracted_value": v }))
                .collect();
            format!("\n\n[ALREADY CLAIMED VALUES]\nThese values are already assigned to OTHER fields by the deterministic engine. You MUST NOT return any of them:\n{}",
                serde_json::to_string_pretty(&list).unwrap_or_default())
        };

        // 🌟 [PRESENCE FILTER v2] 부재 필드는 스키마 자체에서 빠졌으므로
        //    "null 로 답하라" 는 지시문이 더는 필요 없습니다. 로그만 남깁니다.
        if !absent_in_cat.is_empty() {
            emit_term(&format!(
                "  🚧 [PRESENCE FILTER] Category '{}' | 존재 {}필드 / 부재 {}필드 (스키마에서 제거) {:?}",
                cat.to_uppercase(),
                cat_schema_fields.len().saturating_sub(absent_in_cat.len()),
                absent_in_cat.len(),
                absent_in_cat.iter().take(10).collect::<Vec<_>>()
            ));
        }
        let absent_ctx = String::new();

        emit_term(&format!("[TRADING STEP B] Extracting category '{}' for {}...", cat.to_uppercase(), doc_type));
        log_task_progress(app_handle, &task.id, &json!({
            "category": format!("Extraction ({})", cat.to_uppercase()),
            "summary": format!("Extracting {} fields...", cat),
            "spinner": "⠋"
        }));

        if let Some(gen) = model.qwen3_5_generator.lock().await.as_mut() {
            
            
            
            let params = crate::openai_types::ChatCompletionParameters {
                messages: vec![
                    crate::openai_types::ChatCompletionRequestMessage::System(
                        crate::openai_types::ChatCompletionRequestSystemMessage {
                            content: format!("[PUG CONTENT — attribute-stripped]\n{}{}{}", content_pug, claimed_ctx, absent_ctx),
                            name: None,
                        },
                    ),
                    crate::openai_types::ChatCompletionRequestMessage::User(
                        crate::openai_types::ChatCompletionRequestUserMessage {
                            content: crate::openai_types::ChatCompletionRequestUserMessageContent::Text(
                                schema_prompt,
                            ),
                            name: None,
                        },
                    ),
                ],
                model: "qwen3.5".to_string(),
                max_tokens: Some(1024),
                temperature: Some(0.0),
                top_p: Some(0.95),
                ..Default::default()
            };
            let res = gen.generate(
                params,
                Some(cancellation_token.clone()),
                
                Some(format!("{}_{}_{}", task.id, page_label, cat)),
                None, None, None
            ).await?;
            let mut tile_json = crate::parsing::parse_json_from_llm(&res);
            
            if let Some(obj) = tile_json.as_object_mut() {
                let ks: Vec<String> = obj.keys().cloned().collect();
                let mut protected = 0usize;
                for k in ks {
                    if assigned_fields.contains_key(&k) {
                        obj.remove(&k);
                        emit_term(&format!("    🛡️ [PLINKO PROTECT] '{}' 는 결정론 확정값을 유지하고 LLM 결과를 폐기합니다.", k));
                        crate::utils::score_dynamics::record_field_seen(&k);
                        crate::utils::score_dynamics::record_near_miss(&k);
                        protected += 1;
                    }
                }
                if protected > 0 {
                    crate::utils::score_dynamics::record_baseline(
                        "trading.llm_override_blocked",
                        protected as f32,
                    );
                }
            }
            if !absent_fields.is_empty() {
                let mut dropped: Vec<String> = Vec::new();
                prune_absent_keys(&mut tile_json, &absent_fields, &mut dropped);
                if !dropped.is_empty() {
                    emit_term(&format!(
                        "    🚫 [PRESENCE DROP] Category '{}' | LLM 이 부재 필드 {}개를 채웠으나 폐기합니다: {:?}",
                        cat.to_uppercase(), dropped.len(), dropped.iter().take(12).collect::<Vec<_>>()
                    ));
                }
            }
            crate::model::merge_json_manual(&mut final_data_map, cat, tile_json);
        }
    }

    
    emit_term(&format!(
        "[TRADING PAGE {}/{}] ✅ 페이지 추출 완료 (doc_type='{}', lang='{}')",
        page_idx + 1, total_pages, doc_type, doc_lang
    ));
    page_results.push((doc_type.clone(), doc_lang.clone(), final_data_map));
    }

    model.deep_purge_resources().await;
    crate::utils::resources::wait_for_resources_settled(1200, 800, Some(cancellation_token), model.device_config.gpu_id as u32).await?;

    if page_results.is_empty() {
        return Err(anyhow::anyhow!("Trading extraction produced no result from any page."));
    }

    let mut merged_order: Vec<String> = Vec::new();
    let mut merged_docs: std::collections::HashMap<String, (String, serde_json::Map<String, Value>, usize)> =
        std::collections::HashMap::new();

    for (dt, dl, map) in page_results.into_iter() {
        if !merged_order.iter().any(|x| x == &dt) { merged_order.push(dt.clone()); }
        let slot = merged_docs
            .entry(dt.clone())
            .or_insert_with(|| (dl.clone(), serde_json::Map::new(), 0usize));
        merge_trading_page_map(&mut slot.1, &map);
        slot.2 += 1;
    }

    emit_term(&format!(
        "[TRADING MERGE] 페이지 {}장 → 문서 {}건으로 병합: {:?}",
        total_pages,
        merged_order.len(),
        merged_order.iter()
            .map(|d| format!("{}({}p)", d, merged_docs.get(d).map(|s| s.2).unwrap_or(0)))
            .collect::<Vec<_>>()
    ));

    
    for doc_type in merged_order.into_iter() {
    let (doc_lang, merged_map, merged_page_count) = match merged_docs.remove(&doc_type) {
        Some(v) => v,
        None => continue,
    };

    if doc_type.eq_ignore_ascii_case("TRACKING") {
        emit_term(
            "  📦 [PARCEL EXIT] doc_type='TRACKING' 은 무역 서식이 아니라 택배 라벨입니다. \
             무역 스키마가 없어 청크가 생성되지 않으므로 trading 저장을 건너뜁니다. \
             (commerce 트랙에서 재추출하십시오)"
        );
        continue;
    }
    emit_term(&format!(
        "\n[TRADING DOC] ▶ doc_type='{}' (페이지 {}장 병합) 저장 파이프라인 시작",
        doc_type, merged_page_count
    ));

    let mut extracted_data = Value::Object(merged_map);    
    {
        // 🌟 other_parties / settlement 추가. 비전 경로는 이미 party_name 을 루트에 올리고
        //    있어 두 경로의 루트 축이 어긋나 있었습니다.
        const TRADE_GROUPS_FLAT: [&str; 8] = [
            "header", "parties", "other_parties", "logistics",
            "financials", "conditions", "settlement", "cargo",
        ];

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

        let source = extracted_data.clone();
        let mut hoisted: Vec<String> = Vec::new();

        for group in TRADE_GROUPS_FLAT.iter() {
            let src = match source.get(*group).and_then(|v| v.as_object()) {
                Some(o) => o.clone(),
                None => continue,
            };
            let obj = extracted_data.as_object_mut().unwrap();
            for (k, v) in src {
                if v.is_null() { continue; }
                if let Some(s) = v.as_str() {
                    if s.trim().is_empty() || s == "N/A" { continue; }
                }
                let name = canonical_name(&k);
                if obj.get(&name).map_or(false, |x| !x.is_null()) { continue; }
                obj.insert(name.clone(), v.clone());
                hoisted.push(name);
            }
        }

        
        for (arr_key, promote) in [
            ("containers", vec!["container_number", "seal_number"]),
            ("line_items", vec!["hs_code"]),
        ] {
            let arr = match source.get(arr_key).and_then(|v| v.as_array()) {
                Some(a) => a.clone(),
                None => continue,
            };
            let obj = extracted_data.as_object_mut().unwrap();
            for field in promote {
                if obj.get(field).map_or(false, |x| !x.is_null()) { continue; }
                if let Some(v) = arr.iter().find_map(|it| it.get(field)) {
                    obj.insert(field.to_string(), v.clone());
                    hoisted.push(field.to_string());
                }
            }
        }

        emit_term(&format!(
            "[TRADING STEP C] 🌟 [TRADING FLATTEN v3] data 루트로 승격한 축 {}개: {:?}",
            hoisted.len(),
            hoisted.iter().take(12).collect::<Vec<_>>()
        ));
        
        {
            let arr_hoisted = hoist_array_identifiers(&mut extracted_data);
            if arr_hoisted > 0 {
                emit_term(&format!(
                    "[TRADING STEP C] 🧬 [ARRAY FLATTEN] 배열 카테고리에서 식별자 축 {}개를 루트로 승격했습니다.",
                    arr_hoisted
                ));
            }
        }

        {
            let legacy = extracted_data.get("reference_number")
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty() && s != "N/A");

            if let Some(val) = legacy {
                
                let prefix: String = val
                    .chars()
                    .take_while(|c| c.is_ascii_alphabetic() || *c == '_')
                    .collect::<String>()
                    .to_uppercase();

                if let Some(field) = crate::logic::trade_reference_field_of(&prefix) {
                    let already = extracted_data.get(field)
                        .and_then(|v| v.as_str())
                        .map_or(false, |s| !s.trim().is_empty() && s != "N/A");
                    if !already {
                        extracted_data.as_object_mut().unwrap()
                            .insert(field.to_string(), json!(val.clone()));
                        emit_term(&format!(
                            "  🔀 [REFERENCE PROMOTION] reference_number='{}' 를 접두어 '{}' 기준으로 '{}' 축으로 승격했습니다.",
                            val, prefix, field
                        ));
                    }
                } else if !prefix.is_empty() {
                    emit_term(&format!(
                        "  ⚪ [REFERENCE PROMOTION SKIP] reference_number='{}' 의 접두어 '{}' 는 알려진 서식 코드가 아니라 승격하지 않습니다.",
                        val, prefix
                    ));
                }
            }
        }

        {
            if let Some(self_field) = crate::logic::trade_reference_field_of(&doc_type) {
                let self_ref = extracted_data.get(self_field)
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default();
                let own = extracted_data.get("doc_number")
                    .and_then(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default();
                if !self_ref.is_empty() && !own.is_empty()
                    && normalize_entity_key(&self_ref) == normalize_entity_key(&own)
                {
                    extracted_data.as_object_mut().unwrap().remove(self_field);
                    emit_term(&format!(
                        "  🧹 [SELF-REFERENCE DROP] '{}' 가 자기 자신({})을 가리키고 있어 릴레이 축에서 제거했습니다.",
                        self_field, own
                    ));
                }
            }
        }

        normalize_trading_data(&mut extracted_data, &doc_lang);
        emit_term("[TRADING STEP C] 🔢 [NORMALIZE] 수치/날짜/통화 축 정규화 완료 (팀 통계 집계 가능 상태)");

        let natural_text = parsing::json_to_natural_language(&extracted_data);
        let masked_text = natural_text.clone();
        if let Some(obj) = extracted_data.as_object_mut() {
            obj.insert("text".to_string(), json!(natural_text));
            obj.insert("masked_text".to_string(), json!(masked_text));
            obj.insert("mode".to_string(), json!("shipping"));
            obj.insert("type".to_string(), json!(doc_type.clone()));
        }
    }

    let store = {
        let store_guard = store_mutex.lock().await;
        store_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Store not initialized"))?.clone()
    };
    
    let doc_number = extracted_data.get("doc_number")
        .or_else(|| extracted_data.get("document_number"))
        .and_then(|s| s.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s.as_str() != "N/A")
        .or_else(|| {
            extracted_data.get("header")
                .and_then(|h| h.get("doc_number").or_else(|| h.get("document_number")))
                .and_then(|s| s.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty() && s.as_str() != "N/A")
        })
        .unwrap_or_else(|| {
            let seed = extracted_data.get("text")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| serde_json::to_string(&extracted_data).unwrap_or_default());
            let fallback = format!("AUTO-{}-{}", doc_type, crate::utils::hash::digest(&seed));
            emit_term(&format!(
                "  ⚠️ [DOC NUMBER FALLBACK] '{}' 문서에서 문서번호를 찾지 못했습니다. 내용 기반 결정론 ID '{}' 를 사용합니다. (task_id 를 쓰면 재스캔마다 중복 문서가 생깁니다)",
                doc_type, fallback
            ));
            fallback
        });

    
    let clean_no = normalize_entity_key(&doc_number);
    emit_term(&format!("[TRADING] 🔑 문서 식별자 확정: '{}' → 정규화 '{}'", doc_number, clean_no));
    let index_val = entity_index(&doc_type, &team_id, &doc_number);
    let hashed_item_id = entity_id(&team_id, index_val);

    if let Some(obj) = extracted_data.as_object_mut() {
        obj.insert("id".to_string(), json!(hashed_item_id.clone()));
        obj.insert("index".to_string(), json!(index_val));
        obj.insert("doc_type".to_string(), json!(doc_type.clone()));
        obj.insert("doc_number".to_string(), json!(doc_number.clone()));
        obj.insert("no".to_string(), json!(doc_number.clone()));
        obj.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
    }

    let text_to_embed = extracted_data.get("text").and_then(|v| v.as_str()).map(|s| s.to_string()).unwrap_or_default();
    let item_digest = crate::utils::hash::digest(&text_to_embed);
    let item_vector = model.get_embedding(text_to_embed.clone()).await.unwrap_or(vec![0.0; 384]);

    let (cc_val, bcc, ref_val) = crate::parsing::trading_envelope(
        &team_id,
        &doc_type,
        &extracted_data,
        &hashed_item_id,
    );
    emit_term(&format!(
        "  🧭 [TRADING ENVELOPE] cc='{}' | bcc='{}' | ref='{}' (거래 건 축)",
        cc_val, bcc, ref_val
    ));

    
    save_item(&store, "items", &hashed_item_id, &doc_type, extracted_data.clone(), Some(item_vector.clone()),
        &from_addr, &team_id, &cc_val, &bcc, &ref_val, Some(&item_digest)).await;

    let relay_targets = crate::logic::related_trading(&doc_type);
    let mut relay_linked = 0usize;
    let mut relay_drafted = 0usize;
    let mut relay_skipped: Vec<String> = Vec::new();
    let mut relay_draft_types: Vec<&'static str> = Vec::new();
    let mut relay_promoted_types: Vec<&'static str> = Vec::new();

    for foreign_type in relay_targets {
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        let (mine_field, foreign_field) = match crate::logic::trading_relay_pair(&doc_type, foreign_type) {
            Some(p) => p,
            None => continue,
        };
        
        let ref_raw = extracted_data.get(mine_field)
            .and_then(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && s.as_str() != "N/A");

        let ref_display = match ref_raw {
            Some(r) => r,
            None => {
                relay_skipped.push(format!("{}({})", foreign_type, mine_field));
                continue;
            }
        };

        let clean_ref = normalize_entity_key(&ref_display);
        if clean_ref.is_empty() {
            relay_skipped.push(format!("{}({})", foreign_type, mine_field));
            continue;
        }

        
        if clean_ref == clean_no {
            if !foreign_type.eq_ignore_ascii_case(&doc_type) {
                emit_term(&format!(
                    "  ⚠️ [RELAY SELF-LOOP / SUSPECT] {}.{} 가 자기 문서번호({})와 같습니다. 이 참조가 기대하는 문서 타입은 '{}' 인데 이 문서는 '{}' 입니다. doc_number 오배정으로 '{}' 갈래가 소멸할 수 있습니다.",
                    doc_type, mine_field, ref_display, foreign_type, doc_type, foreign_type
                ));
            } else {
                emit_term(&format!(
                    "  🧹 [RELAY SELF-LOOP] {}.{} 가 자기 문서번호({})와 같아 릴레이를 건너뜁니다.",
                    doc_type, mine_field, ref_display
                ));
            }
            continue;
        }

        
        
        
        let foreign_index = entity_index(foreign_type, &team_id, &ref_display);
        let mine_col = crate::logic::trading_index_column(&doc_type);
        let foreign_col = crate::logic::trading_index_column(foreign_type);

        
        extracted_data.as_object_mut().unwrap()
            .insert(foreign_col.clone(), json!(foreign_index));
        emit_term(&format!(
            "  🔑 [TRADING INDEX] {}.{} = {} (근거 {}='{}' → 정규화 '{}')",
            doc_type, foreign_col, foreign_index, mine_field, ref_display, clean_ref
        ));
        
        let mut hit: Option<(String, Value)> = None;

        {
            let idx_filter = format!(
                "type = '{}' AND data LIKE '%\"index\":{}%'",
                foreign_type,
                foreign_index
            );
                        
            if let Ok(results) = store.get_all_items("items", 1, 0, Some(idx_filter)).await {
                if let Some(doc) = results.into_iter().next() {
                    if let Ok(data) = serde_json::from_str::<Value>(&doc.json_data) {
                        hit = Some((doc.id, data));
                        emit_term(&format!(
                            "  🔍 [RELAY LOOKUP 1st] index={} 로 '{}' 문서 발견: '{}'",
                            foreign_index, foreign_type, &hit.as_ref().unwrap().0
                        ));
                    }
                }
            }
        }

        if hit.is_none() {
            if let Ok(Some((foreign_id, foreign_data))) = store.find_item_by_property("items", foreign_field, &json!(doc_number)).await {
                
                let found_type = foreign_data.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if found_type == foreign_type {
                    hit = Some((foreign_id.clone(), foreign_data));
                    emit_term(&format!(
                        "  🔍 [RELAY LOOKUP 2nd] find_item_by_property('{}', '{}') 로 '{}' 문서 발견: '{}'",
                        foreign_field, doc_number, foreign_type, &foreign_id
                    ));
                } else {
                    emit_term(&format!(
                        "  ⚠️ [RELAY TYPE GUARD] '{}' 필드 매칭 문서의 type='{}' 이 기대 '{}' 와 불일치하여 스킵.",
                        foreign_field, found_type, foreign_type
                    ));
                }
            }
        }
        
        if hit.is_none() {
            if let Ok(Some((foreign_id, foreign_data))) = store.find_item_by_property("items", "doc_number", &json!(ref_display)).await {
                let found_type = foreign_data.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if found_type == foreign_type {
                    hit = Some((foreign_id.clone(), foreign_data));
                    emit_term(&format!(
                        "  🔍 [RELAY LOOKUP 3rd] find_item_by_property('doc_number', '{}') 로 '{}' 문서 발견: '{}'",
                        ref_display, foreign_type, &foreign_id
                    ));
                } else {
                    emit_term(&format!(
                        "  ⚠️ [RELAY TYPE GUARD] doc_number 매칭 문서의 type='{}' 이 기대 '{}' 와 불일치하여 스킵.",
                        found_type, foreign_type
                    ));
                }
            }
        }

        if let Some((fid, fdata)) = hit.clone() {
            let found_type = fdata.get("type")
                .or_else(|| fdata.get("doc_type"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if !found_type.is_empty() && found_type != foreign_type {
                emit_term(&format!(
                    "  🚫 [RELAY TYPE GUARD v2] '{}' 는 type='{}' 인데 '{}' 로 갱신하려 했습니다. 문서 변조를 막기 위해 이 릴레이를 폐기합니다.",
                    fid, found_type, foreign_type
                ));
                hit = None;
            }
        }

        match hit {
            Some((foreign_id, mut foreign_data)) => {
                let was_draft = foreign_data.get("updated_at").and_then(|v| v.as_i64()).unwrap_or(0) == 0;
                emit_term(&format!(
                    "[TRADING RELAY] Found existing {} document '{}' (draft: {}).",
                    foreign_type, foreign_id, was_draft
                ));

                {
                    let o = foreign_data.as_object_mut().unwrap();

                    o.insert(mine_col.clone(), json!(index_val));                    
                    o.insert(foreign_field.to_string(), json!(doc_number.clone()));

                    if was_draft {
                        o.insert("updated_at".to_string(), json!(chrono::Utc::now().timestamp_millis()));
                    }
                    if o.get("mode").is_none() {
                        o.insert("mode".to_string(), json!("shipping"));
                    }
                    if o.get("type").is_none() {
                        o.insert("type".to_string(), json!(foreign_type));
                    }
                    if o.get("doc_type").is_none() {
                        o.insert("doc_type".to_string(), json!(foreign_type));
                    }
                }

                let merged_text = parsing::json_to_natural_language(&foreign_data);
                let merged_vector = model.get_embedding(merged_text.clone()).await.unwrap_or(vec![0.0; 384]);
                foreign_data.as_object_mut().unwrap().insert("text".to_string(), json!(merged_text.clone()));
                foreign_data.as_object_mut().unwrap().insert("masked_text".to_string(), json!(merged_text));

                let foreign_bcc = entity_bcc(foreign_type, &cc_val);
                save_item(&store, "items", &foreign_id, foreign_type, foreign_data, Some(merged_vector),
                    &from_addr, &team_id, &cc_val, &foreign_bcc, &ref_val, None).await;
                relay_linked += 1;
                relay_promoted_types.push(foreign_type);
                emit_term(&format!(
                    "  ✅ [TRADING RELAY] {} '{}' 에 {}='{}' / {}={} 역주입 완료.",
                    foreign_type, foreign_id, foreign_field, doc_number, mine_col, index_val
                ));
            },
            None => {
                let draft_id = entity_id(&team_id, foreign_index);
                let mut draft_data = json!({});
                if let Some(obj) = draft_data.as_object_mut() {
                    obj.insert("id".to_string(), json!(draft_id.clone()));
                    obj.insert("type".to_string(), json!(foreign_type));
                    obj.insert("index".to_string(), json!(foreign_index));
                    obj.insert("doc_number".to_string(), json!(""));
                    obj.insert("no".to_string(), json!(""));
                    obj.insert(foreign_field.to_string(), json!(doc_number.clone()));
                    obj.insert(mine_col.clone(), json!(index_val));
                    
                    obj.insert(foreign_col.clone(), json!(foreign_index));
                    obj.insert("updated_at".to_string(), json!(0));
                    obj.insert("mode".to_string(), json!("shipping"));
                    
                    obj.insert("text".to_string(), json!(format!("{} draft (ref: {} = {})", foreign_type, foreign_field, doc_number)));
                }
                let foreign_bcc = entity_bcc(foreign_type, &cc_val);
                save_item(&store, "items", &draft_id, foreign_type, draft_data, None,
                    &from_addr, &team_id, &cc_val, &foreign_bcc, &ref_val, None).await;
                relay_drafted += 1;
                relay_draft_types.push(foreign_type);
                emit_term(&format!(
                    "  📝 [TRADING RELAY DRAFT v3] {} draft '{}' 생성 ({}='{}', index={}).",
                    foreign_type, draft_id, foreign_field, doc_number, foreign_index
                ));
            }
        }
    }

    if !relay_skipped.is_empty() {
        emit_term(&format!(
            "  ⚪ [RELAY NO EVIDENCE] 참조 값이 없어 건너뛴 관계 {}개: {:?}",
            relay_skipped.len(),
            relay_skipped.iter().take(12).collect::<Vec<_>>()
        ));
    }
    emit_term(&format!(
        "  🔗 [RELAY SUMMARY] doc_type='{}' | 기존 문서 연결 {}건 | draft 생성 {}건",
        doc_type, relay_linked, relay_drafted
    ));

    
    save_item(&store, "items", &hashed_item_id, &doc_type, extracted_data.clone(), Some(item_vector.clone()),
        &from_addr, &team_id, &cc_val, &bcc, &ref_val, Some(&item_digest)).await;

    {
        let chunk_count = index_item_chunks(
            &store,
            &model,
            &hashed_item_id,
            &doc_type,
            &doc_lang,
            &extracted_data,
            true,               
            &cc_val,
            &bcc,
            &ref_val,
            "shipping",
            &url,
            cancellation_token,
            app_handle,
            &task.id,
            false,              
        ).await.unwrap_or(0);

        emit_term(&format!(
            "  🧩 [TRADING CHUNK INDEX] item_id='{}' | 청크 {}건 인덱싱 완료 (doc_type='{}')",
            hashed_item_id, chunk_count, doc_type
        ));
    }

    let mut stats_diff: std::collections::HashMap<String, (i64, i64, i64)> = std::collections::HashMap::new();

    {
        let prev = store.get_item_by_id("items", &hashed_item_id).await.ok().flatten();
        match prev {
            None => {
                let e = stats_diff.entry(doc_type.clone()).or_insert((0, 0, 0));
                e.1 += 1; 
                e.2 += 1; 
                emit_term(&format!("  📊 [STATS] doc_type='{}' 신규 문서로 집계합니다.", doc_type));
            },
            Some(existing) => {
                let was_draft = existing.updated_at_ts == 0;
                if was_draft {
                    let e = stats_diff.entry(doc_type.clone()).or_insert((0, 0, 0));
                    e.0 -= 1; 
                    e.1 += 1; 
                    e.2 += 1;
                    emit_term(&format!("  📊 [STATS] doc_type='{}' draft → 완성 문서로 전환합니다.", doc_type));
                } else {
                    emit_term(&format!("  📊 [STATS] doc_type='{}' 기존 문서 갱신이므로 count 를 증가시키지 않습니다.", doc_type));
                }
            }
        }
    }

    for t in relay_draft_types.iter() {
        let e = stats_diff.entry(t.to_string()).or_insert((0, 0, 0));
        e.0 += 1; 
    }
    for t in relay_promoted_types.iter() {
        let e = stats_diff.entry(t.to_string()).or_insert((0, 0, 0));
        e.0 -= 1; 
        e.1 += 1; 
        e.2 += 1; 
    }
    if !relay_draft_types.is_empty() || !relay_promoted_types.is_empty() {
        emit_term(&format!(
            "  📊 [RELAY STATS] draft 신규 {}건 {:?} | draft → 완성 {}건 {:?}",
            relay_draft_types.len(), relay_draft_types,
            relay_promoted_types.len(), relay_promoted_types
        ));
    }
    
    let now_ms_metrics = chrono::Utc::now().timestamp_millis();
    let metrics_input: Vec<Value> = vec![extracted_data.clone()].into_iter().map(|it| {
        let mut v = it;
        if let Some(o) = v.as_object_mut() {
            if o.get("type").is_none() { o.insert("type".to_string(), json!(doc_type.clone())); }
            if o.get("mode").is_none() { o.insert("mode".to_string(), json!("shipping")); }
            if o.get("updated_at").is_none() { o.insert("updated_at".to_string(), json!(now_ms_metrics)); }
            if o.get("created_at").is_none() { o.insert("created_at".to_string(), json!(now_ms_metrics)); }

            let rel_keys: Vec<String> = o.keys()
                .filter(|k| k.starts_with("rel_"))
                .cloned()
                .collect();
            for k in rel_keys { o.remove(&k); }
        }
        v
    }).collect();

    let _ = crate::utils::metrics::update_team_base_metrics(&store, &team_id, &cc_val, &metrics_input, stats_diff.clone()).await;
    emit_term(&format!(
        "  📊 [TEAM METRICS] doc_type='{}' 통계 반영 완료 | 집계 축: amount, amount_subtotal, amount_tax, freight_amount, insurance_amount, local_charges, package_count, weight_gross, weight_net, volume, created_at",
        doc_type
    ));

    let _ = store.update_message_status(&task.id, crate::logic::parse_status("complete"), Some("Trading Extraction Complete")).await;

    emit_term(&format!(
        "[TRADING DOC] ✅ doc_type='{}' 저장 완료 (페이지 {}장 병합)",
        doc_type, merged_page_count
    ));
    }
    

    let payload_done = json!({
        "task_id": task.id,
        "category": "Done",
        "summary": format!("Trading extraction complete. {} page(s) processed.", total_pages),
        "spinner": "✅",
        "data": null
    });
    let _ = app_handle.emit("extraction-progress", &payload_done);
    log_task_progress(app_handle, &task.id, &payload_done);
    // 🌟 [SDS] 태스크 경계에서 관측을 디스크에 확정하고 스코프를 내립니다.
    //    dirty 가 false 면 파일을 쓰지 않으므로 불필요한 I/O 가 없습니다.
    //    이 호출이 있어야 score_dynamics.json 이 처음 생성됩니다.
    emit_term(&format!("[TRADING] {}", crate::utils::score_dynamics::report()));
    crate::utils::score_dynamics::flush();
    crate::utils::score_dynamics::leave_scope();
    println!("[TRADING] Task {} completed. {} page(s) processed.", task.id, total_pages);
    Ok(())
}

const ARRAY_HOIST_KEYS: &[(&[&str], &[&str])] = &[
    (&["containers"],          &["container_number", "seal_number", "type_size"]),
    (&["items", "line_items"], &["hs_code", "item_code"]),
    (&["charges"],             &["charge_code"]),
];
fn hoist_array_identifiers(data: &mut serde_json::Value) -> usize {
    let obj = match data.as_object_mut() { Some(o) => o, None => return 0 };
    let mut hoisted = 0usize;
    for (cats, keys) in ARRAY_HOIST_KEYS {
        
        let mut rows: Vec<serde_json::Value> = Vec::new();
        for cat in *cats {
            if let Some(arr) = obj.get(*cat).and_then(|v| v.as_array()) {
                rows.extend(arr.iter().cloned());
            }
        }
        if rows.is_empty() { continue; }
        for key in *keys {
            let mut vals: Vec<String> = Vec::new();
            for row in &rows {
                let v = match row.get(*key) { Some(x) => x, None => continue };
                let s = match v {
                    serde_json::Value::String(s) => s.trim().to_string(),
                    serde_json::Value::Number(n) => n.to_string(),
                    _ => continue,
                };
                if s.is_empty() { continue; }
                if vals.iter().any(|e| e == &s) { continue; }
                vals.push(s);
            }
            if vals.is_empty() { continue; }
            
            if vals.len() == 1 {
                obj.insert(key.to_string(), serde_json::json!(vals[0]));
            } else {
                obj.insert(key.to_string(), serde_json::json!(vals));
            }
            hoisted += 1;
        }
    }
    hoisted
}
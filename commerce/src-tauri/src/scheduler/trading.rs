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
// =====================================================================
// 🌟 [TITLE AXIS ANCHOR] 제목 축 판정에 쓰는 두 개의 개념 앵커
// ---------------------------------------------------------------------
//  ── 왜 영어 한 벌인가 (단일 언어 하드코딩이 아닌 이유) ──
//   bias.json 의 search_bridge.abstract_bridge 가 영어 브릿지 구만 두고
//   교차언어 매칭을 다국어 임베딩에 맡기는 것과 같은 계보입니다.
//   언어별 사전을 만들면 언어가 늘 때마다 코드를 고쳐야 하지만,
//   granite-embedding-97m-multilingual-r2 가 이미 그 일을 합니다.
//   ('문서유형' / '書類種別' / 'tipo de documento' 는 전부 아래 앵커와 공명합니다)
//
//  ── 왜 두 축인가 ──
//   Purchase_Order 문서에는 'Letter of Credit (L/C)' 가 payment_terms 값으로
//   인쇄되어 있습니다. 제목 축만 세우면 PO 가 LC 로 뒤집힙니다.
//   '이 라벨이 자기 종류를 선언하는가' vs '다른 문서를 참조하는가' 를
//   상대 코사인으로 먼저 가르면 그 경로가 물리적으로 사라집니다.
//   절대 임계치 없이 두 앵커의 대소 비교만 사용하므로 매직 상수가 없습니다.
// =====================================================================
const TRADE_TITLE_LABEL_ANCHOR: &str = "document type, kind of document, type of form, \
     name of this document, title of this document, document name, form name, \
     document code, form code, classification of this document";
const TRADE_REFERENCE_LABEL_ANCHOR: &str = "referenced document number, related document number, \
     reference number of another document, master document number, associated document, \
     payment terms, terms of payment, drawn under credit, issued under, \
     attached documents, required documents, enclosed documents, remark, note";
/// 🌟 [ITEM ATTRIBUTE ANCHOR] '한 줄짜리 품목의 속성' 개념 축.
///
///  ── 왜 세 번째 축이 필요한가 (실측) ──
///   기존 게이트는 '자기선언 vs 타문서참조' 2지 선택이라 '둘 다 아님'을 표현할 수 없습니다.
///   그래서 2페이지의 라벨이 이렇게 오통과했습니다.
///     🏷️ [TITLE LABEL GATE] 'item code'   | 자기선언 0.6187 vs 타문서참조 0.5412 → 표제 후보 유지
///     🏷️ [TITLE LABEL GATE] 'description' | 자기선언 0.6000 vs 타문서참조 0.5295 → 표제 후보 유지
///   'item code' 는 타 문서 참조가 아니므로 참조 축에 지고, 자연히 표제 축이 이깁니다.
///   품목 속성이라는 제3의 선택지를 주면 그 경로가 물리적으로 사라집니다.
///
///  ── 왜 영어 한 벌인가 ──
///   위 두 앵커와 같은 계보입니다. 판정은 다국어 임베딩 코사인으로만 이뤄지므로
///   '품목코드' / '品目コード' / 'código de artículo' 도 같은 축과 공명합니다.
const TRADE_ITEM_ATTRIBUTE_ANCHOR: &str = "line item attribute, attribute of one product row, \
     item code, stock keeping unit, article number, product description, \
     quantity, unit of measure, unit price, line total, amount of this row, \
     table column header, row number in a list, subtotal, discount, total quantity";
/// 🌟 [ROW MARKER ANCHOR] 표 행 구분자 / 그룹 키 등 '표제가 될 수 없는' 라인.
///    실측에서 "[ Item 2 ]" 와 "metadata:" 가 표제 후보로 올라왔습니다.
const TRADE_ROW_MARKER_ANCHOR: &str = "row separator, table row marker, line item index, \
     item number in a list, section key, group key, metadata key, \
     continued from previous page, page break marker, list bullet";
/// 🌟 [TITLE CANDIDATE] 상단 밴드에서 '표제(전문)일 수 있는 값' 을 수집합니다.
#[derive(Debug, Clone)]
struct TitleCandidate {
    /// 라벨이 있으면 그 라벨. 표제 헤딩(라벨 없음)이면 빈 문자열.
    label: String,
    value: String,
    line: usize,
}
/// 🌟 [TITLE BAND] 상단 밴드에서 표제 후보를 뽑습니다.
///
///  ── 밴드 근거 ──
///   vision_encoder.rs::run_title_gate 가 상단 30% 를 제목 밴드로 보는 것과
///   같은 레이아웃 구조 사실입니다. 서식 전문은 문서 최상단에 인쇄되고,
///   본문의 'as per commercial invoice' 같은 언급은 그 아래에 나옵니다.
///
///  ── 두 경로 ──
///   ① 라벨-값 페어 : 'document_type | Purchase Order' 처럼 라벨이 붙은 값
///   ② 라벨 없는 라인 : 'COMMERCIAL INVOICE' 처럼 단독 헤딩
///   실제 스캔 서식은 ②, YAML/폼 기반 문서는 ① 로 들어옵니다. 둘 다 받습니다.
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
// =====================================================================
// 🌟 [TITLE VALUE RESOLVER] '표제로 인정할 값' 을 한 곳에서 결정합니다.
// ---------------------------------------------------------------------
//  ── 왜 공용 함수인가 ──
//   같은 판정을 세 곳이 필요로 합니다.
//     ① STEP A 제목 축
//     ② PAGE CONTINUITY (연속 페이지 판정)
//     ③ MODE REROUTE 프로브 (커머스 진입 시 무역 문서 감지)
//   각자 인라인으로 두면 세 판정이 서로 어긋납니다. 실제로 ②가 ①과 달라
//   "[ Item 2 ]" 를 표제로 인정해 2페이지를 독립 문서로 만들었습니다.
//
//  ── 4축 상대 비교 (절대 임계치 없음) ──
//   라벨이 있으면 : 자기선언 / 타문서참조 / 품목속성 중 자기선언이 최대일 때만 통과
//   라벨이 없으면 : 값 자체를 자기선언 / 행구분자 / 품목속성 과 비교, 자기선언 최대일 때만 통과
//   전부 대소 비교이므로 매직 상수가 없고 언어 리터럴도 없습니다.
// =====================================================================
pub(crate) async fn resolve_title_values(
    model: &LogisModel,
    light_pug: &str,
    band_ratio: f32,
    emit_term: &(dyn Fn(&str) + Send + Sync),
    verbose: bool,
) -> Vec<String> {
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
            TRADE_TITLE_LABEL_ANCHOR.to_string(),
            TRADE_REFERENCE_LABEL_ANCHOR.to_string(),
            TRADE_ITEM_ATTRIBUTE_ANCHOR.to_string(),
            TRADE_ROW_MARKER_ANCHOR.to_string(),
        ])
        .await
        .unwrap_or_else(|_| vec![vec![0.0; 384]; 4]);
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
        let keep = ts > rs && ts > is;
        label_is_title.insert(lt.clone(), keep);
        if verbose {
            emit_term(&format!(
                "     🏷️ [TITLE LABEL GATE] '{}' | 자기선언 {:.4} vs 타문서참조 {:.4} vs 품목속성 {:.4} → {}",
                lt, ts, rs, is,
                if keep { "표제 후보 유지" } else if is >= ts { "품목 속성 → 제외" } else { "참조 라벨 → 제외" }
            ));
        }
    }
    // ── 값 축 ──
    let mut headless: Vec<String> = Vec::new();
    let mut kept: Vec<String> = Vec::new();
    for c in cands.iter() {
        if c.label.trim().is_empty() {
            if !headless.iter().any(|e| e == &c.value) { headless.push(c.value.clone()); }
            continue;
        }
        if label_is_title.get(&humanize(&c.label)).copied().unwrap_or(false) {
            if !kept.iter().any(|e| e == &c.value) { kept.push(c.value.clone()); }
        }
    }
    if !headless.is_empty() {
        let he = model.get_embedding_batch(headless.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; headless.len()]);
        for (i, v) in headless.iter().enumerate() {
            if he[i].iter().all(|&x| x == 0.0) { continue; }
            let ts = crate::utils::ai_utils::cosine_similarity(&he[i], &anchors[0]);
            let rw = crate::utils::ai_utils::cosine_similarity(&he[i], &anchors[3]);
            let is = crate::utils::ai_utils::cosine_similarity(&he[i], &anchors[2]);
            if ts > rw && ts > is {
                if !kept.iter().any(|e| e == v) { kept.push(v.clone()); }
            } else if verbose {
                emit_term(&format!(
                    "     🧹 [HEADLESS DROP] '{}' | 자기선언 {:.4} vs 행구분자 {:.4} vs 품목속성 {:.4} → 표제 아님",
                    v, ts, rw, is
                ));
            }
        }
    }
    if verbose {
        emit_term(&format!(
            "  🪪 [TITLE AXIS] 표제 후보 값 {}개: {:?}",
            kept.len(), kept.iter().take(8).collect::<Vec<_>>()
        ));
    }
    kept
}
// =====================================================================
// 🌟 [PAGE CONTINUITY] 이 페이지가 '새 문서' 인가 '앞 문서의 연속' 인가
// ---------------------------------------------------------------------
//  ── 실측 사고 ──
//   Purchase_Order_Part4.pdf 는 단일 PO 문서인데 2페이지로 분할되었고,
//   2페이지(품목 2행 꼬리 + summary)가 독립 문서로 재분류되어
//     👑 [TRADE CODE COSINE] 'ID' | Top: +5.6726
//   가 되었습니다. 그 결과 존재하지 않는 수입신고(ID) 문서가 저장되고
//   doc_number='DECL-2026-0815' / due_date=2024-12-31 / payment_status=PAID
//   가 전부 창작되었습니다. PO 는 품목 1행과 합계 62500 만 남았습니다.
//
//  ── 판정 근거 (어휘 사전 아님, 다국어 코사인) ──
//   서식은 정의상 '자기 종류를 선언하는 표제' 를 1회 인쇄합니다.
//   표제가 없는 페이지는 그 자체로 독립 문서가 될 수 없습니다.
//   ① 표제 축 : 이 페이지의 표제 후보가 TRADE_DOC_TITLES 와 공명하는가
//   ② 정체 축 : 이 페이지에 '자기 문서번호' 라벨이 있는가
//   두 축이 모두 없으면 앞 페이지의 연속으로 봅니다.
//   판정은 전부 임베딩 코사인이며 언어 리터럴이 없습니다.
// =====================================================================
struct PageIdentityVerdict {
    /// 표제 축 최고 점수 (없으면 f32::MIN)
    title_top: f32,
    /// 자기 문서번호 라벨 최고 점수 (없으면 f32::MIN)
    self_id_top: f32,
    /// 독립 문서로 인정할 수 있는가
    is_standalone: bool,
}
/// 🌟 [SELF IDENTITY ANCHOR] '이 문서 자신의 번호' 를 뜻하는 라벨 개념.
///    TRADE_TITLE_LABEL_ANCHOR 와 같은 계보(영어 한 벌 + 다국어 임베딩)입니다.
const TRADE_SELF_ID_LABEL_ANCHOR: &str = "document number of this document, \
     number printed under the title of this form, own reference number of this form, \
     invoice number, order number, certificate number, declaration number, \
     bill of lading number, waybill number, policy number, claim number, \
     statement number, receipt number, licence number, booking number";
// =====================================================================
// 🌟 [MODE REROUTE PROBE] 커머스로 들어온 문서가 실은 무역 서식인지 판정합니다.
// ---------------------------------------------------------------------
//  ── 왜 필요한가 (실측 사고) ──
//   process_task 의 분기는 `search_mode == "shipping"` 하나뿐입니다.
//   사용자가 커머스 탭에서 Purchase Order PDF 를 올리면 mode='commerce' 이므로
//   커머스 파이프라인으로 들어가고, 거기에는 무역 서식이라는 선택지가 없어
//     [Scheduler] Deterministic Classified Page Type: order
//   로 강제 분류됩니다. 무역 스키마·릴레이·팀 통계가 전부 소실됩니다.
//
//  ── 왜 본문 축으로는 못 가르는가 ──
//   커머스 'order' 앵커에 "purchase order management" 가 들어 있습니다.
//   즉 본문 코사인은 원리적으로 두 개념을 분리할 수 없습니다.
//   가를 수 있는 축은 '자기 종류를 선언하는 표제' 뿐이며,
//   커머스 목록 페이지에는 그 라벨이 애초에 존재하지 않습니다.
//
//  ── 판정 (다국어 코사인 / 임계치 없음) ──
//   ① resolve_title_values 로 자기선언 값만 남깁니다. 0개면 즉시 커머스 유지.
//   ② 무역 서식 전문 55개 + 커머스 페이지 타입 6개를 하나의 뱅크로 세우고
//      bank_neutral_key_scores 로 채점합니다. (행/열 이중 센터링)
//   ③ argmax 가 무역 코드일 때만 리라우트합니다.
//   커머스 앵커가 같은 경쟁판에 있으므로 '무역이 커머스를 이겼다' 는 사실 자체가 근거입니다.
// =====================================================================
#[derive(Debug, Clone)]
pub struct TradeRerouteVerdict {
    pub code: String,
    pub title: String,
    pub score: f32,
    pub rival: String,
    pub rival_score: f32,
}
pub async fn probe_trade_document(
    model: &LogisModel,
    light_pug: &str,
    doc_lang: &str,
    emit_term: &(dyn Fn(&str) + Send + Sync),
) -> Option<TradeRerouteVerdict> {
    let values = resolve_title_values(model, light_pug, 0.30, emit_term, true).await;
    if values.is_empty() {
        emit_term("  ⚪ [MODE PROBE] 자기 종류를 선언하는 표제가 없습니다. 커머스 판정을 유지합니다.");
        return None;
    }
    let val_embs = model.get_embedding_batch(values.clone()).await
        .unwrap_or_else(|_| vec![vec![0.0; 384]; values.len()]);
    // ── 무역 서식 전문 + 커머스 페이지 타입을 한 판에 올립니다 ──
    let mut bias_defs: Vec<(String, String, String)> = Vec::new();
    for (code, title) in TRADE_DOC_TITLES.iter() {
        bias_defs.push(("trade".to_string(), code.to_string(), title.to_string()));
    }
    const COMMERCE_TYPES: [&str; 6] = ["order", "goods", "tracking", "review", "coupon", "event"];
    for c in COMMERCE_TYPES.iter() {
        let anchor = crate::parsing::get_page_type_classification_bias(c, doc_lang);
        for p in crate::utils::ai_utils::split_bias_phrases_full(&anchor) {
            bias_defs.push(("commerce".to_string(), c.to_string(), p));
        }
    }
    let mut uniq: Vec<String> = Vec::new();
    for (_, _, p) in bias_defs.iter() {
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
    let bank: Vec<(String, String, Vec<f32>)> = bias_defs.iter()
        .map(|(c, k, p)| (c.clone(), k.clone(), emb_of(p))).collect();
    let no_prej: Vec<(String, String, Vec<f32>)> = Vec::new();
    let scores = crate::utils::ai_utils::bank_neutral_key_scores(&val_embs, &bank, &no_prej);
    if scores.is_empty() { return None; }
    let is_commerce = |k: &str| COMMERCE_TYPES.iter().any(|c| *c == k);
    for (k, s) in scores.iter().take(6) {
        emit_term(&format!(
            "     📐 [MODE PROBE] {} {} | Score: {:+.4}",
            if is_commerce(k) { "🛒" } else { "🚢" }, k, s
        ));
    }
    let (top_key, top_score) = scores[0].clone();
    let rival = scores.iter()
        .find(|(k, _)| is_commerce(k) != is_commerce(&top_key))
        .cloned()
        .unwrap_or_else(|| (String::new(), 0.0));
    if is_commerce(&top_key) {
        emit_term(&format!(
            "  🛒 [MODE KEEP] 표제 축 승자가 커머스 '{}' ({:+.4}) 입니다. 무역 리라우트를 하지 않습니다.",
            top_key, top_score
        ));
        return None;
    }
    if top_score <= 0.0 {
        emit_term(&format!(
            "  ⚪ [MODE PROBE] 무역 코드 '{}' 가 1위지만 점수 {:+.4} 로 평균 이하입니다. 리라우트하지 않습니다.",
            top_key, top_score
        ));
        return None;
    }
    let title = TRADE_DOC_TITLES.iter()
        .find(|(c, _)| *c == top_key.as_str())
        .map(|(_, t)| t.to_string())
        .unwrap_or_default();
    Some(TradeRerouteVerdict {
        code: top_key,
        title,
        score: top_score,
        rival: rival.0,
        rival_score: rival.1,
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

    if let Ok(re) = regex::Regex::new(r"\b\d{4}[.\-]\d{2}[.\-]\d{2,4}\b") {
        if let Some(m) = re.find(&upper) {
            found.push(format!("hs:{}", m.as_str()));
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

    (!found.is_empty(), found)
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

fn normalize_trading_data(item: &mut Value, doc_lang: &str) {
    const NUMERIC_KEYS: [&str; 15] = [
        "amount", "amount_subtotal", "amount_tax", "freight_amount", "insurance_amount",
        "local_charges", "package_count", "weight_gross", "weight_net", "volume",
        "unit_price", "total_price", "quantity", "insured_amount", "premium_amount",
    ];
    const DATE_KEYS: [&str; 6] = [
        "issue_date", "expiry_date", "etd", "eta", "shipping_date", "registration_date",
    ];

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

    fn to_iso_date(v: &Value) -> Option<String> {
        let s = match v {
            Value::String(s) => s.trim().to_string(),
            Value::Number(n) => n.to_string(),
            _ => return None,
        };
        if s.is_empty() || s == "N/A" || s == "null" { return None; }
        if s.contains('T') && s.chars().count() >= 19 { return Some(s); }

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

    fn walk(v: &mut Value, numeric: &[&str], date: &[&str]) {
        match v {
            Value::Object(map) => {
                let keys: Vec<String> = map.keys().cloned().collect();
                for k in keys {
                    if numeric.iter().any(|n| *n == k.as_str()) {
                        let converted = map.get(&k).and_then(to_number);
                        if let Some(num) = converted {
                            map.insert(k.clone(), json!(num));
                        }
                        continue;
                    }
                    if date.iter().any(|d| *d == k.as_str()) {
                        let converted = map.get(&k).and_then(to_iso_date);
                        if let Some(iso) = converted {
                            map.insert(k.clone(), json!(iso));
                        }
                        continue;
                    }
                    if let Some(child) = map.get_mut(&k) {
                        if child.is_object() || child.is_array() {
                            walk(child, numeric, date);
                        }
                    }
                }
            },
            Value::Array(arr) => {
                for it in arr.iter_mut() { walk(it, numeric, date); }
            },
            _ => {}
        }
    }

    walk(item, &NUMERIC_KEYS, &DATE_KEYS);

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

/// 🌟 [CONTINUATION EXTRACT] 표제 없는 연속 페이지를 앞 문서의 스키마로 추출합니다.
///
///  ── 왜 별도 경로인가 ──
///   STEP A(문서분류)를 건너뛰는 대신, 앞 페이지가 확정한 doc_type 의 스키마로
///   PLINKO 만 수행합니다. 문서 종류를 다시 묻지 않으므로 오분류가 원천 차단됩니다.
///   LLM 도 호출하지 않습니다. 연속 페이지는 대부분 표 행의 꼬리이고,
///   그 값들은 구조적 라벨-값 페어로 전부 회수되기 때문입니다.
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
    let detail_pairs = crate::utils::ai_utils::collect_detail_label_value_pairs(&pug_lines_ref);
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
    }
    if labels.is_empty() {
        return Ok(out);
    }
    let leaf_embs = model.get_embedding_batch(leafs.clone()).await
        .unwrap_or_else(|_| vec![vec![0.0; 384]; leafs.len()]);
    let mut matrix: Vec<Vec<f32>> = vec![vec![-1.0f32; labels.len()]; f_names.len()];
    for f in 0..f_names.len() {
        let fmt = crate::utils::ai_utils::detect_field_format(&f_names[f]);
        for h in 0..labels.len() {
            if leaf_embs[h].iter().all(|&v| v == 0.0) { continue; }
            if !crate::utils::ai_utils::value_matches_format(fmt, &vals[h]) {
                continue;
            }
            matrix[f][h] = crate::utils::ai_utils::weighted_max_pool_sim(
                &leaf_embs[h], &f_label[f], &f_weight[f],
            );
        }
    }
    let centered = crate::utils::ai_utils::double_center_matrix(&matrix);
    let assign = crate::utils::ai_utils::exclusive_assign_by_score(&centered, 0.0, 0.0);
    use crate::logic::trade_field_category;
    let mut assigned = 0usize;
    for (f, a) in assign.iter().enumerate() {
        let (h, own, margin) = match a { Some(v) => *v, None => continue };
        let fname = f_names[f].clone();
        if crate::utils::ai_utils::is_id_link_field(&fname) { continue; }
        let cat = trade_field_category(&fname);
        if cat.is_empty() { continue; }
        if let Some(slot) = out.get_mut(cat).and_then(|v| v.as_object_mut()) {
            slot.insert(fname.clone(), json!(vals[h].clone()));
            assigned += 1;
        }
        emit_term(&format!(
            "    ✨ [CONTINUATION ASSIGN] Label '{}' → Field '{}' (cat: {}) | Score: {:+.4} | Margin: {:+.4} | Line {} | Value: \"{}\"",
            labels[h], fname, cat, own, margin, lines[h] + 1, vals[h]
        ));
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
    // =====================================================================
    // 🌟 [PAGE CONTINUITY GATE] 표제도 자기 번호도 없으면 앞 페이지의 연속입니다.
    //  ── 왜 STEP A 앞인가 ──
    //   STEP A 는 문서를 '반드시' 어떤 코드로 분류합니다. 후보가 좁혀져도
    //   가장 덜 틀린 코드 하나를 무조건 뽑습니다. 그래서 표제가 없는 꼬리 페이지가
    //   들어오면 반드시 오분류합니다. 분류 자체를 시키지 않는 것이 유일한 해법입니다.
    // =====================================================================
    if page_idx > 0 && !page_results.is_empty() {
        let humanize_c = |raw: &str| -> String {
            let h = crate::utils::ai_utils::humanize_url_token(raw);
            if h.trim().is_empty() { raw.trim().to_string() } else { h }
        };
        // ── ① 표제 축 ──
        //
        //  🌟 [SHARED RESOLVER] STEP A 와 동일한 판정기를 씁니다.
        //
        //  ── v1 이 왜 실패했나 (실측) ──
        //   🧭 [PAGE CONTINUITY] 표제 축: +2.1194 | 자기번호 축: 없음 → 독립 문서
        //   [ROW MARKER DROP] 로그가 한 줄도 없었습니다. 즉 아무것도 안 걸러졌습니다.
        //   라벨 게이트가 '자기선언 vs 타문서참조' 2지 선택이라
        //     'item code'   자기선언 0.6187 vs 타문서참조 0.5412 → 통과
        //     'description' 자기선언 0.6000 vs 타문서참조 0.5295 → 통과
        //   가 되었고, 그 값 "SKU-B200" / "Sensor Modules" 가 제목 축을 만들었습니다.
        //   품목 속성 축을 세 번째 선택지로 넣으면 두 라벨이 그쪽으로 갑니다.
        let mut title_top = f32::MIN;
        {
            let vals = resolve_title_values(&model, &light_pug, 0.30, &emit_term, false).await;
            if !vals.is_empty() {
                let ve = model.get_embedding_batch(vals.clone()).await
                    .unwrap_or_else(|_| vec![vec![0.0; 384]; vals.len()]);
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
                let s = crate::utils::ai_utils::bank_neutral_key_scores(&ve, &tbb, &tpb);
                if let Some((_, top)) = s.first() { title_top = *top; }
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
    // 🌟 [BANK-NEUTRAL GROUP SCORING] √(2 ln N) 차감을 폐기합니다.
    //    그룹마다 앵커 구 수가 다르면(shipping 33구 vs customs 17구)
    //    구 수가 적은 그룹이 구조적으로 유리해집니다.
    //    vision_encoder.rs 가 이미 폐기한 편향이라 텍스트 트랙도 동일하게 맞춥니다.
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
    // 🌟 [GROUP EXPANSION GATE]
    //  ── 무엇이 문제였나 (실측) ──
    //   기존 게이트는 `*s <= 0.0` 이었습니다. 그런데 점수는 48개 라인의 최댓값이라
    //   라인이 조금만 많아지면 7개 그룹이 전부 양수가 됩니다.
    //   로그에서 후보가 55개 전부 열려 Depth-1 승리('contract')가 무의미해졌고,
    //   contract 에 없는 CI 가 후보로 들어와 PO 를 이겼습니다.
    //
    //  ── 대체 기준 (매직 상수 아님) ──
    //   '이 문서에서 관측된 양수 그룹 점수 분포의 (평균 + 표준편차)'.
    //   models/siglip2/vision_crop.rs::core_threshold 가 쓰는 것과 같은 도구이며,
    //   문서 자신의 분포에서 유도되므로 문서/언어에 따라 자동으로 조정됩니다.
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
    // =====================================================================
    // 🌟 [TITLE AXIS — 텍스트 트랙]
    // ---------------------------------------------------------------------
    //  vision_encoder.rs 의 run_title_gate + TITLE AXIS NMS INTEGRATION 을
    //  텍스트 트랙에 이식합니다. 같은 문서를 스캔 이미지로 넣든 PDF 로 넣든
    //  같은 코드가 나와야 하므로 두 트랙의 판정 근거를 통일합니다.
    //
    //  ① 라벨 축 : '자기 종류 선언' vs '타 문서 참조' 상대 코사인
    //  ② 값 축   : 살아남은 값 ↔ TRADE_DOC_TITLES 전문 뱅크 (뱅크 중립 채점)
    // =====================================================================
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
            let mut value_texts: Vec<String> = Vec::new();
            for c in cands.iter() {
                let keep = if c.label.trim().is_empty() {
                    true
                } else {
                    label_is_title.get(&humanize(&c.label)).copied().unwrap_or(false)
                };
                if !keep { continue; }
                if !value_texts.iter().any(|e| e == &c.value) {
                    value_texts.push(c.value.clone());
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
                title_scores = crate::utils::ai_utils::bank_neutral_key_scores(
                    &val_embs, &t_bias_bank, &t_prej_bank,
                );
                for (c, s) in title_scores.iter().take(6) {
                    emit_term(&format!("     📐 [TITLE AXIS] {} | Score: {:+.4}", c, s));
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

    // 🌟 [BANK-NEUTRAL CODE SCORING]
    //  ── 왜 바꾸는가 ──
    //   후보가 55개일 때 각 코드의 편견 뱅크 N 이 54배로 폭발해
    //   √(2 ln N) ≈ 3.4 가 차감되고, surprisal_dual_scores 의
    //   `if ps > 0.0 { sc -= ps; }` 가 거의 항상 거짓이 되었습니다.
    //   즉 CI 의 편견 안에 'purchase order' 가 있어도 한 번도 깎지 못했습니다.
    //   행/열 이중 센터링은 뱅크 크기와 무관하므로 이 역설이 사라집니다.
    let mut code_scores: Vec<(String, f32)> = crate::utils::ai_utils::bank_neutral_key_scores(
        &line_embs, &c_bias_bank, &c_prej_bank,
    );
    if code_scores.is_empty() {
        code_scores.push((codes[0].to_string(), 0.0));
    }
    // 🌟 [TITLE AXIS INTEGRATION] 제목 축을 본문 축과 같은 점수 공간에 합산합니다.
    //    가중치 2.0 은 vision_encoder.rs 의 TITLE AXIS NMS INTEGRATION 과 동일하며,
    //    두 트랙이 같은 문서에서 같은 코드를 내도록 맞춘 값입니다.
    //    사후 오버라이드가 아니라 단일 점수 축 편입이므로 NMS 가 결정론적으로 판정합니다.
    if !title_scores.is_empty() {
        let mut merged = 0usize;
        for (cname, cs) in code_scores.iter_mut() {
            if let Some((_, ts)) = title_scores.iter().find(|(t, _)| t == cname) {
                *cs += 2.0 * ts;
                merged += 1;
            }
        }
        code_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        emit_term(&format!(
            "  🪪 [TITLE AXIS MERGE] 본문 축에 제목 축(가중치 2.0)을 합산했습니다. (코드 {}개)",
            merged
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
    emit_term("[TRADING STEP B] Running PLINKO field assignment before LLM...");

    let categories = ["header", "parties", "logistics", "conditions", "financials", "cargo", "items", "containers"];

    let mut final_data_map = serde_json::Map::new();
    final_data_map.insert("header".to_string(), json!({"doc_type": doc_type.clone()}));
    final_data_map.insert("parties".to_string(), json!({}));
    final_data_map.insert("logistics".to_string(), json!({}));
    final_data_map.insert("conditions".to_string(), json!({}));
    final_data_map.insert("financials".to_string(), json!({}));
    final_data_map.insert("cargo".to_string(), json!({}));
    final_data_map.insert("line_items".to_string(), json!([]));
    final_data_map.insert("containers".to_string(), json!([]));

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
    // =====================================================================
    // 🌟 [SECTION RECOVERY] 섹션이 비면 상위 들여쓰기 라벨로 복원합니다.
    // ---------------------------------------------------------------------
    //  ── 실측 사고 ──
    //   로그의 페어 22개가 전부 Section: '' 였습니다.
    //     Line 34 | Section: '' | Label: 'name'    | Value: 'Nova Importers LLC'
    //     Line 45 | Section: '' | Label: 'name'    | Value: 'Global Tech Logistics KR'
    //     Line 37 | Section: '' | Label: 'address' | Value: '456 Market St, ...'
    //     Line 48 | Section: '' | Label: 'address' | Value: '123 Teheran-ro, ...'
    //   buyer / seller 라는 결정적 구분 정보가 통째로 사라졌고,
    //   is_multi_value_field('party_address') 가 참이라 두 주소가 한 값으로 접합되어
    //   sender_address / recipient_address 가 둘 다 비었습니다.
    //
    //  ── 복원 근거 (구조 사실) ──
    //   YAML/PUG 는 들여쓰기로 소속을 표현합니다.
    //   자기보다 얕은 들여쓰기의 가장 가까운 앞 라인이 곧 소속 섹션입니다.
    //   어휘 사전이 아니라 공백 개수만 보므로 언어와 무관합니다.
    // =====================================================================
    {
        // 🌟 [SECTION RECOVERY v2] 들여쓰기가 아니라 '직전 미소비 텍스트 행' 으로 복원합니다.
        //
        //  ── v1 이 왜 실패했나 (실측: [SECTION RECOVERY] 로그 0건) ──
        //   collect_detail_label_value_pairs 의 섹션은 heading 태그(h1~h6/legend/caption)
        //   에서만 나옵니다. 그런데 pdf_page_to_structured_html 은
        //     <table><tr><th>라벨</th><td>값</td></tr>
        //     <tr><td colspan="2">parties:</td></tr>
        //   만 만들고 heading 을 하나도 만들지 않습니다.
        //   게다가 표 행이 전부 같은 깊이라 들여쓰기 비교도 성립하지 않습니다.
        //
        //  ── 올바른 구조 근거 ──
        //   그룹 제목은 '라벨-값 페어가 되지 못한 단독 텍스트 행' 으로 나타납니다.
        //   실측 라인 간격이 이를 뒷받침합니다.
        //     27(related_lc_number) → 34(name)  : 7칸 (parties: / buyer: 두 행)
        //     40(contact_number)    → 45(name)  : 5칸 (seller: 한 행)
        //   따라서 label_line 에서 뒤로 걸어 가장 가까운 '미소비 텍스트 행' 이 곧 섹션입니다.
        //   태그명·어휘를 보지 않고 소비 여부만 보므로 언어와 무관합니다.
        let mut consumed: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for p in detail_pairs.iter() {
            consumed.insert(p.primary_line);
            consumed.insert(p.label_line);
        }
        let mut recovered = 0usize;
        for p in detail_pairs.iter_mut() {
            if !p.section.trim().is_empty() { continue; }
            let li = p.label_line.min(pug_lines.len().saturating_sub(1));
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
            if !sec.is_empty() {
                p.section = sec;
                recovered += 1;
            }
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
    let mut label_count: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for p in &detail_pairs { *label_count.entry(p.label.clone()).or_insert(0) += 1; }

    let mut pair_phrases: Vec<String> = Vec::with_capacity(detail_pairs.len());
    // 🌟 [SECTION-QUALIFIED PHRASE] 섹션이 복원되었으므로 중복 라벨이
    //    'buyer name' / 'seller name' 처럼 서로 다른 구가 됩니다.
    //    이것이 party_address 접합 사고를 구조적으로 막습니다.
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
        // 🌟 [MULTI-VALUE SCOPE] 다중 값 접합은 '같은 섹션 안' 에서만 허용합니다.
        //
        //  ── 실측 사고 ──
        //   phrase_multi 가 섹션을 무시하고 이어붙여
        //     party_address = "456 Market St, ... USA 123 Teheran-ro, ... South Korea"
        //   가 되었습니다. 서로 다른 회사의 주소 두 개가 한 값이 되었습니다.
        //
        //  ── 규칙 ──
        //   value_all 은 '같은 셀 안의 여러 줄' 을 담기 위한 축입니다.
        //   섹션이 다르면 물리적으로 다른 블록이므로 접합 대상이 아닙니다.
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

        
        let pair_abs_floor = 0.50f32;
        let mut leaf_raw: Vec<Vec<f32>> = vec![vec![-1.0f32; unique_labels.len()]; t_field_names.len()];
        let mut sec_raw:  Vec<Vec<f32>> = vec![vec![-1.0f32; unique_labels.len()]; t_field_names.len()];

        for f in 0..t_field_names.len() {
            let f_fmt = crate::utils::ai_utils::detect_field_format(&t_field_names[f]);
            let f_multi = crate::utils::ai_utils::is_multi_value_field(&t_field_names[f]);
            let f_strict = matches!(
                f_fmt,
                crate::utils::ai_utils::FieldFormat::Date
                    | crate::utils::ai_utils::FieldFormat::TrackingCode
                    | crate::utils::ai_utils::FieldFormat::Numeric
                    | crate::utils::ai_utils::FieldFormat::Phone
                    | crate::utils::ai_utils::FieldFormat::Address
                    | crate::utils::ai_utils::FieldFormat::Text
            );

            // 🌟 [BANK COHESION] 이 필드 라벨 뱅크의 내부 응집도를 1회만 계산합니다.
            //    응집도가 높은 필드(동의어가 촘촘한 필드)는 편견 뱅크와도
            //    구조적으로 가깝게 나오므로 그만큼 관대한 기준이 필요합니다.
            let f_cohesion = crate::utils::ai_utils::bank_internal_cohesion(&t_label_embs[f]);
            for h in 0..unique_labels.len() {
                if leaf_embs[h].iter().all(|&v| v == 0.0) { continue; }
                let own = crate::utils::ai_utils::weighted_max_pool_sim(
                    &leaf_embs[h], &t_label_embs[f], &t_label_weights[f]
                );
                if own < pair_abs_floor { continue; }
                let prej = if t_prej_embs[f].is_empty() {
                    0.0
                } else {
                    crate::utils::ai_utils::max_pool_sim(&leaf_embs[h], &t_prej_embs[f])
                };
                // 🌟 [RELATIVE PREJUDICE GATE]
                //  ── 실측 사고 ──
                //   🚫 'address' → 'recipient_address' | Label: 0.8061 <= Prej: 0.8374
                //   전 조합 중 최고 점수가 삭제되어, 편견이 약한 party_address 만 남았고
                //   buyer 주소와 seller 주소가 한 값으로 접합되었습니다.
                //   ("456 Market St, ... USA 123 Teheran-ro, ... South Korea")
                //  ── 왜 상대 비교인가 ──
                //   "이 라벨이 어느 필드에 속하는가" 는 배타 배정이 이미 판정합니다.
                //   편견의 역할은 '정반대 개념 차단' 이므로, 자기 점수를 뱅크 응집도만큼의
                //   여유 이상으로 앞설 때만 폐기합니다. 새 매직 상수가 없습니다.
                if crate::utils::ai_utils::prejudice_dominates(own, prej, f_cohesion) {
                    emit_term(&format!("    🚫 [TRADING PREJUDICE GATE] '{}' → '{}' | Label: {:.4} | Prej: {:.4} | Cohesion: {:.4} (상대 우위 초과)",
                        unique_labels[h], t_field_names[f], own, prej, f_cohesion));
                    continue;
                }
                let pair_val = if f_multi { &phrase_multi[h] } else { &phrase_single[h] };
                // 🌟 [FORMAT GATE ALWAYS] f_strict 분기를 제거합니다.
                //
                //  ── 왜 제거하는가 ──
                //   f_strict 는 Identifier / Link / Enum / Synthesis 를 게이트에서 뺐습니다.
                //   그런데 value_matches_format 은 그 포맷들에 대해서도 명확한 판정을 갖고
                //   있으므로 예외를 둘 이유가 없습니다.
                //   오히려 예외 때문에 Enum 필드(incoterms 등)에 수치가 들어가는 것을
                //   별도 게이트(TRADING ENUM NUMERIC GATE)로 따로 막아야 했습니다.
                //   그 별도 게이트는 아래에 그대로 남겨 두어 이중 방어를 유지합니다.
                if pair_val.trim().is_empty()
                    || !crate::utils::ai_utils::value_matches_format(f_fmt, pair_val) {
                    emit_term(&format!("    🚫 [TRADING VALUE FORMAT GATE] '{}' → '{}' ({:?}) | 값 \"{}\" 형식 불일치",
                        unique_labels[h], t_field_names[f], f_fmt, pair_val));
                    continue;
                }
                let _ = f_strict;
                if f_fmt == crate::utils::ai_utils::FieldFormat::Enum
                    && crate::utils::ai_utils::is_pure_numeric_value(pair_val) {
                    emit_term(&format!("    🚫 [TRADING ENUM NUMERIC GATE] '{}' → '{}' | 값 \"{}\" 은 순수 수치",
                        unique_labels[h], t_field_names[f], pair_val));
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
        for h in 0..unique_labels.len() {
            let mut sec_sum = 0.0f32;
            let mut sec_cnt = 0usize;
            for f in 0..t_field_names.len() {
                if leaf_raw[f][h] < 0.0 { continue; }
                if sec_raw[f][h] < 0.0 { continue; }
                sec_sum += sec_raw[f][h];
                sec_cnt += 1;
            }
            let sec_mean = if sec_cnt > 0 { sec_sum / (sec_cnt as f32) } else { 0.0 };
            for f in 0..t_field_names.len() {
                if leaf_raw[f][h] < 0.0 { continue; }
                let sec_term = if sec_cnt > 1 && sec_raw[f][h] >= 0.0 {
                    sec_raw[f][h] - sec_mean
                } else {
                    0.0
                };
                t_matrix[f][h] = leaf_raw[f][h] + SECTION_WEIGHT * sec_term;
            }
        }

        let t_assign = crate::utils::ai_utils::exclusive_assign_by_score(&t_matrix, 0.0, 0.0);

        use crate::logic::trade_field_category;

        for (f, a) in t_assign.iter().enumerate() {
            let (h, score, margin) = match a { Some(v) => *v, None => continue };
            let fname = t_field_names[f].clone();
            if crate::utils::ai_utils::is_id_link_field(&fname) { continue; }

            let f_multi = crate::utils::ai_utils::is_multi_value_field(&fname);
            let val = if f_multi { phrase_multi[h].clone() } else { phrase_single[h].clone() };
            if val.trim().is_empty() { continue; }

            let cat = trade_field_category(&fname);
            if cat.is_empty() {
                emit_term(&format!("    ⚪ [TRADING CATEGORY UNMAPPED] '{}' 는 8개 카테고리에 매핑되지 않아 루트에만 주입합니다.", fname));
            } else if let Some(slot) = final_data_map.get_mut(cat).and_then(|v| v.as_object_mut()) {
                slot.insert(fname.clone(), json!(val.clone()));
            }

            assigned_fields.insert(fname.clone(), val.clone());
            emit_term(&format!("    ✨ [TRADING PLINKO ASSIGN] Label '{}' → Field '{}' (cat: {}) | Score: {:+.4} | Margin: {:+.4} | Line {} | Value: \"{}\"",
                unique_labels[h], fname, if cat.is_empty() { "-" } else { cat }, score, margin, phrase_line[h] + 1, val));
        }

        emit_term(&format!("  ✅ [TRADING PLINKO] LLM 없이 {}개 필드 확정 완료.", assigned_fields.len()));
    }

    
    model.secure_vram_relay(crate::model::ModelSize::Qwen3_5, None, Some(cancellation_token.clone()), false, None).await?;

    for cat in &categories {
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        let schema_prompt = crate::parsing::get_trade_category_schema(cat, &doc_type);

        if schema_prompt.contains("SCHEMA:\n{}") || schema_prompt.contains("SCHEMA:\n[ {} ]") {
            emit_term(&format!("[TRADING STEP B] Category '{}' has no fields for {}. Skipping.", cat.to_uppercase(), doc_type));
            continue;
        }

        if *cat != "items" && *cat != "containers" {
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
                            content: format!("[PUG CONTENT — attribute-stripped]\n{}{}", content_pug, claimed_ctx),
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
                for k in ks {
                    if assigned_fields.contains_key(&k) {
                        obj.remove(&k);
                        emit_term(&format!("    🛡️ [PLINKO PROTECT] '{}' 는 결정론 확정값을 유지하고 LLM 결과를 폐기합니다.", k));
                    }
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
        const TRADE_GROUPS_FLAT: [&str; 6] =
            ["header", "parties", "logistics", "financials", "conditions", "cargo"];

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
            emit_term(&format!(
                "  🧹 [RELAY SELF-LOOP] {}.{} 가 자기 문서번호({})와 같아 릴레이를 건너뜁니다.",
                doc_type, mine_field, ref_display
            ));
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

        
        
        
        
        
        //
        
        
        
        
        let mut hit: Option<(String, Value)> = None;
        
        
        
        
        
        //
        
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
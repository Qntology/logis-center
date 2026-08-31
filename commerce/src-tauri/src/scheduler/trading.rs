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

/// 🌟 [TRADE STRUCTURE GATE] 국제 표준 '포맷' 만으로 무역 서식 여부를 판정합니다.
///  ── 왜 하드코딩이 아닌가 ──
///   여기 등장하는 리터럴은 어떤 언어의 어휘도 아니고 국제 표준 식별자 규격입니다.
///     · ISO 6346 : 컨테이너 번호 = 대문자 4 + 숫자 7
///     · IATA     : AWB 번호 = 항공사 3자리 + 8자리
///     · WCO HS   : 4+2(+2~4) 자리 관세 분류 코드
///     · Incoterms 2020 : ICC 가 정한 3자 대문자 표준 약어 11종
///   ai_utils::value_matches_format / id_shape_signature 가 이미 쓰는
///   '구조 판정' 과 같은 계열이며, bias.json 의 trade_schema 도 동일 규격을 명시합니다.
///
///  ── 무엇에 쓰는가 ──
///   이 증거가 하나라도 잡히면 그 문서는 물리적으로 '택배 라벨' 일 수 없습니다.
///   택배 라벨에는 컨테이너 번호도, Incoterms 도, HS Code 도 존재하지 않습니다.
///   따라서 parcel 그룹(= TRACKING) 을 거부하는 veto 근거로만 사용합니다.
fn trade_structural_evidence(pug: &str) -> (bool, Vec<String>) {
    let upper = pug.to_uppercase();
    let mut found: Vec<String> = Vec::new();

    // ① ISO 6346 컨테이너 번호
    if let Ok(re) = regex::Regex::new(r"\b[A-Z]{4}\s?\d{7}\b") {
        if let Some(m) = re.find(&upper) {
            found.push(format!("container:{}", m.as_str().trim()));
        }
    }

    // ② IATA Air Waybill 번호
    if let Ok(re) = regex::Regex::new(r"\b\d{3}-\d{8}\b") {
        if let Some(m) = re.find(&upper) {
            found.push(format!("awb:{}", m.as_str()));
        }
    }

    // ③ WCO HS Code (구분자 포함 형태만 인정 — 순수 숫자열은 오탐이 큽니다)
    if let Ok(re) = regex::Regex::new(r"\b\d{4}[.\-]\d{2}[.\-]\d{2,4}\b") {
        if let Some(m) = re.find(&upper) {
            found.push(format!("hs:{}", m.as_str()));
        }
    }

    // ④ Incoterms 2020 표준 3자 코드 (토큰 완전일치)
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

    // ⑤ B/L 표기 (해상 선하증권에만 인쇄되는 표준 약어)
    if upper.contains("B/L") {
        found.push("bl_label".to_string());
    }

    (!found.is_empty(), found)
}

/// 🌟 [PAGE MERGE] 페이지 단위 추출 결과를 하나의 문서 맵으로 접습니다.
///  ── 병합 규칙 ──
///   · 객체(header/parties/logistics/conditions/financials/cargo)
///     : 앞 페이지 값이 우선. 비어 있을 때만 뒤 페이지 값으로 채웁니다.
///       1페이지가 원본 서식이고 2페이지 이후는 continuation sheet 이기 때문입니다.
///   · 배열(line_items/containers)
///     : 뒤 페이지 항목을 이어붙입니다. 완전 동일 항목만 중복 제거합니다.
///       (품목 목록은 페이지를 넘겨가며 이어지는 것이 정상입니다)
///
///  ── merge_json_manual 을 쓰지 않는 이유 ──
///   그 함수는 객체 병합에서 무조건 덮어씁니다(insert).
///   페이지 병합은 '앞 페이지 우선' 이어야 하므로 반대 정책이 필요합니다.
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
        // ── 배열 축 : 이어붙이기 ──
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

        // ── 객체 축 : 빈 슬롯만 채우기 ──
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

        // ── 스칼라 축 ──
        if is_empty_val(src_val) { continue; }
        let need = match target.get(cat) {
            None => true,
            Some(cur) => is_empty_val(cur),
        };
        if need { target.insert(cat.clone(), src_val.clone()); }
    }
}

/// 🌟 [TRADING NORMALIZE] commerce 의 normalize_data 와 같은 역할을 무역 축에 적용합니다.
///  ── 왜 필요한가 ──
///   update_team_base_metrics 는 data 루트의 수치 축을 스캔해 min / max / avg 를 만듭니다.
///   그런데 LLM 은 금액을 "1,250.00 USD", 중량을 "12,500 KG", 날짜를 "15/03/2026" 처럼
///   문자열로 돌려주므로, 정규화가 없으면 통계 축이 통째로 죽습니다.
///   commerce 경로에는 normalize_data 가 있지만 trading 경로에는 그 단계 자체가
///   존재한 적이 없어서 "평균 / 최대 / 최소" 쿼리가 성립하지 않았습니다.
///
///  ── 무엇을 하는가 ──
///   ① 수치 축을 f64 로 확정 (천 단위 콤마 / 통화기호 / 단위 접미어 제거)
///   ② 날짜 축을 ISO 8601 로 확정
///   ③ currency 를 대문자 ISO 4217 로 확정, 없으면 문서 언어 기준 기본값
///   ④ etd / eta 를 started_at / expired_at 로 승격 (commerce 의 기간 축과 동일 이름)
///   중첩된 line_items / containers 안의 수치도 함께 정규화합니다.
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
        // DD-MM-YYYY 형태 보정
        if day > 31 && year <= 31 {
            year = nums[2];
            day = nums[1];
            month = nums[0];
        }
        if year < 100 { year += if year > 50 { 1900 } else { 2000 }; }
        // 15/03/2026 처럼 일이 앞에 온 경우 보정
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

/// 🌟 [PDF STRUCTURE RECOVERY] PDF 한 페이지 텍스트를 '라벨-값 구조' 로 복원합니다.
///  ── 무엇이 문제였나 ──
///   기존 변환은 모든 줄을 <div> 로만 만들었습니다.
///   collect_detail_label_value_pairs 는 tr → th/td 구조를 전제로 하므로
///   PDF 경로에서는 페어가 영구적으로 0개였습니다.
///   (로그 실측: 4개 페이지 전부 "구조적 라벨-값 페어 0개 확보")
///   그 결과 40개 스키마 필드를 전부 2B LLM 단독 추론에 맡겨야 했고,
///   Task1 의 CI 페이지는 HEADER 가 통째로 빈값이 되었습니다.
///
///  ── 판정 규칙 (어휘 사전 없음, 문자열 구조만) ──
///   R1: 첫 콜론(':' 또는 '：') 앞이 라벨, 뒤가 값.
///       단 URL('http://')과 시각('14:30')을 콜론으로 자르지 않도록,
///       콜론 뒤에 공백이 있거나 콜론 앞이 2자 이상 비숫자일 때만 인정합니다.
///   R2: 콜론이 없으면 '2칸 이상 연속 공백 또는 탭' 을 구분자로 봅니다.
///       PDF 표는 셀 사이가 넓은 공백으로 렌더링되므로 이것이 사실상의 셀 경계입니다.
///   R3: 라벨은 40자 이하이고 알파벳/한글 문자를 하나 이상 가져야 합니다.
///       (순수 숫자 라벨은 표의 값이지 라벨이 아닙니다)
///   R4: 위 어느 것도 아니면 기존처럼 <div> 한 줄로 둡니다.
///
///  반환값: (HTML, 복원된 라벨-값 행 개수)
fn pdf_page_to_structured_html(page_text: &str) -> (String, usize) {
    fn esc(s: &str) -> String {
        s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
    }

    fn is_label_like(s: &str) -> bool {
        let t = s.trim();
        if t.is_empty() { return false; }
        if t.chars().count() > 40 { return false; }
        if !t.chars().any(|c| c.is_alphabetic()) { return false; }
        // 순수 숫자 덩어리가 라벨 자리에 온 경우는 값입니다.
        let digits = t.chars().filter(|c| c.is_ascii_digit()).count();
        let alnum = t.chars().filter(|c| c.is_alphanumeric()).count().max(1);
        digits * 2 < alnum
    }

    /// 콜론 분해. URL / 시각을 자르지 않습니다.
    fn split_by_colon(line: &str) -> Option<(String, String)> {
        let chars: Vec<char> = line.chars().collect();
        for (i, c) in chars.iter().enumerate() {
            if *c != ':' && *c != '：' { continue; }
            if i == 0 { continue; }
            let head: String = chars[..i].iter().collect();
            let tail: String = chars[i + 1..].iter().collect();

            // 'http://' / 'https://' 방어
            if tail.starts_with("//") { continue; }
            // '14:30' 방어 — 콜론 양쪽이 모두 숫자면 시각입니다.
            let prev_digit = chars[i - 1].is_ascii_digit();
            let next_digit = chars.get(i + 1).map_or(false, |c| c.is_ascii_digit());
            if prev_digit && next_digit { continue; }

            if !is_label_like(&head) { continue; }
            return Some((head.trim().to_string(), tail.trim().to_string()));
        }
        None
    }

    /// 2칸 이상 연속 공백 / 탭 분해.
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
                // 라벨-값이 아닌 줄(제목/문단/표 헤더)은 단일 셀 행으로 유지합니다.
                rows.push_str(&format!(
                    "<tr><td colspan=\"2\">{}</td></tr>\n",
                    esc(line.trim())
                ));
            }
        }
    }

    (format!("<table>\n{}</table>", rows), pair_cnt)
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

    // ── 모델 로드 ──
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

    // ── HTML 전처리 ──
    // 🌟 [SOURCE RESOLUTION v3 / PAGE-WISE]
    //  ── 무엇이 바뀌었나 ──
    //   v2 는 문서 전체를 String 하나로 만들었습니다. PDF 5장이 한 덩어리가 되어
    //   STEP A 가 doc_type 을 1개만 뽑고, STEP B 도 1회만 돌았습니다.
    //   여기서는 '페이지 배열' 을 만들어 STEP A/B 를 페이지마다 독립 수행하고,
    //   추론 결과는 STEP C 직전에 doc_type 별로 합칩니다.
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

    // ── URL 파싱 (페이지 루프 밖에서 1회만) ──
    //    기존에는 이 함수 안에서 resolve_absolute_url 을 두 번 호출하고
    //    두 번째가 첫 번째를 shadow 하는 죽은 코드가 있었습니다. 한 번으로 통합합니다.
    let (url, _origin_candidate) = crate::utils::url_utils::resolve_absolute_url(&task_data).await;

    let total_pages = page_htmls.len();
    let mut page_results: Vec<(String, String, serde_json::Map<String, Value>)> = Vec::new();

    // 🌟 [PAGE LOOP OPEN] 이 아래 STEP A / STEP B 전체가 페이지마다 1회씩 수행됩니다.
    //    (Rust 는 들여쓰기를 문법으로 삼지 않으므로 기존 코드의 들여쓰기는 그대로 둡니다)
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

    // 🌟 [PUG PIPELINE] 원문 HTML을 직접 사용하지 않습니다.
    //    ① pre_clean_html      : script/style/noscript/iframe/svg 제거, 허용 속성만 유지
    //    ② convert_to_clean_pug : DOM → PUG 변환 (NoAttributesMode = 속성 노이즈 제거)
    //    ③ truncate_pug_context : 토큰 상한 적용
    //    이 3단 파이프라인을 거친 결과를 변수에 저장하여
    //    이후 STEP A(분류) / STEP B(추출)에서 재사용합니다.
    let clean_html_content = parsing::pre_clean_html(&raw_html_content);

    // 🌟 [URL FIX] base_url 을 None 이 아닌 실제 추출 주소로 전달하여
    //    상대경로 href 가 절대경로 해석되도록 합니다.
    let raw_pug =
        parsing::convert_to_clean_pug(&clean_html_content, PugMode::NoAttributesMode, Some(&url));
    let light_pug = model
        .truncate_pug_context(&raw_pug, false, 2000, None)
        .await;

    // 문서 언어 감지 (페이지마다 재확정 — 다국어 묶음 PDF 대응)
    doc_lang = crate::utils::lang_utils::detect_document_language(&light_pug);
    println!("[TRADING] Detected document language (page {}): {}", page_idx + 1, doc_lang);

    // =====================================================================
    // 🌟 [TRADING STEP A v2] doc_type 2뎁스 분류 (그룹 → 코드)
    // ---------------------------------------------------------------------
    //  ── 왜 2뎁스인가 ──
    //   commerce 는 6개 도메인(order/goods/tracking/review/coupon/event)을
    //   벡터 1회로 갈랐지만, trading 은 27개라 1회 코사인으로 정확히 가를 수 없습니다.
    //   (get_trade_doc_classification_prompt 도 같은 이유로 GROUPS 를 먼저 제시합니다)
    //
    //   그래서 slice_config 가 이미 좌표를 공유하는 그룹 단위로 먼저 좁힙니다.
    //     ① Contract & Payment   : PO / PI / SC / LC
    //     ② Shipping & Transport : CI / PL / BL / AWB / SA / DO / AN / BC
    //     ③ Customs              : ED / ID / CINV / CO
    //     ④ Inspection           : IC / WC / CA / PHYTO / HC / BEN_CERT
    //     ⑤ Special & Legal      : DGD / MSDS / POA / BIZ_LIC / INS
    //     ⑥ Parcel               : TRACKING
    //
    //   그룹 내 혼동은 slice_config / category_schema 가 동일 좌표·동일 스키마를
    //   공유하므로 추출 품질에 영향이 없습니다. 그룹 간 오분류만 막으면 됩니다.
    //
    //  ── LLM 호출 절감 ──
    //   코사인 마진이 충분하면 LLM 을 아예 부르지 않습니다.
    //   마진이 부족한 경우에만 '그 그룹의 코드 목록'만 제시하여 1회 호출합니다.
    // =====================================================================
    emit_term("[TRADING STEP A] Classifying trade document type (2-depth)...");
    log_task_progress(app_handle, &task.id, &json!({
        "category": "Classification", "summary": "Identifying trade document group...", "spinner": "⠋"
    }));

    // ── 임베딩 모델 확보 (LLM 이전에 코사인 분류부터) ──
    model.check_embedding_downloaded().await?;
    model.ensure_embedding().await?;

    // 🌟 [ANCHOR SOURCE] 그룹/코드 앵커 사전은 logic.rs 가 소유합니다.
    //  ── 왜 옮겼는가 ──
    //   같은 사전을 비전 파이프라인(models/siglip2/vision_encoder.rs)도 씁니다.
    //   지역 const 로 두면 서식이 하나 늘 때마다 두 곳을 고쳐야 하고,
    //   텍스트 트랙과 비전 트랙의 판정 근거가 갈라집니다.
    //   editing point 를 하나로 고정하기 위해 참조만 남깁니다.
    //
    //   · logic::TRADE_GROUPS        : 그룹 앵커 (편견 = 다른 그룹의 bias)
    //   · logic::TRADE_GROUP_CODES   : 그룹 → 코드 목록
    //   · logic::trade_code_anchor() : 코드별 앵커 구
    use crate::logic::{TRADE_GROUPS, TRADE_GROUP_CODES as GROUP_CODES, trade_code_anchor};

    // ── 질의(문서) 측 : 라인 단위 분해 ──
    //  ── 왜 바꾸는가 ──
    //   기존에는 light_pug(최대 9000 토큰) 전체를 384차원 1벡터로 만들었습니다.
    //   문서 전체 평균이라 그룹 간 미세한 차이가 전부 소멸합니다.
    //   commerce 가 슬라이딩 윈도우를 쓰는 이유와 정확히 같은 문제이므로,
    //   '텍스트를 지닌 라인' 만 뽑아 라인별로 채점하고 그룹별 최댓값을 취합니다.
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

    // ── 구조 증거 (언어 무관 / 국제 표준 포맷) ──
    let (has_trade_marker, trade_markers) = trade_structural_evidence(&light_pug);
    if has_trade_marker {
        emit_term(&format!("  🔩 [TRADE STRUCTURE] 국제 표준 포맷 증거 발견: {:?}", trade_markers));
    } else {
        emit_term("  ⚪ [TRADE STRUCTURE] 국제 표준 포맷 증거가 없습니다. (택배 라벨 가능성 열림)");
    }

    // ── 뎁스 1 : 그룹 구 뱅크 SURPRISAL ──
    //  surprisal = (max - μ_global)/σ_global - √(2 ln N)
    //  ai_utils::surprisal_dual_scores 가 이미 구현해 둔 극값이론 정규화를 그대로 씁니다.
    //  · 센트로이드 폐기 → 구 단위 Max-Pool
    //  · 뱅크 크기 편향 제거 → shipping(24구) 이 parcel(9구) 보다 불리해지지 않음
    //  · 편견 상쇄        → 다른 그룹의 bias 구가 자동으로 이 그룹의 편견이 됨
    //  surprisal > 0 = "N개를 무작위로 뽑은 기대 최댓값보다 실제로 더 가깝다" 이므로
    //  0 은 극값이론에서 유도된 값이며 매직 상수가 아닙니다.
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

    // 구 문자열은 6개 그룹 앵커에서만 나오므로 유일 구만 1회 임베딩하고 재사용합니다.
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

    let empty_names: Vec<String> = Vec::new();
    let empty_banks: Vec<Vec<Vec<f32>>> = Vec::new();
    let empty_skip: Vec<bool> = Vec::new();

    let mut group_best: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    for le in line_embs.iter() {
        if le.iter().all(|&v| v == 0.0) { continue; }
        let (fs, _) = crate::utils::ai_utils::surprisal_dual_scores(
            le, &g_bias_bank, &g_prej_bank, &empty_names, &empty_banks, &empty_skip,
        );
        for s in fs {
            let e = group_best.entry(s.key.clone()).or_insert(f32::MIN);
            if s.surprisal > *e { *e = s.surprisal; }
        }
    }

    let mut group_scores: Vec<(String, f32)> = group_best.into_iter().collect();
    group_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if group_scores.is_empty() {
        group_scores.push(("shipping".to_string(), 0.0));
    }
    for (g, s) in group_scores.iter() {
        emit_term(&format!("  📐 [TRADE GROUP] {} | Surprisal(max over lines): {:+.4}", g, s));
    }

    let mut best_group = group_scores[0].0.clone();
    let group_margin = group_scores[0].1
        - group_scores.get(1).map(|x| x.1).unwrap_or(group_scores[0].1);

    // 🌟 [TRACKING VETO] 구조 증거가 존재하는 문서는 물리적으로 택배 라벨일 수 없습니다.
    //    기존에는 parcel 이 이기면 codes.len()==1 분기로 TRACKING 이 무검증 확정되었습니다.
    //    여기서 그룹 단계에 veto 를 두면 그 무검증 경로 자체가 도달 불가가 됩니다.
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

    // ── 뎁스 2 : 후보 코드 집합 ──
    //  🌟 group_margin 을 '로그용 미사용 변수' 로 두지 않고 실제 판정에 씁니다.
    //     증거(surprisal > 0)가 있는 모든 그룹의 코드를 합집합으로 두어,
    //     1위와 2위가 사실상 동률일 때 그룹으로 잘못 좁히는 사고를 막습니다.
    let mut codes: Vec<&str> = GROUP_CODES.iter()
        .find(|(g, _)| *g == best_group)
        .map(|(_, c)| c.to_vec())
        .unwrap_or_else(|| vec!["Unknown"]);
    for (g, s) in group_scores.iter() {
        if g == &best_group { continue; }
        if *s <= 0.0 { continue; }
        if g == "parcel" && has_trade_marker { continue; }
        if let Some((_, extra)) = GROUP_CODES.iter().find(|(gn, _)| gn == g) {
            for c in extra.iter() {
                if !codes.iter().any(|x| x == c) { codes.push(c); }
            }
        }
    }
    emit_term(&format!("  🎯 [TRADE CODE CANDIDATES] {:?}", codes));

    // ── 코드 구 뱅크 SURPRISAL (편견 = 경쟁 코드의 앵커 구) ──
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

    let mut code_best: std::collections::HashMap<String, f32> = std::collections::HashMap::new();
    for le in line_embs.iter() {
        if le.iter().all(|&v| v == 0.0) { continue; }
        let (fs, _) = crate::utils::ai_utils::surprisal_dual_scores(
            le, &c_bias_bank, &c_prej_bank, &empty_names, &empty_banks, &empty_skip,
        );
        for s in fs {
            let e = code_best.entry(s.key.clone()).or_insert(f32::MIN);
            if s.surprisal > *e { *e = s.surprisal; }
        }
    }

    let mut code_scores: Vec<(String, f32)> = code_best.into_iter().collect();
    code_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if code_scores.is_empty() {
        code_scores.push((codes[0].to_string(), 0.0));
    }
    for (c, s) in code_scores.iter() {
        emit_term(&format!("    📐 [TRADE CODE] {} | Surprisal: {:+.4}", c, s));
    }

    // 🌟 [FINAL TRACKING VETO] 코드 단계에서도 한 번 더 막습니다.
    //    합집합 확장으로 TRACKING 이 후보에 들어온 경우를 방어합니다.
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

    // ── 뎁스 3 : 마진 부족 시에만 LLM 폴백 (그룹 내 코드만 제시) ──
    //    마진 기준은 절대 임계치가 아니라 '2순위와 사실상 동률인가' 라는 부호 판정입니다.
    //    코사인 공간에서 0.01 미만은 노이즈 수준이므로 구분 불가로 간주합니다.
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
        // 🌟 [PAGE TYPE PROMPT 통합] 하드코딩 대신 개선된 page_type_prompt("shipping") 을 사용합니다.
        //    코사인 점수를 후보 목록에 동봉하여 벡터 근거를 함께 전달합니다.
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
            // 🌟 [KV SESSION PER PAGE] 페이지마다 KV 스냅샷 키를 분리합니다.
            //    분리하지 않으면 2페이지가 1페이지의 KV 캐시를 재사용해
            //    1페이지의 문맥으로 2페이지를 분류하게 됩니다.
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

    // =====================================================================
    // 🌟 [TRADING STEP B v2] PLINKO 선행 + 미확정 카테고리만 LLM
    // ---------------------------------------------------------------------
    //  ── 무엇이 바뀌었나 ──
    //   v1 은 light_pug 전체를 System 에 넣고 카테고리 8개에 LLM 을 8번 호출했습니다.
    //   bias.json 의 무역 bias/prejudice 뱅크(bias_schema.rs 의 27종 분기)를
    //   단 한 번도 쓰지 않았고, 형식 게이트도 없어
    //   "총 중량 | 1,250" 셀이 amount 로 들어가도 막을 방법이 없었습니다.
    //
    //  ── v2 구조 (commerce 상세 경로와 동일) ──
    //   B-1  구조적 라벨-값 페어 수집   (collect_detail_label_value_pairs)
    //   B-2  라벨 뱅크 / 편견 뱅크 구축 (label_phrase_bank / prejudice_phrase_bank)
    //   B-3  형식 게이트                (detect_field_format / value_matches_format)
    //   B-4  이중 센터링 + 배타 배정    (double_center_matrix / exclusive_assign_by_score)
    //   B-5  확정된 필드는 LLM 없이 주입
    //   B-6  미확정 카테고리만 LLM 호출
    // =====================================================================
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

    // ── B-0 : 정제된 PUG로 컨텍스트 생성 (원문 HTML 직접 사용 금지) ──
    //    ── 왜 ListMode 인가 ──
    //     DetailMode 는 모든 속성(id/class/style/href/onclick...)을 그대로 남기므로
    //     토큰의 대부분이 사이트별 잡음으로 채워지고,
    //     0.6B 모델이 그것을 '의미'로 오인해 환각을 생성합니다.
    //     ListMode 는:
    //       · id / class / style 을 제거
    //       · input[value], selected option 텍스트는 보존
    //       · href 등 필수 이동 속성만 유지
    //     하여 페어 추출에 필요한 값은 살리면서 속성 노이즈를 제거합니다.
    //     이는 light_pug(NoAttributesMode)과 동일한 정제 철학이며,
    //     원문 HTML을 LLM 컨텍스트로 직접 사용하지 않는 원칙을 관철합니다.
    let content_pug = {
        let full_pug =
            parsing::convert_to_clean_pug(&clean_html_content, PugMode::ListMode, Some(&url));
        model
            .truncate_pug_context(&full_pug, true, 2000, None)
            .await
    };
    let pug_lines: Vec<String> = content_pug.lines().map(|s| s.to_string()).collect();
    let pug_lines_ref: Vec<&str> = pug_lines.iter().map(|s| s.as_str()).collect();

    // ── B-1 : 구조적 라벨-값 페어 ──
    let detail_pairs = crate::utils::ai_utils::collect_detail_label_value_pairs(&pug_lines_ref);
    emit_term(&format!("  🧷 [TRADING PAIR] 구조적 라벨-값 페어 {}개 확보", detail_pairs.len()));
    for p in &detail_pairs {
        emit_term(&format!(
            "    Line {} | Section: '{}' | Label: '{}' | Value: '{}'",
            p.primary_line + 1, p.section, p.label, p.value
        ));
    }

    // ── B-2 : 스키마 필드 + 라벨/편견 뱅크 ──
    //    bias_schema.rs 의 무역 분기(27종)가 이미 40여 필드를 정의하고 있습니다.
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

    // 🌟 [SELF-POISON GUARD] commerce 와 동일하게 자기 자신을 설명하는 편견 구를 박탈합니다.
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

    // ── B-3 : 페어 라벨 임베딩 + 형식 게이트 ──
    let mut unique_labels: Vec<String> = Vec::new();
    let mut unique_leaf: Vec<String> = Vec::new();
    let mut unique_section: Vec<String> = Vec::new();
    let mut label_count: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for p in &detail_pairs { *label_count.entry(p.label.clone()).or_insert(0) += 1; }

    let mut pair_phrases: Vec<String> = Vec::with_capacity(detail_pairs.len());
    for p in &detail_pairs {
        let dup = label_count.get(&p.label).copied().unwrap_or(0) > 1;
        if dup && !p.section.trim().is_empty() {
            pair_phrases.push(format!("{} {}", p.section.trim(), p.label));
        } else {
            pair_phrases.push(p.label.clone());
        }
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

        // 각 유일 라벨의 대표값 / 병합값 사전 계산
        let mut phrase_single: Vec<String> = vec![String::new(); unique_labels.len()];
        let mut phrase_multi: Vec<String> = vec![String::new(); unique_labels.len()];
        let mut phrase_line: Vec<usize> = vec![0usize; unique_labels.len()];
        for (pi, ph) in pair_phrases.iter().enumerate() {
            let h = match unique_labels.iter().position(|u| u == ph) { Some(v) => v, None => continue };
            let p = &detail_pairs[pi];
            if phrase_single[h].is_empty() && !p.value.trim().is_empty() {
                phrase_single[h] = p.value.clone();
                phrase_line[h] = p.primary_line;
            }
            let av = p.value_all.trim();
            if !av.is_empty() && !phrase_multi[h].contains(av) {
                if phrase_multi[h].is_empty() {
                    phrase_multi[h] = av.to_string();
                } else {
                    phrase_multi[h].push(' ');
                    phrase_multi[h].push_str(av);
                }
            }
        }

        // 행렬 구축 (형식 게이트 + 편견 게이트를 배정 '전'에 적용)
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
                if prej >= own {
                    emit_term(&format!("    🚫 [TRADING PREJUDICE GATE] '{}' → '{}' | Label: {:.4} <= Prej: {:.4}",
                        unique_labels[h], t_field_names[f], own, prej));
                    continue;
                }

                let pair_val = if f_multi { &phrase_multi[h] } else { &phrase_single[h] };
                if f_strict {
                    if pair_val.trim().is_empty()
                        || !crate::utils::ai_utils::value_matches_format(f_fmt, pair_val) {
                        emit_term(&format!("    🚫 [TRADING VALUE FORMAT GATE] '{}' → '{}' ({:?}) | 값 \"{}\" 형식 불일치",
                            unique_labels[h], t_field_names[f], f_fmt, pair_val));
                        continue;
                    }
                }
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

        // ── B-4 : 섹션 대비 항 + 배타 배정 ──
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

        // ── B-5 : 확정된 필드를 카테고리 슬롯에 직접 주입 ──
        //    🌟 [SHARED MAPPING] 필드 → 카테고리 매핑은 logic.rs 가 소유합니다.
        //    비전 히트맵(카테고리 단위 크롭)이 같은 매핑을 써야
        //    '크롭한 영역' 과 '추출 프롬프트가 요구하는 필드' 가 어긋나지 않습니다.
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

    // ── B-6 : PLINKO 로 확정되지 못한 카테고리만 LLM 호출 ──
    model.secure_vram_relay(crate::model::ModelSize::Qwen3_5, None, Some(cancellation_token.clone()), false, None).await?;

    for cat in &categories {
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        let schema_prompt = crate::parsing::get_trade_category_schema(cat, &doc_type);

        if schema_prompt.contains("SCHEMA:\n{}") || schema_prompt.contains("SCHEMA:\n[ {} ]") {
            emit_term(&format!("[TRADING STEP B] Category '{}' has no fields for {}. Skipping.", cat.to_uppercase(), doc_type));
            continue;
        }

        // 🌟 [PLINKO SKIP] 이 카테고리의 필드가 전부 PLINKO 로 확정되었으면 LLM 을 부르지 않습니다.
        //    (line_items / containers 는 배열이라 개수를 알 수 없으므로 항상 LLM 을 탑니다)
        if *cat != "items" && *cat != "containers" {
            let filled = final_data_map.get(*cat)
                .and_then(|v| v.as_object())
                .map(|o| o.iter().filter(|(k, _)| *k != "doc_type").count())
                .unwrap_or(0);
            // 스키마 필드 수를 프롬프트에서 세어 비교합니다. (라인당 필드 1개)
            let schema_field_count = schema_prompt.lines()
                .filter(|l| l.trim_start().starts_with('"'))
                .count();
            if schema_field_count > 0 && filled >= schema_field_count {
                emit_term(&format!("  ⚡ [TRADING LLM SKIP] Category '{}' 는 PLINKO 가 {}/{} 필드를 전부 확정하여 LLM 호출을 생략합니다.",
                    cat.to_uppercase(), filled, schema_field_count));
                continue;
            }
        }

        // 🌟 [ALREADY CLAIMED] PLINKO 가 이미 가져간 값을 LLM 이 다시 반환하지 못하게 막습니다.
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
            // 🌟 [PUG CONTEXT] 원문 HTML이 아닌, 정제된 ListMode PUG를 컨텍스트로 사용합니다.
            //    content_pug는 이미 pre_clean_html → convert_to_clean_pug(ListMode) →
            //    truncate_pug_context 파이프라인을 거친 결과입니다.
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
                // 🌟 [KV SESSION PER PAGE] 페이지 × 카테고리 단위로 KV 키를 분리합니다.
                Some(format!("{}_{}_{}", task.id, page_label, cat)),
                None, None, None
            ).await?;
            let mut tile_json = crate::parsing::parse_json_from_llm(&res);

            // 🌟 [PLINKO PROTECT] PLINKO 가 확정한 필드는 LLM 결과로 덮어쓰지 않습니다.
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

    // 🌟 [PAGE RESULT COLLECT] 이 페이지의 추론 결과를 보관하고 다음 페이지로 넘어갑니다.
    emit_term(&format!(
        "[TRADING PAGE {}/{}] ✅ 페이지 추출 완료 (doc_type='{}', lang='{}')",
        page_idx + 1, total_pages, doc_type, doc_lang
    ));
    page_results.push((doc_type.clone(), doc_lang.clone(), final_data_map));
    }
    // 🌟 [PAGE LOOP END] 페이지 단위 STEP A / STEP B 종료

    // 모델 해제 후 임베딩 준비 (페이지마다 파기하면 Qwen3.5 를 매 페이지 재로딩하므로
    // 전 페이지 추출이 끝난 뒤 딱 한 번만 수행합니다)
    model.deep_purge_resources().await;
    crate::utils::resources::wait_for_resources_settled(1200, 800, Some(cancellation_token), model.device_config.gpu_id as u32).await?;

    if page_results.is_empty() {
        return Err(anyhow::anyhow!("Trading extraction produced no result from any page."));
    }

    // =====================================================================
    // 🌟 [PAGE MERGE] 페이지별 추론 결과를 doc_type 기준으로 합칩니다.
    // ---------------------------------------------------------------------
    //  · 5장짜리 B/L      → doc_type 이 전부 BL → 1건으로 병합
    //  · CI+PL+BL 묶음    → doc_type 3종 → 3건으로 분리 저장
    //    (세 서식을 한 아이템에 뭉개면 doc_number 도 amount 도 서로 덮어씁니다)
    // =====================================================================
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

    // 🌟 [DOC LOOP OPEN] 병합된 문서마다 STEP C ~ STEP F 를 독립 수행합니다.
    for doc_type in merged_order.into_iter() {
    let (doc_lang, merged_map, merged_page_count) = match merged_docs.remove(&doc_type) {
        Some(v) => v,
        None => continue,
    };
    emit_term(&format!(
        "\n[TRADING DOC] ▶ doc_type='{}' (페이지 {}장 병합) 저장 파이프라인 시작",
        doc_type, merged_page_count
    ));

    let mut extracted_data = Value::Object(merged_map);

    // =====================================================================
    // 🌟 [TRADING STEP C v2] 루트 평탄화 + 자연어 변환 + 임베딩 텍스트 생성
    // ---------------------------------------------------------------------
    //  ── 무엇이 빠져 있었나 ──
    //   v1 은 중첩 구조({ header:{...}, logistics:{...} })를 그대로 저장했습니다.
    //   그런데 Dexie 인덱스는 'data.vessel' / 'data.pol' 같은 1뎁스 경로만 봅니다.
    //   extract_from_image 는 TRADING FLATTEN v3 로 이 문제를 이미 해결했지만
    //   HTML 경로에는 그 블록이 없어, 같은 문서라도 입력 경로에 따라
    //   검색 가능 여부가 달라지는 상태였습니다.
    //
    //  ── 규칙 ──
    //   중첩 그룹의 잎을 전부 루트로 끌어올리고,
    //   이름은 bias.json 의 search_bridge.path_alias 로 canonical 화합니다.
    //   (build_dexie_plan 의 normalize_path 와 같은 이름 공간을 씁니다)
    // =====================================================================
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

        // ── 배열 축 : 첫 원소만 대표 축으로 승격 ──
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

        // 🌟 [REFERENCE PROMOTION] 레거시 'reference_number' 에 담긴 값을
        //    접두어 구조로 판정해 올바른 참조 축으로 승격시킵니다.
        //    ── 왜 필요한가 ──
        //     기존 스키마는 참조 축이 reference_number 하나뿐이었고, relay 는
        //     reference_invoice / reference_lc / reference_booking 만 읽었습니다.
        //     그래서 로그의 BL 이 reference_number="CI-2026-08001" 을 손에 쥐고도
        //     relay 루프에서 continue 로 빠져나가 허브 키가 소실되었습니다.
        //    ── 판정 근거 ──
        //     문서번호 접두어는 어휘가 아니라 '서식 코드' 입니다.
        //     trade_reference_field_of 가 이미 그 코드 사전을 갖고 있으므로
        //     여기서는 접두어 토큰만 잘라 그 사전에 물어봅니다.
        {
            let legacy = extracted_data.get("reference_number")
                .and_then(|v| v.as_str())
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty() && s != "N/A");

            if let Some(val) = legacy {
                // 'CI-2026-08001' → 'CI' / 'HBL-55432219-01' → 'HBL'
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

        // 🌟 [SELF-REFERENCE GUARD] 자기 자신을 가리키는 참조 축은 릴레이 대상이 아닙니다.
        //    (예: BL 문서의 reference_bl 에 자기 doc_number 가 들어온 경우)
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

        // 🌟 [NORMALIZE] commerce 의 normalize_data 에 해당하는 단계가 trading 에는
        //    아예 없었습니다. 정규화 없이 저장하면 amount 가 "1,250.00 USD" 라는
        //    문자열이라 update_team_base_metrics 의 min/max/avg 축이 통째로 죽습니다.
        //    자연어 변환 '이전' 에 수행해야 text 컬럼에도 정규화된 값이 실립니다.
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

    // =====================================================================
    // 🌟 [TRADING STEP D] 저장
    // ---------------------------------------------------------------------
    // bcc 규칙: commerce 는 hash("{page_type}{cc}") 이지만,
    // trading 은 hash("{doc_type}{cc}") 를 사용합니다.
    // 이렇게 해야 같은 cc 안에서 BL / CI / PL 이 각각 다른 bcc 로 분리되어
    // 프론트엔드 TYPE_SETS.shipping 필터에서 서식별로 조회할 수 있습니다.
    // =====================================================================
    let store = {
        let store_guard = store_mutex.lock().await;
        store_guard.as_ref().ok_or_else(|| anyhow::anyhow!("Store not initialized"))?.clone()
    };

    // 🌟 [DOC NUMBER RESOLVE v2]
    //  ── 무엇이 문제였나 ──
    //   기존 폴백은 task.id 였습니다. task_id 는 스캔마다 새로 생기므로
    //   같은 문서를 다시 스캔해도 index / id / ref 가 전부 달라져
    //   upsert 가 아니라 신규 행이 무한히 쌓였습니다.
    //   (로그 실측: "Its doc number is task_1787731795587")
    //  ── v2 ──
    //   ① 루트 → header 순으로 doc_number 를 찾습니다. (STEP C 가 루트로 승격시켰습니다)
    //   ② 그래도 없으면 '이 문서의 내용' 에서 결정론 ID 를 만듭니다.
    //      같은 문서를 다시 스캔하면 같은 텍스트 → 같은 digest → 같은 id 가 되어
    //      중복이 아니라 갱신으로 처리됩니다.
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

    // 🌟 [ENTITY KEY v5] index / id 를 단일 계약 함수로만 만듭니다.
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

    // 🌟 trading bcc: doc_type 기반 (commerce 의 page_type 기반과 같은 역할)
    let cc_val = task.cc.clone();
    let bcc = entity_bcc(&doc_type, &cc_val);
    let ref_val = task.r#ref.clone();

    // 🌟 v4 : items 단일 저장.
    save_item(&store, "items", &hashed_item_id, &doc_type, extracted_data.clone(), Some(item_vector.clone()),
        &task.from, &team_id, &task.cc, &bcc, &ref_val, Some(&item_digest)).await;

    // =====================================================================
    // 🌟 [TRADING STEP E v2] Index 기반 양방향 Relay + Draft 생성
    // ---------------------------------------------------------------------
    //  ── v1 의 결함 3가지 ──
    //   ① trading_relay_field 가 (from, to) 에 대해 '단 하나의 필드'만 돌려주어
    //      CI 쪽에서는 존재하지 않는 CI.reference_invoice 를 읽어 항상 0건이었습니다.
    //   ② 참조를 문자열 doc_number 로만 저장해 표기 흔들림(대소문자/하이픈)에 어긋났고,
    //      reference_bl / reference_ci 는 Dexie 인덱스가 없어 풀스캔이었습니다.
    //   ③ 상대 문서가 아직 없으면 그냥 Skip 해서, 나중에 그 문서가 들어와도
    //      먼저 들어온 문서와 절대 연결되지 않았습니다.
    //      (commerce 는 draft 를 만들어 두고 나중에 채웁니다)
    //
    //  ── v2 구조 (commerce order↔tracking 과 동일) ──
    //   내 index  = crc32(hash_id(doc_type   + team_id + 정규화 doc_number))
    //   상대 index = crc32(hash_id(foreign_t + team_id + 정규화 참조번호))
    //   내 문서에  data.rel_{foreign}  = 상대 index
    //   상대 문서에 data.rel_{mine}    = 내 index
    //   상대가 없으면 draft 를 만들어 rel_ 축을 미리 채웁니다.
    // =====================================================================
    // =====================================================================
    // 🌟 [TRADING STEP E v3 / DIRECTION-FIXED]
    // ---------------------------------------------------------------------
    //  ── v2 의 결함 3가지 (로그 실측) ──
    //   ① index 방향 역전
    //      trading_relay_pair("CI","BL") 이 ("doc_number","reference_invoice") 였습니다.
    //      그래서 clean_ref 에 'CI 자신의 doc_number' 가 들어가
    //        crc32(hash("BL" + team + CI의 doc_number))
    //      를 만들었고, BL 의 실제 index
    //        crc32(hash("BL" + team + BL의 doc_number))
    //      와 구조적으로 절대 일치할 수 없었습니다.
    //      (로그: CI.rel_bl = 4100281351 (from doc_number='ta5k1787731795587'))
    //   ② draft 의 foreign_field 에 '내 doc_number' 를 주입
    //      그 필드로 다시 폴백 조회를 하니, 방금 만든 BL draft 를
    //      다음 루프(CI→PL)가 히트시켜 같은 문서를 PL 로 덮어썼습니다.
    //      (로그: 0x2febc09d... 가 Type: BL → Type: PL 로 3줄 만에 변조)
    //   ③ rel_* 를 숫자로 저장
    //      update_team_base_metrics 가 crc32 해시를 수치 통계 축으로 집계했습니다.
    //      (로그: "rel_bl": {"max": 4100281351.0, "min": 3029041598.0})
    //
    //  ── v3 계약 ──
    //   mine_field    : 내 data 에서 '상대의 doc_number' 가 들어 있는 필드
    //   foreign_field : 상대 data 에서 '내 doc_number' 가 들어 있는 필드
    //   rel_*         : 문자열로 저장 (통계 축 오염 차단)
    //   타입 가드     : 조회로 찾은 문서의 type 이 다르면 절대 덮어쓰지 않습니다.
    // =====================================================================
    let relay_targets = crate::logic::related_trading(&doc_type);
    let mut relay_linked = 0usize;
    let mut relay_drafted = 0usize;
    let mut relay_skipped: Vec<String> = Vec::new();

    for foreign_type in relay_targets {
        if cancellation_token.load(Ordering::Relaxed) { return Err(anyhow::anyhow!("Task cancelled")); }

        let (mine_field, foreign_field) = match crate::logic::trading_relay_pair(&doc_type, foreign_type) {
            Some(p) => p,
            None => continue,
        };

        // ── ① 내 문서에서 '상대의 doc_number' 를 읽습니다 ──
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

        // 🌟 자기 자신을 가리키면 릴레이가 성립하지 않습니다.
        if clean_ref == clean_no {
            emit_term(&format!(
                "  🧹 [RELAY SELF-LOOP] {}.{} 가 자기 문서번호({})와 같아 릴레이를 건너뜁니다.",
                doc_type, mine_field, ref_display
            ));
            continue;
        }

        // ── ② 상대 index 를 결정론으로 재현합니다 ──
        //    상대 문서가 저장될 때 쓰는 식과 '완전히 같은 함수' 를 씁니다.
        //    문자열 조립을 손으로 반복하지 않으므로 인자 순서가 어긋날 여지가 없습니다.
        let foreign_index = entity_index(foreign_type, &team_id, &ref_display);
        let mine_col = crate::logic::trading_index_column(&doc_type);
        let foreign_col = crate::logic::trading_index_column(foreign_type);

        // 🌟 rel_* 는 문자열로 저장합니다. (수치 통계 축 오염 차단)
        extracted_data.as_object_mut().unwrap()
            .insert(foreign_col.clone(), json!(foreign_index));
        emit_term(&format!(
            "  🔑 [TRADING INDEX] {}.{} = {} (근거 {}='{}' → 정규화 '{}')",
            doc_type, foreign_col, foreign_index, mine_field, ref_display, clean_ref
        ));

        // ── ③ 상대 문서 조회 (타입 필터 포함) ──
        //    commerce 릴레이가 인덱스 값으로 정확히 상대를 찾듯이,
        //    trading 릴레이도 '상대의 타입 + 상대의 참조 필드' 로 조회해야 합니다.
        //    find_item_by_property 는 타입 컬럼을 필터링하지 않으므로,
        //    get_all_items 의 SQL 필터로 타입 + 참조 필드를 동시 조건으로 걸어야 합니다.
        //
        //    조회 우선순위:
        //      1순위: 상대의 인덱스 값 (타입별 고유) + 타입 필터
        //      2순위: 상대가 나를 가리키는 참조 필드 + 타입 필터
        //      3순위: 상대의 doc_number + 타입 필터 (최후 폴백)
        let mut hit: Option<(String, Value)> = None;
        // ── 1순위: 상대의 data.index 로 역참조 (KEY-SCOPED) ──
        //    commerce 의 Relay("tracking","order") 가
        //    data.order === order.data.index 로 O(log n) 조회하는 것과 동일한 원리.
        //    find_item_by_property 가 이미 "property":"값" KEY-SCOPED prefilter 를 사용하므로
        //    오탐 없이 1회 조회로 확정합니다.
        //
        //    ⚠️ 추가 개선: 타입 필터를 함께 걸어 다른 서식의 같은 값 충돌을 방지합니다.
        {
            let idx_filter = format!(
                "type = '{}' AND data LIKE '%\"index\":{}%'",
                foreign_type,
                foreign_index
            );
            // 🌟 [INDEX PARITY] store.rs 의 canonicalize_data 가 index 를 Integer 로 확정하므로
            //    "index":1234567890 형태로 매칭합니다. (문자열 "1234567890" 아님)
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
        // ── 2순위: 상대의 참조 필드로 역참조 (KEY-SCOPED via find_item_by_property) ──
        //    find_item_by_property 는 내부에서
        //    data LIKE '%"reference_invoice":"CI-2026-08001"%' 형태로
        //    키까지 포함한 needle 을 사용하므로 오탐이 없습니다.
        if hit.is_none() {
            if let Ok(Some((foreign_id, foreign_data))) = store.find_item_by_property("items", foreign_field, &json!(doc_number)).await {
                // 🌟 [TYPE GUARD] 같은 참조 값을 다른 서식이 갖고 있을 수 있으므로 타입을 검증합니다.
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
        // ── 3순위: 상대의 doc_number 로 역참조 (KEY-SCOPED via find_item_by_property) ──
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

        // 🌟 [TYPE GUARD v2] 모든 조회 경로에 타입 필터가 포함되었지만,
        //    이중 방어막으로 한 번 더 검증합니다.
        //    특히 3순위 폴백에서 doc_number가 우연히 일치하는 경우를 차단합니다.
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
                    // 상대 문서에 내 index 를 꽂습니다. (양방향, 숫자)
                    // 🌟 [TYPE CONSISTENCY] json!(index_val.to_string()) 은
                    //    canonicalize_data 가 rel_* 를 Numeric 으로 확정하기 전에
                    //    문자열 "12345" 로 들어갑니다. 결국 숫자로 변환되긴 하지만,
                    //    같은 함수 안에서 내 문서 쪽은 json!(foreign_index) 로
                    //    숫자로 넣고 있어 일관성이 없습니다.
                    //    양쪽 모두 숫자로 통일합니다.
                    o.insert(mine_col.clone(), json!(index_val));
                    // 🌟 문자열 참조는 '나를 가리키는 필드' 에 '내 doc_number' 를 넣습니다.
                    //    v2 는 여기에 clean_ref(= 상대 번호)를 넣어 방향이 뒤집혀 있었습니다.
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
                    &task.from, &team_id, &task.cc, &foreign_bcc, &ref_val, None).await;
                relay_linked += 1;
                emit_term(&format!(
                    "  ✅ [TRADING RELAY] {} '{}' 에 {}='{}' / {}={} 역주입 완료.",
                    foreign_type, foreign_id, foreign_field, doc_number, mine_col, index_val
                ));
            },
            None => {
                // 🌟 [DRAFT v3] 상대 문서가 아직 없으면 미리 만들어 둡니다.
                //    상대가 실제로 들어오면 같은 index 를 갖게 되어 자동으로 이어집니다.
                //
                //    ── v3 변경 사항 ──
                //    1. doc_number 에 '상대의 고유 문서번호'가 아닌 빈 문자열을 넣습니다.
                //       (상대 문서는 자기 자신의 문서번호를 갖습니다)
                //    2. '나를 가리키는 참조 필드'에 '내 문서번호'를 넣습니다.
                //       (예: PL draft의 reference_invoice = CI의 doc_number)
                //    3. index 는 '상대 타입 + 참조값'으로 계산합니다.
                //       (상대가 실제로 들어왔을 때 같은 index가 됩니다)
                let draft_id = entity_id(&team_id, foreign_index);
                let mut draft_data = json!({});
                if let Some(obj) = draft_data.as_object_mut() {
                    obj.insert("id".to_string(), json!(draft_id.clone()));
                    obj.insert("type".to_string(), json!(foreign_type));
                    obj.insert("index".to_string(), json!(foreign_index));
                    // 🌟 [v3 FIX] doc_number 에 '상대의 고유 문서번호'가 아닌 빈 문자열을 넣습니다.
                    //    상대 문서는 자기 자신의 문서번호를 갖습니다.
                    //    기존 코드는 여기에 '참조값(=내 문서번호)'을 넣어서,
                    //    자기 자신을 찾는 원인이 되었습니다.
                    obj.insert("doc_number".to_string(), json!(""));
                    obj.insert("no".to_string(), json!(""));
                    // 🌟 나를 가리키는 필드에는 '내 문서번호'를 넣습니다.
                    //    예: PL 문서의 reference_invoice = "CI-2026-08001"
                    obj.insert(foreign_field.to_string(), json!(doc_number.clone()));
                    // 🌟 [INDEX PARITY] 내 인덱스도 숫자로 역참조.
                    //    commerce 의 order.data.tracking = tracking.data.index 패턴.
                    //    기존 .to_string() 은 프론트엔드 Dexie 인덱스 매칭에서
                    //    타입 불일치를 일으켰습니다.
                    obj.insert(mine_col.clone(), json!(index_val));
                    // 🌟 [INDEX PARITY] 상대의 인덱스도 숫자로.
                    //    기존 .to_string() → 숫자로 변경.
                    obj.insert(foreign_col.clone(), json!(foreign_index));
                    obj.insert("updated_at".to_string(), json!(0));
                    obj.insert("mode".to_string(), json!("shipping"));
                    // 🌟 text 에도 참조 관계만 기록합니다.
                    obj.insert("text".to_string(), json!(format!("{} draft (ref: {} = {})", foreign_type, foreign_field, doc_number)));
                }
                let foreign_bcc = entity_bcc(foreign_type, &cc_val);
                save_item(&store, "items", &draft_id, foreign_type, draft_data, None,
                    &task.from, &team_id, &task.cc, &foreign_bcc, &ref_val, None).await;
                relay_drafted += 1;
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

    // 최종 저장 (relay 로 updated 필드가 추가된 경우)
    save_item(&store, "items", &hashed_item_id, &doc_type, extracted_data.clone(), Some(item_vector.clone()),
        &task.from, &team_id, &task.cc, &bcc, &ref_val, Some(&item_digest)).await;

    // =====================================================================
    // 🌟 [TRADING STEP E-2] 청크 인덱싱 (PHASE A~E)
    // ---------------------------------------------------------------------
    //  ── 왜 필요한가 ──
    //   lib.rs 의 STAGE-4 는 item_chunks 테이블을 코사인 검색합니다.
    //   commerce 는 상세/리스트 양쪽 모두 청크를 저장하는데,
    //   trading 은 이 단계가 없어 "선적항이 부산인 B/L" 같은 필드 레벨 질의가
    //   구조적으로 0건이었습니다. (아이템 벡터 1개만으로는 값이 희석됩니다)
    //
    //  ── index_item_chunks 재사용 ──
    //   이 함수는 이미 PHASE A(분할) ~ PHASE E(저장) 전 과정과
    //   음차 별칭 생성까지 포함하고 있으므로 그대로 호출합니다.
    //   bias_schema.rs 의 무역 분기가 doc_type 에 대응하는 필드를 돌려주므로
    //   뱅크가 비어 조기 종료되는 일도 없습니다.
    // =====================================================================
    {
        let chunk_count = index_item_chunks(
            &store,
            &model,
            &hashed_item_id,
            &doc_type,
            &doc_lang,
            &extracted_data,
            true,               // is_detail : 무역 서식은 항상 단일 문서 상세
            &task.cc,
            &bcc,
            &ref_val,
            "shipping",
            &url,
            cancellation_token,
            app_handle,
            &task.id,
            false,              // skip_transliteration: 무역 문서는 음차 필요하므로 false
        ).await.unwrap_or(0);

        emit_term(&format!(
            "  🧩 [TRADING CHUNK INDEX] item_id='{}' | 청크 {}건 인덱싱 완료 (doc_type='{}')",
            hashed_item_id, chunk_count, doc_type
        ));
    }

    // =====================================================================
    // 🌟 [TRADING STEP F] Metrics + 완료
    // =====================================================================
    // 🌟 [STATS DIFF v2] 신규/갱신을 구분하고, draft 였다가 완성된 경우를 감산합니다.
    //  ── v1 의 결함 ──
    //   무조건 e.1 += 1 이라, 같은 문서를 재스캔할 때마다 count 가 늘었습니다.
    //   (로그 실측: CI 문서가 실제로 1건인데 "CI": { "count": 2 })
    //   또 draft 로 만들어 둔 문서가 실제로 들어와도 draft 가 줄지 않았습니다.
    let mut stats_diff: std::collections::HashMap<String, (i64, i64, i64)> = std::collections::HashMap::new();
    {
        let prev = store.get_item_by_id("items", &hashed_item_id).await.ok().flatten();
        match prev {
            None => {
                let e = stats_diff.entry(doc_type.clone()).or_insert((0, 0, 0));
                e.1 += 1; // count
                e.2 += 1; // global
                emit_term(&format!("  📊 [STATS] doc_type='{}' 신규 문서로 집계합니다.", doc_type));
            },
            Some(existing) => {
                let was_draft = existing.updated_at_ts == 0;
                if was_draft {
                    let e = stats_diff.entry(doc_type.clone()).or_insert((0, 0, 0));
                    e.0 -= 1; // draft 감산
                    e.1 += 1; // count 증가
                    e.2 += 1;
                    emit_term(&format!("  📊 [STATS] doc_type='{}' draft → 완성 문서로 전환합니다.", doc_type));
                } else {
                    emit_term(&format!("  📊 [STATS] doc_type='{}' 기존 문서 갱신이므로 count 를 증가시키지 않습니다.", doc_type));
                }
            }
        }
    }

    // 🌟 [METRICS GUARD v2] commerce 의 update_team_base_metrics 호출부와 동일한
    //    '최소 계약' 을 강제하고, 통계에 들어가서는 안 되는 축을 제거합니다.
    //    · type / mode      : draft·count 분류 키
    //    · updated_at       : draft 판정 축
    //    · created_at       : 시간축(최초/최근) 집계 축
    //    · rel_* 제거       : crc32 해시값이므로 min/max 통계에 아무 의미가 없습니다.
    //      (로그 실측: "rel_bl": { "max": 4100281351.0, "min": 3029041598.0 })
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

    let _ = crate::utils::metrics::update_team_base_metrics(&store, &team_id, &task.cc, &metrics_input, stats_diff.clone()).await;
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
    // 🌟 [DOC LOOP END] 병합 문서 단위 STEP C ~ STEP F 종료

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
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use serde_json::{json, Value};

// =====================================================================
// 🌟 [RESEARCH MAP → REDUCE] 심층 리포트
// ---------------------------------------------------------------------
//  ── 예전 ──
//   검색 상위 3건만 문맥으로 받아 고정 단계 3개를 0.6B 로 차례로 생성했지만, 단계 답은 다음 프롬프트로
//   넘어가지 않아(진행 로그에 앞 200바이트만, 한글이면 글자 중간을 잘라 패닉) 최종 보고서는 3건만 보고 썼습니다.
//  ── 지금 ──
//   ① Map   : 저장된 문서 전체를 Rust 로 집계합니다 (유형 · 월별 · 상태 · 수치 합계 · 범주 상위값). 결정적이고 모델이 필요 없습니다.
//   ② Reduce: 집계표 + 검색 상위 문서로 생성 한 번. 숫자는 집계표가 정답이고 모델은 해석만 합니다.
//   생성 호출이 4회에서 1회로 줄고, 보고서가 3건이 아니라 전체 데이터를 근거로 합니다.
// =====================================================================

/// 합계를 내는 수치 칸 (data 안에서 이 이름으로 끝나는 칸).
const NUMERIC_FIELDS: &[&str] = &[
    "amount", "amount_tax", "freight_amount", "invoice_value", "sale_price", "supply_price", "shipping_fee",
    "discount", "quantity", "package_count", "weight", "weight_gross", "gross_weight_kg", "volume",
];
/// 상위값을 세는 범주 칸.
const CATEGORY_FIELDS: &[&str] = &["status", "carrier", "brand_name", "category", "currency", "doc_type", "incoterms"];
/// 집계표에 남기는 줄 수 상한 (프롬프트 길이를 묶습니다).
const TOP_TYPES: usize = 10;
const TOP_MONTHS: usize = 12;
const TOP_SUMS: usize = 12;
const TOP_VALUES: usize = 5;

/// 집계 입력 한 줄.
#[derive(Debug, Clone)]
pub struct ResearchRow {
    pub doc_type: String,
    /// epoch ms
    pub created_at: i64,
    /// 0 이면 draft (아직 확정 전)
    pub updated_at: i64,
    pub data: Value,
}

impl From<&crate::store::TradeDocument> for ResearchRow {
    fn from(doc: &crate::store::TradeDocument) -> Self {
        Self {
            doc_type: doc.r#type.clone(),
            created_at: doc.created_at_ts,
            updated_at: doc.updated_at_ts,
            data: serde_json::from_str(&doc.json_data).unwrap_or(Value::Null),
        }
    }
}

/// 수치 칸 합계.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FieldSum {
    pub sum: f64,
    pub count: usize,
}

/// 문서 전체 집계 (Map 결과).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ResearchDigest {
    pub documents: usize,
    pub drafts: usize,
    pub by_type: BTreeMap<String, usize>,
    /// 작성 월(YYYY-MM)별 문서 수
    pub by_month: BTreeMap<String, usize>,
    /// "유형.칸 경로" → 합계
    pub sums: BTreeMap<String, FieldSum>,
    /// "칸" → 값별 건수
    pub categories: BTreeMap<String, HashMap<String, usize>>,
}

fn numeric_value(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64().filter(|x| x.is_finite()),
        Value::String(s) => crate::utils::canonical::parse_number_run(s, false),
        _ => None,
    }
}

fn category_value(v: &Value) -> Option<String> {
    match v {
        Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn month_of(ms: i64) -> String {
    if ms <= 0 {
        return "unknown".to_string();
    }
    chrono::DateTime::from_timestamp_millis(ms)
        .map(|t| t.format("%Y-%m").to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

impl ResearchDigest {
    /// 행들을 집계합니다. 객체는 세 단계까지 내려가며 칸 이름(경로의 끝)으로 수치 · 범주를 가립니다.
    pub fn from_rows<'a>(rows: impl IntoIterator<Item = &'a ResearchRow>) -> Self {
        let mut d = Self::default();
        for row in rows {
            d.documents += 1;
            if row.updated_at == 0 {
                d.drafts += 1;
            }
            let doc_type = if row.doc_type.trim().is_empty() { "unknown" } else { row.doc_type.trim() };
            *d.by_type.entry(doc_type.to_string()).or_default() += 1;
            *d.by_month.entry(month_of(row.created_at)).or_default() += 1;
            d.visit(doc_type, "", &row.data, 0);
        }
        d
    }

    fn visit(&mut self, doc_type: &str, path: &str, v: &Value, depth: usize) {
        let Value::Object(map) = v else { return };
        for (key, value) in map {
            let child = if path.is_empty() { key.clone() } else { format!("{}.{}", path, key) };
            if value.is_object() {
                if depth < 2 {
                    self.visit(doc_type, &child, value, depth + 1);
                }
                continue;
            }
            if NUMERIC_FIELDS.contains(&key.as_str()) {
                if let Some(x) = numeric_value(value) {
                    let slot = self.sums.entry(format!("{}.{}", doc_type, child)).or_default();
                    slot.sum += x;
                    slot.count += 1;
                }
            } else if CATEGORY_FIELDS.contains(&key.as_str()) {
                if let Some(c) = category_value(value) {
                    *self.categories.entry(key.clone()).or_default().entry(c).or_default() += 1;
                }
            }
        }
    }

    /// 프롬프트에 넣을 집계표. 줄 수에 상한이 있어 문서가 많아도 길이가 묶입니다.
    pub fn render(&self) -> String {
        fn fmt_num(x: f64) -> String {
            if x.fract() == 0.0 && x.abs() < 1e15 { format!("{}", x as i64) } else { format!("{:.2}", x) }
        }
        let mut out = format!(
            "## Aggregate statistics over all {} stored documents (exact, computed by the app — do not recount)\n- drafts (not yet confirmed): {}\n",
            self.documents, self.drafts
        );
        let mut types: Vec<_> = self.by_type.iter().collect();
        types.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        let types: Vec<String> = types.iter().take(TOP_TYPES).map(|(t, n)| format!("{} {}", t, n)).collect();
        out.push_str(&format!("- documents by type: {}\n", types.join(", ")));
        let months: Vec<String> = self
            .by_month
            .iter()
            .rev()
            .take(TOP_MONTHS)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(|(m, n)| format!("{} {}", m, n))
            .collect();
        out.push_str(&format!("- documents created per month: {}\n", months.join(", ")));
        let mut sums: Vec<_> = self.sums.iter().collect();
        sums.sort_by(|a, b| b.1.count.cmp(&a.1.count).then(a.0.cmp(b.0)));
        for (field, s) in sums.into_iter().take(TOP_SUMS) {
            out.push_str(&format!(
                "- {}: total {} over {} documents (average {})\n",
                field,
                fmt_num(s.sum),
                s.count,
                fmt_num(s.sum / s.count.max(1) as f64)
            ));
        }
        for (field, values) in &self.categories {
            let mut top: Vec<_> = values.iter().collect();
            top.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            let top: Vec<String> = top.iter().take(TOP_VALUES).map(|(v, n)| format!("{} {}", v, n)).collect();
            out.push_str(&format!("- most frequent {}: {}\n", field, top.join(", ")));
        }
        out
    }
}

/// 문자 경계에서 자릅니다 (예전 `&s[..200]` 은 한글 중간에서 패닉했습니다).
pub fn clip_chars(s: &str, max_chars: usize) -> String {
    match s.char_indices().nth(max_chars) {
        Some((at, _)) => format!("{}…", &s[..at]),
        None => s.to_string(),
    }
}

impl crate::model::LogisModel {
    /// 🌟 [RESEARCH REDUCE] 집계표(`digest`)와 검색 상위 문서(`top_documents`)로 보고서를 한 번에 생성합니다.
    pub async fn run_deep_research(
        &self,
        query: String,
        digest: &ResearchDigest,
        top_documents: String,
        app_handle: &tauri::AppHandle,
        cancel_token: Option<Arc<AtomicBool>>,
    ) -> anyhow::Result<String> {
        let mut status_history = format!("### 🔍 Deep Research: '{}'\n\n", query);
        status_history.push_str(&format!(
            "✅ Aggregated {} stored documents ({} types) in Rust.\n\n**⏳ Writing the report...**\n",
            digest.documents,
            digest.by_type.len()
        ));
        crate::utils::logger::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

        let prompt = format!(
            "{}\n## Most relevant documents\n{}\n\n## Query\n{}\n\n\
             Write a concise trade and commerce intelligence report that answers the query. \
             Use the aggregate statistics as ground truth and quote their numbers; do not invent numbers. \
             Use the documents only as examples. Mention trends across months and notable concentrations.",
            digest.render(),
            if top_documents.trim().is_empty() { "(none)" } else { top_documents.as_str() },
            query
        );
        let report = self.run_inference_text(prompt, None, cancel_token, None, None).await?;

        status_history.push_str("### 📊 Final Research Report\n\n");
        status_history.push_str(&report);
        crate::utils::logger::log_task_progress(app_handle, "research", &json!({ "text": status_history }));
        println!("[DEEP-RESEARCH] 보고서 {}자 · 앞부분: {}", report.chars().count(), clip_chars(&report.replace('\n', " "), 200));
        Ok(report)
    }
}

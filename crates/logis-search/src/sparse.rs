//! Sparse lexical & neural-sparse inverted index with Q8 postings, tombstones,
//! structured logistics/commerce tokenization, and MaxScore upper-bound pruning
//! (Phase 4 – Part 2, Sparseup-fr style).

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

pub const SPARSE_MAGIC: [u8; 4] = *b"LGSP";
pub const SPARSE_VERSION: u32 = 1;

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantizedPosting {
    pub doc_id: u32,
    pub weight_q8: u8,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseSegmentHeader {
    pub magic: [u8; 4],
    pub version: u32,
    pub doc_count: u32,
    pub term_count: u32,
    pub scale_max: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DocMeta {
    ext_id: String,
    mode: String,
    cc: String,
    doc_type: String,
    doc_len: u32,
    tombstone: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TermList {
    max_w: f32,
    postings: Vec<QuantizedPosting>,
}

/// Deterministic 32-bit FNV-1a hash for lexical term IDs (high bit set to separate from neural token IDs `0..2^31`).
#[inline]
pub fn lexical_term_id(term: &str) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for b in term.as_bytes() {
        h ^= *b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h | 0x8000_0000
}

fn is_arabic_diacritic(c: char) -> bool {
    let u = c as u32;
    c == '\u{0640}' || (0x064B..=0x065F).contains(&u) || u == 0x0670
}

fn is_cjk_char(c: char) -> bool {
    let u = c as u32;
    (0x4E00..=0x9FFF).contains(&u)
        || (0x3400..=0x4DBF).contains(&u)
        || (0x3040..=0x30FF).contains(&u)
}

/// Extract structured tokens for commerce/trade/multilingual retrieval:
/// - Preserves hyphenated/dotted SKU, tracking numbers, and HS codes (`8471.30`, `SKU-99A`, `20260625-100159`)
/// - Normalizes comma-formatted amounts (`329,800원` -> `329800`) and Arabic/fullwidth digits
/// - Strips Arabic harakat/tatweel (`U+0640`)
/// - Generates CJK character bigrams so unspaced Japanese/Chinese queries match reliably.
pub fn extract_structured_tokens(text: &str) -> Vec<String> {
    let norm_digits = logis_schema::text::normalize_digits_ascii(text);
    // Collapse thousands separators between ASCII digits (`329,800원` -> `329800원`) before splitting on `,`.
    let raw_chars: Vec<char> = norm_digits.chars().filter(|&c| !is_arabic_diacritic(c)).collect();
    let mut cleaned = String::with_capacity(raw_chars.len());
    for (idx, &c) in raw_chars.iter().enumerate() {
        if c == ','
            && idx > 0
            && idx + 1 < raw_chars.len()
            && raw_chars[idx - 1].is_ascii_digit()
            && raw_chars[idx + 1].is_ascii_digit()
        {
            continue;
        }
        cleaned.push(c);
    }

    let mut out = Vec::new();
    for raw in cleaned.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                ',' | ';' | ':' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '<' | '>' | '?' | '!' | '،' | '؛' | '、' | '，' | '；'
            )
    }) {
        let tok = raw.trim_matches(|c: char| !c.is_alphanumeric() && c != '-' && c != '.' && c != '_' && c != '/');
        if tok.is_empty() {
            continue;
        }
        let low = tok.to_lowercase();
        if low.chars().count() >= 2 {
            out.push(low.clone());
        }
        // Split compound identifiers (`SKU-99A` -> `sku`, `99a`; `8471.30.00` -> `8471`, `847130`)
        if low.contains('-') || low.contains('.') || low.contains('/') || low.contains('_') {
            for part in low.split(|c| matches!(c, '-' | '.' | '/' | '_')) {
                if part.chars().count() >= 2 {
                    out.push(part.to_string());
                }
            }
            let compact: String = low.chars().filter(|c| c.is_alphanumeric()).collect();
            if compact.len() >= 4 && compact != low {
                out.push(compact);
            }
        }
        // Leading numeric stem (`329800원` / `20kg` -> `329800` / `20`, `kg`)
        let digits: String = low.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.len() >= 2 && digits.len() < low.len() {
            let suffix = &low[digits.len()..];
            if !suffix.chars().any(|c| c.is_ascii_digit()) {
                out.push(digits);
                if suffix.chars().count() >= 1 {
                    out.push(suffix.to_string());
                }
            }
        }
        // CJK character bigrams for unspaced Chinese/Japanese phrases
        let chars: Vec<char> = low.chars().collect();
        if chars.len() >= 2 && chars.iter().any(|&c| is_cjk_char(c)) {
            for w in chars.windows(2) {
                if is_cjk_char(w[0]) || is_cjk_char(w[1]) {
                    let mut bi = String::with_capacity(8);
                    bi.push(w[0]);
                    bi.push(w[1]);
                    out.push(bi);
                }
            }
        }
    }
    out
}

/// BM25 / structured lexical analyzer producing sparse `(term_id, tf_weight)` vectors.
pub struct Bm25Analyzer;

impl Bm25Analyzer {
    pub fn analyze_document(text: &str) -> (u32, Vec<(u32, f32)>) {
        let toks = extract_structured_tokens(text);
        let doc_len = toks.len() as u32;
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for t in toks {
            *counts.entry(lexical_term_id(&t)).or_insert(0) += 1;
        }
        let mut vec: Vec<(u32, f32)> = counts
            .into_iter()
            .map(|(tid, tf)| (tid, (1.0 + tf as f32).ln()))
            .collect();
        vec.sort_by(|a, b| b.1.total_cmp(&a.1));
        vec.truncate(256);
        (doc_len, vec)
    }

    pub fn analyze_query(query: &str) -> Vec<(u32, f32)> {
        let toks = extract_structured_tokens(query);
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for t in toks {
            *counts.entry(lexical_term_id(&t)).or_insert(0) += 1;
        }
        let mut vec: Vec<(u32, f32)> = counts
            .into_iter()
            .map(|(tid, tf)| (tid, 1.0 + (tf as f32).ln()))
            .collect();
        vec.sort_by(|a, b| b.1.total_cmp(&a.1));
        vec.truncate(64);
        vec
    }
}

/// Persistent Q8 sparse inverted index with tombstones and MaxScore upper-bound pruning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SparseIndex {
    docs: Vec<DocMeta>,
    ext_to_doc: HashMap<String, u32>,
    postings: BTreeMap<u32, TermList>,
    scale_max: f32,
}

impl SparseIndex {
    pub fn new() -> Self {
        Self {
            docs: Vec::new(),
            ext_to_doc: HashMap::new(),
            postings: BTreeMap::new(),
            scale_max: 8.0,
        }
    }

    pub fn active_doc_count(&self) -> usize {
        self.docs.iter().filter(|d| !d.tombstone).count()
    }

    pub fn term_count(&self) -> usize {
        self.postings.len()
    }

    fn quantize(&self, w: f32) -> u8 {
        let clamped = (w / self.scale_max.max(1e-3)).clamp(0.0, 1.0);
        (clamped * 255.0).round().clamp(1.0, 255.0) as u8
    }

    fn dequantize(&self, q: u8) -> f32 {
        (q as f32 / 255.0) * self.scale_max
    }

    pub fn delete(&mut self, ext_id: &str) {
        if let Some(old_id) = self.ext_to_doc.remove(ext_id) {
            if let Some(meta) = self.docs.get_mut(old_id as usize) {
                meta.tombstone = true;
            }
        }
    }

    pub fn upsert(
        &mut self,
        ext_id: &str,
        mode: &str,
        cc: &str,
        doc_type: &str,
        doc_len: u32,
        weights: &[(u32, f32)],
    ) {
        // Tombstone any prior version of this document.
        self.delete(ext_id);
        let doc_id = self.docs.len() as u32;
        self.docs.push(DocMeta {
            ext_id: ext_id.to_string(),
            mode: mode.to_string(),
            cc: cc.to_string(),
            doc_type: doc_type.to_string(),
            doc_len: doc_len.max(1),
            tombstone: false,
        });
        self.ext_to_doc.insert(ext_id.to_string(), doc_id);

        for &(tid, w) in weights {
            if !w.is_finite() || w <= 0.0 {
                continue;
            }
            let q = self.quantize(w);
            let dw = self.dequantize(q);
            let entry = self.postings.entry(tid).or_insert_with(|| TermList {
                max_w: 0.0,
                postings: Vec::new(),
            });
            if dw > entry.max_w {
                entry.max_w = dw;
            }
            entry.postings.push(QuantizedPosting { doc_id, weight_q8: q });
        }
    }

    pub fn upsert_text(&mut self, ext_id: &str, mode: &str, cc: &str, doc_type: &str, text: &str) {
        let (len, vec) = Bm25Analyzer::analyze_document(text);
        self.upsert(ext_id, mode, cc, doc_type, len, &vec);
    }

    /// Search with IDF weighting and MaxScore upper-bound pruning.
    /// Produces exact top-`limit` `(ext_id, score)` sorted descending.
    pub fn search(
        &self,
        query_weights: &[(u32, f32)],
        limit: usize,
        mode_filter: Option<&str>,
        cc_filter: Option<&str>,
        type_filter: Option<&[String]>,
    ) -> Vec<(String, f32)> {
        if query_weights.is_empty() || limit == 0 {
            return Vec::new();
        }
        let n_docs = self.active_doc_count().max(1) as f32;

        // Gather term posting lists with IDF * query_weight and term max contribution.
        let mut lists: Vec<(&TermList, f32, f32)> = Vec::new();
        for &(tid, qw) in query_weights {
            if !qw.is_finite() || qw <= 0.0 {
                continue;
            }
            if let Some(tl) = self.postings.get(&tid) {
                let df = tl.postings.len() as f32;
                let idf = ((n_docs + 1.0) / (df + 0.5)).ln().max(0.1);
                let eff_q = qw * idf;
                lists.push((tl, eff_q, eff_q * tl.max_w));
            }
        }
        if lists.is_empty() {
            return Vec::new();
        }
        // Sort terms by descending upper-bound contribution (MaxScore ordering).
        lists.sort_by(|a, b| b.2.total_cmp(&a.2));

        // Suffix sums of upper bounds for early termination.
        let mut suffix_ub = vec![0.0f32; lists.len() + 1];
        for i in (0..lists.len()).rev() {
            suffix_ub[i] = suffix_ub[i + 1] + lists[i].2;
        }

        let mut scores: HashMap<u32, f32> = HashMap::new();
        let mut threshold = 0.0f32;
        let mut scored_count = 0usize;

        for (ti, (tl, eff_q, _ub)) in lists.iter().enumerate() {
            let rem_ub = suffix_ub[ti];
            let can_prune_new = scores.len() >= limit && threshold > 0.0 && rem_ub <= threshold;

            for p in &tl.postings {
                let Some(meta) = self.docs.get(p.doc_id as usize) else {
                    continue;
                };
                if meta.tombstone {
                    continue;
                }
                if let Some(m) = mode_filter {
                    if !m.is_empty() && meta.mode != m {
                        continue;
                    }
                }
                if let Some(c) = cc_filter {
                    if !c.is_empty() && meta.cc != c {
                        continue;
                    }
                }
                if let Some(types) = type_filter {
                    if !types.is_empty() && !types.iter().any(|t| t == &meta.doc_type) {
                        continue;
                    }
                }
                if can_prune_new && !scores.contains_key(&p.doc_id) {
                    continue;
                }
                let contrib = eff_q * self.dequantize(p.weight_q8);
                *scores.entry(p.doc_id).or_insert(0.0) += contrib;
                scored_count += 1;
            }
            // Refresh kth threshold once we have enough candidates.
            if scores.len() >= limit && scored_count > limit * 2 {
                let mut vals: Vec<f32> = scores.values().copied().collect();
                vals.sort_by(|a, b| b.total_cmp(a));
                if let Some(&kth) = vals.get(limit - 1) {
                    threshold = kth;
                }
            }
        }

        let mut ranked: Vec<(String, f32)> = scores
            .into_iter()
            .filter_map(|(did, s)| {
                self.docs
                    .get(did as usize)
                    .filter(|m| !m.tombstone)
                    .map(|m| (m.ext_id.clone(), s))
            })
            .collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        ranked.truncate(limit);
        ranked
    }

    pub fn search_text(
        &self,
        query: &str,
        limit: usize,
        mode_filter: Option<&str>,
        cc_filter: Option<&str>,
        type_filter: Option<&[String]>,
    ) -> Vec<(String, f32)> {
        let qv = Bm25Analyzer::analyze_query(query);
        self.search(&qv, limit, mode_filter, cc_filter, type_filter)
    }

    /// Persist atomically to `dir/sparse_index.json` (`.part` + `sync_all` + rename).
    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let dst = dir.join("sparse_index.json");
        let tmp = dir.join(format!("sparse_index.json.part.{}", std::process::id()));
        let bytes = serde_json::to_vec(self)?;
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
        }
        if dst.exists() {
            let _ = std::fs::remove_file(&dst);
        }
        std::fs::rename(&tmp, &dst)?;
        Ok(())
    }

    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join("sparse_index.json");
        if !path.exists() {
            return Ok(Self::new());
        }
        let bytes = std::fs::read(&path)?;
        let idx: Self = serde_json::from_slice(&bytes)?;
        if idx.scale_max <= 0.0 {
            bail!("corrupt sparse index scale_max");
        }
        Ok(idx)
    }

    pub fn has_doc(&self, ext_id: &str) -> bool {
        self.ext_to_doc
            .get(ext_id)
            .and_then(|&d| self.docs.get(d as usize))
            .is_some_and(|m| !m.tombstone)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn structured_tokens_sku_hs_arabic_cjk() {
        let toks = extract_structured_tokens("SKU-99A HS 8471.30 329,800원 فاتورة رقم 20260625-100159 東京倉庫");
        assert!(toks.iter().any(|t| t == "sku-99a" || t == "sku99a"));
        assert!(toks.iter().any(|t| t == "8471.30" || t == "847130"));
        assert!(toks.iter().any(|t| t == "20260625-100159"));
        assert!(toks.iter().any(|t| t == "فاتورة"));
        assert!(toks.iter().any(|t| t == "東京"));
    }

    #[test]
    fn maxscore_and_tombstone_roundtrip() -> Result<()> {
        let mut idx = SparseIndex::new();
        idx.upsert_text("doc-1", "shipping", "kr", "BL", "Bill of Lading BL-2026-9901 HS 8471.30 Samsung OLED");
        idx.upsert_text("doc-2", "commerce", "kr", "order", "주문번호 20260625-100159 홍길동 329800원");
        idx.upsert_text("doc-3", "shipping", "jp", "CI", "Commercial Invoice 東京倉庫 8471.30");

        let r1 = idx.search_text("BL-2026-9901", 5, Some("shipping"), None, None);
        assert_eq!(r1.first().map(|x| x.0.as_str()), Some("doc-1"));

        let r2 = idx.search_text("東京倉庫", 5, Some("shipping"), None, None);
        assert_eq!(r2.first().map(|x| x.0.as_str()), Some("doc-3"));

        // Tombstone & save/load roundtrip
        idx.delete("doc-1");
        let tmp = std::env::temp_dir().join(format!("logis-sparse-test-{}", std::process::id()));
        idx.save(&tmp)?;
        let loaded = SparseIndex::open(&tmp)?;
        assert!(!loaded.has_doc("doc-1"));
        assert!(loaded.has_doc("doc-2"));
        let r3 = loaded.search_text("BL-2026-9901", 5, Some("shipping"), None, None);
        assert!(r3.is_empty());
        let _ = std::fs::remove_dir_all(&tmp);
        Ok(())
    }
}

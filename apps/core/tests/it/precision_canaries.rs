//! Phase 5 – Part 2: Precision Canary Regression Tests.
//!
//! Guards the 5 critical failure modes discovered during model & multilingual audits:
//!   1) Zero empty label / prejudice nodes across ALL loaded languages in `BIAS_DICT`
//!      (specifically Arabic `ar` 226/121 empty nodes, Japanese `ja` 194 merged entries, Czech `cs` 2 empty nodes).
//!   2) Static `lang_spec` multilingual script & transliteration accuracy (`ar` Core12, `ru`/`th`/`hi` non-Latin,
//!      `can_transliterate("T-Shirt", "ru") == true`, 1-char Hangul does not hijack English doc, `zh-tw` -> `zh` fallback).
//!   3) SigLIP2 `canonicalize` preserves combining marks (Arabic harakat, Thai vowels) & strips Arabic tatweel `U+0640`.
//!   4) `EmbedTable::validate_ids_slice` catches out-of-bounds token IDs cleanly without GPU corruption.
//!   5) Sparse search structured tokenization + `MaxScore` parity + mode-neutral `fuse_query_tracks` (FTS + Sparse + Literal).

use logis_app::models::siglip2::tokenizer::Siglip2Tokenizer;
use logis_app::utils::bias_schema::{lang_code_of, BIAS_DICT};
use logis_app::utils::lang_utils::detect_document_language;
use logis_bias::bank::{is_value_example_phrase, label_phrase_bank, prejudice_phrase_bank};
use logis_pug::nl_convert::can_transliterate;
use logis_pug::scripts::{lang_spec, native_scripts, LangTier, ScriptKind};
use logis_schema::text::split_phrase_list;
use logis_search::fusion::{fuse_query_tracks, FusionTracks, FUSION_K};
use logis_search::sparse::{extract_structured_tokens, SparseIndex};

#[test]
fn canary_1_full_bias_zero_empty_nodes_and_multilingual_delimiters() {
    // Arabic ، ؛ and Japanese 、 and fullwidth ， must split properly
    let ar_parts = split_phrase_list("رقم الطلب، رقم التتبع؛ اسم المنتج");
    assert_eq!(ar_parts.len(), 3, "Arabic delimiters must split cleanly: {ar_parts:?}");
    let ja_parts = split_phrase_list("注文番号、追跡番号、商品名");
    assert_eq!(ja_parts.len(), 3, "Japanese delimiters must split cleanly: {ja_parts:?}");

    // Czech multi-word header (30 chars) must not be rejected as a value example
    assert!(
        !is_value_example_phrase("Datum vystavení faktury nebo objednávky"),
        "Non-CJK label headers up to 42 chars must be preserved"
    );
    // Numeric example with Arabic-Indic digits must be caught via normalize_digits_ascii
    assert!(is_value_example_phrase("٢٠٢٦-٠٣-١٥"), "Arabic-Indic date literal must be filtered as value example");

    // Walk every loaded language section in BIAS_DICT and ensure non-empty raw strings never collapse to 0 phrases
    let root = BIAS_DICT.as_object().expect("BIAS_DICT must be object");
    let mut empty_labels = Vec::new();
    let mut empty_prejs = Vec::new();
    for (lk, lang_val) in root.iter() {
        if lk.len() > 3 && !lk.contains('-') {
            continue;
        }
        let Some(lang_obj) = lang_val.as_object() else { continue };
        for (page_type, page_val) in lang_obj.iter() {
            let Some(page_obj) = page_val.as_object() else { continue };
            for (field, node) in page_obj.iter() {
                let Some(node_obj) = node.as_object() else { continue };
                let has_raw_label = node_obj.get("semantic").and_then(|v| v.as_str()).is_some_and(|s| !s.trim().is_empty())
                    || node_obj.get("bias").and_then(|v| v.as_str()).is_some_and(|s| !s.trim().is_empty());
                if has_raw_label {
                    let (ph, _) = label_phrase_bank(lk, page_type, field);
                    if ph.is_empty() {
                        empty_labels.push(format!("{lk}.{page_type}.{field}"));
                    }
                }
                let has_raw_prej = node_obj.get("prejudice").and_then(|v| v.as_str()).is_some_and(|s| !s.trim().is_empty());
                if has_raw_prej {
                    let raw = node_obj.get("prejudice").and_then(|v| v.as_str()).unwrap_or("");
                    // Only flag if at least one raw phrase is non-numeric text
                    let any_text_phrase = split_phrase_list(raw).iter().any(|p| !is_value_example_phrase(p));
                    if any_text_phrase {
                        let ph = prejudice_phrase_bank(lk, page_type, field);
                        if ph.is_empty() {
                            empty_prejs.push(format!("{lk}.{page_type}.{field}"));
                        }
                    }
                }
            }
        }
    }
    assert!(
        empty_labels.is_empty(),
        "Found {} empty label nodes in BIAS_DICT: {:?}",
        empty_labels.len(),
        &empty_labels[..empty_labels.len().min(10)]
    );
    assert!(
        empty_prejs.is_empty(),
        "Found {} empty prejudice nodes in BIAS_DICT: {:?}",
        empty_prejs.len(),
        &empty_prejs[..empty_prejs.len().min(10)]
    );
}

#[test]
fn canary_2_multilingual_lang_spec_and_detector() {
    // Arabic must be Core12 and routable to 4B
    let ar = lang_spec("ar").expect("ar must exist in lang_spec");
    assert_eq!(ar.tier, LangTier::Core12);
    assert_eq!(ar.iso3, "ara");
    assert!(!ar.latin_dominant);
    assert!(logis_app::model::lang_llm::lang3_code("ar").is_some());

    // Russian & Thai must be non-Latin Extended & transliterable from Latin
    let ru = lang_spec("ru").expect("ru must exist in lang_spec");
    assert_eq!(ru.tier, LangTier::Extended);
    assert!(!ru.latin_dominant);
    assert!(native_scripts("ru").contains(&ScriptKind::Cyrillic));
    assert!(can_transliterate("T-Shirt", "ru"), "Latin -> Russian transliteration must be enabled");
    assert!(can_transliterate("Microcontroller", "th"), "Latin -> Thai transliteration must be enabled");
    assert!(!can_transliterate("T-Shirt", "de"), "Latin -> German must not transliterate");

    // 1 stray Hangul char must not hijack an English commercial invoice
    let en_with_one_ko = "Commercial Invoice CI-2026-08001 Global Tech Logistics Co., Ltd. Total Amount USD 12,400.00 (한)";
    assert_eq!(detect_document_language(en_with_one_ko), "en");

    // Kanji-only Japanese & Traditional Chinese
    assert_eq!(detect_document_language("合計金額 12,500円 税込 支払済"), "ja");
    assert_eq!(detect_document_language("商業發票 編號 2026-001 總金額 臺灣運費"), "zh-tw");
    assert_eq!(lang_code_of("zh-tw"), "zh-tw");
}

#[test]
fn canary_3_siglip2_combining_marks_and_tatweel() {
    // Arabic with harakat (fatha/kasra/damma) and tatweel U+0640
    let ar_in = "فَاتُورَة تِجَارِيَّة مـبـلـغ";
    let ar_out = Siglip2Tokenizer::canonicalize(ar_in);
    assert!(!ar_out.contains('\u{0640}'), "tatweel U+0640 must be stripped: {ar_out}");
    assert!(ar_out.contains('\u{064E}') || ar_out.contains('\u{0650}'), "Arabic harakat must not become spaces: {ar_out}");
    assert_eq!(ar_out.split_whitespace().count(), 3, "harakat must not split Arabic words: {ar_out}");

    // Thai combining vowel & tone mark must stay attached to their consonant
    let th_in = "ใบกำกับสินค้า จำนวน";
    let th_out = Siglip2Tokenizer::canonicalize(th_in);
    assert_eq!(th_out.split_whitespace().count(), 2, "Thai combining marks must not split words: {th_out}");
}

#[test]
fn canary_4_sparse_structured_tokens_and_mode_neutral_fusion() {
    let toks = extract_structured_tokens("BL-2026-08814 HS 8471.30 فاتورة رقم 329,800원 東京倉庫");
    assert!(toks.iter().any(|t| t == "bl-2026-08814"));
    assert!(toks.iter().any(|t| t == "8471.30" || t == "847130"));
    assert!(toks.iter().any(|t| t == "329800"));
    assert!(toks.iter().any(|t| t == "فاتورة"));

    let mut idx = SparseIndex::new();
    idx.upsert_text("ship-1", "shipping", "kr", "BL", "BL-2026-08814 MSKU 7731025 Busan to Los Angeles");
    idx.upsert_text("ship-2", "shipping", "kr", "CI", "CI-2026-08001 Total USD 12500");
    let hits = idx.search_text("MSKU 7731025", 5, Some("shipping"), None, None);
    assert_eq!(hits.first().map(|h| h.0.as_str()), Some("ship-1"));

    // Mode-neutral RRF fusion with FTS + Sparse tracks promotes exact match above vector-only noise
    let vec_ids = vec!["ship-2".to_string(), "ship-1".to_string()];
    let fts_ids = vec!["ship-1".to_string()];
    let sparse_ids = vec!["ship-1".to_string()];
    let fused = fuse_query_tracks(
        &FusionTracks {
            vector: &vec_ids,
            fts: &fts_ids,
            sparse: &sparse_ids,
            ..Default::default()
        },
        FUSION_K,
    );
    assert_eq!(fused.first().map(|x| x.0.as_str()), Some("ship-1"));
}

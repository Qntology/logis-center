//! Thin trade facade bridging `logis-vision-doc` onto `louis-vision` Tier 1 (`lv-doc` + `lv-core`).

use image::GrayImage;
use lv_core::siglip2::{preprocess_for, PatchGrid, SiglipCompatProfile, DEFAULT_MAX_PATCHES};
use lv_doc::vision::grounding::{
    padded_retry_bbox, DocDomainBank, GroundingLexicon, PADDED_RETRY_FRAC,
};
use lv_doc::vision::lattice_verify::{
    verify_ledger_with_shadow_top3, LatticeVerifyDecision, ShadowToken,
};
use lv_doc::vision::legibility::LegibilityMap;
use lv_doc::vision::table_grid_reverse::{inject_virtual_gridlines, VIRTUAL_GRIDLINE_LUMA};

/// Trade document taxonomy bank injected into `lv-doc` (`DocDomainBank`).
#[derive(Debug, Default)]
pub struct TradeDocDomainBank;

impl DocDomainBank for TradeDocDomainBank {
    fn group_anchors(&self) -> Vec<(String, String)> {
        vec![
            ("invoice".to_string(), "commercial invoice proforma".to_string()),
            ("packing_list".to_string(), "packing list weight list".to_string()),
            ("bill_of_lading".to_string(), "bill of lading airwaybill".to_string()),
            ("letter_of_credit".to_string(), "letter of credit documentary credit".to_string()),
            ("certificate_of_origin".to_string(), "certificate of origin".to_string()),
            ("customs_declaration".to_string(), "customs export import declaration".to_string()),
        ]
    }

    fn field_anchors(&self, doc_type: &str, doc_lang: &str) -> Vec<(String, String, String)> {
        let mut out = vec![
            (doc_type.to_string(), "unit".to_string(), format!("{doc_lang} PCS EA CTN PKG")),
            (doc_type.to_string(), "quantity".to_string(), "quantity qty total units".to_string()),
            (doc_type.to_string(), "amount".to_string(), "total amount invoice value".to_string()),
        ];
        if matches!(doc_type, "bill_of_lading" | "invoice") {
            out.push((
                doc_type.to_string(),
                "bl_number".to_string(),
                "B/L NO AWB NO".to_string(),
            ));
        }
        out
    }

    fn is_self_reference_field(&self, doc_type: &str, field: &str) -> bool {
        matches!(
            (doc_type, field),
            ("invoice", "invoice_number")
                | ("bill_of_lading", "bl_number")
                | ("letter_of_credit", "lc_number")
        )
    }
}

/// Trade grounding lexicon preserving `unit = "PCS"` and printed-caption echo guards.
#[derive(Debug, Default)]
pub struct TradeGroundingLexicon {
    pub custom_echoes: Vec<String>,
}

impl GroundingLexicon for TradeGroundingLexicon {
    fn is_exempt(&self, field: &str, value: &str) -> bool {
        if matches!(field, "party_role" | "doc_type" | "incoterms" | "currency") {
            return true;
        }
        if field == "unit" || field.ends_with("_unit") {
            let up = value.trim().to_ascii_uppercase();
            return matches!(
                up.as_str(),
                "PCS" | "PC" | "EA" | "SET" | "BOX" | "PKG" | "CTN" | "PLT" | "PLTS" | "KG" | "MT" | "CBM"
            );
        }
        false
    }

    fn is_printed_label_echo(&self, _field: &str, value: &str) -> bool {
        let up = value.trim().to_ascii_uppercase();
        if matches!(
            up.as_str(),
            "AIRWAYBILL / BILL OF LADING"
                | "COMMERCIAL INVOICE"
                | "PACKING LIST"
                | "BUYER (IF NOT CONSIGNEE)"
                | "PORT OF LOADING"
                | "PORT OF DISCHARGE"
                | "DESCRIPTION OF GOODS"
                | "COUNTRY OF ORIGIN"
        ) {
            return true;
        }
        self.custom_echoes
            .iter()
            .any(|lbl| lbl.trim().eq_ignore_ascii_case(up.as_str()))
    }
}

pub fn preprocess_logis_doc_v1(
    rgb: &[u8],
    width: u32,
    height: u32,
    max_patches: Option<usize>,
) -> anyhow::Result<PatchGrid> {
    preprocess_for(
        SiglipCompatProfile::LogisDocV1,
        rgb,
        width,
        height,
        max_patches.unwrap_or(DEFAULT_MAX_PATCHES),
    )
}

pub fn retry_bbox_with_margin(
    bbox: (u32, u32, u32, u32),
    legibility: &LegibilityMap,
    orig_w: u32,
    orig_h: u32,
) -> Option<(u32, u32, u32, u32)> {
    padded_retry_bbox(bbox, legibility, orig_w, orig_h)
}

pub fn reverse_table_grid(gray: &GrayImage, min_row_bands: usize) -> Option<(GrayImage, Vec<u32>, Vec<u32>)> {
    inject_virtual_gridlines(gray, min_row_bands)
}

pub fn verify_trade_ledger_shadow(
    raw_ocr: &str,
    candidates: &[(&[u8], f32)],
    proposed: &str,
) -> LatticeVerifyDecision {
    let tokens: Vec<ShadowToken> = candidates
        .iter()
        .map(|&(bytes, prob)| ShadowToken {
            bytes: bytes.iter().copied().collect(),
            prob,
        })
        .collect();
    verify_ledger_with_shadow_top3(raw_ocr, &tokens, proposed)
}

pub const TRADE_RETRY_PAD_FRAC: f32 = PADDED_RETRY_FRAC;
pub const TRADE_VIRTUAL_GRIDLINE_LUMA: u8 = VIRTUAL_GRIDLINE_LUMA;

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;
    use lv_doc::vision::legibility::PatchLegibility;

    #[test]
    fn trade_facade_wires_logis_doc_v1_pcs_exemption_and_padded_retry() {
        let lex = TradeGroundingLexicon::default();
        assert!(lex.is_exempt("unit", "PCS"));
        assert!(lex.is_exempt("item_unit", "PLTS"));
        assert!(lex.is_printed_label_echo("consignee", "AIRWAYBILL / BILL OF LADING"));

        let bank = TradeDocDomainBank;
        assert!(bank.is_self_reference_field("invoice", "invoice_number"));
        assert!(!bank.field_anchors("invoice", "en").is_empty());

        let (w, h) = (64u32, 48u32);
        let rgb = vec![200u8; (w * h * 3) as usize];
        let g = preprocess_logis_doc_v1(&rgb, w, h, None).unwrap();
        assert!(g.cells() > 0);

        let leg = LegibilityMap {
            rows: 4,
            cols: 4,
            ink: vec![0.4; 16],
            sharpness: vec![0.5; 16],
            verdict: vec![PatchLegibility::Legible; 16],
            split_applied: false,
            ink_gate: 0.1,
            sharp_gate: 0.1,
        };
        let retried = retry_bbox_with_margin((20, 20, 40, 36), &leg, 100, 100).unwrap();
        assert!(retried.0 < 20 && retried.1 < 20 && retried.2 > 40 && retried.3 > 36);

        let decision = verify_trade_ledger_shadow("  2000 PCS", &[(b"2000 PCS", 0.91)], "2000 PCS");
        assert!(matches!(decision, LatticeVerifyDecision::Verified { .. }));

        let mut tbl = GrayImage::from_pixel(128, 48, Luma([255u8]));
        for &by in &[6u32, 26] {
            for &cx in &[8u32, 38, 68, 98] {
                for dy in 0..8 {
                    for dx in 0..14 {
                        tbl.put_pixel(cx + dx, by + dy, Luma([20u8]));
                    }
                }
            }
        }
        let (_out_img, vx, _vy) = reverse_table_grid(&tbl, 2).unwrap();
        assert!(vx.len() >= 3);
    }
}

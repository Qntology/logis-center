//! Trade identifier repair (`lv-domain`) and Korean numeral/fact verification (`lv-agent`).

use lv_agent::factcheck::{draft_or_fallback, mentions, unsupported_numbers};
use lv_domain::{repair_code, CodeScheme};

/// Repairs noisy OCR trade identifiers / hub keys under `CodeScheme`.
pub fn repair_trade_code(raw: &str, scheme: CodeScheme) -> String {
    repair_code(raw, scheme)
}

/// Checks whether draft Korean/Arabic numbers are fully grounded in `source`, falling back on mismatch.
pub fn verify_trade_draft_facts(
    draft: &str,
    source: &str,
    fallback: &str,
) -> (String, bool, Vec<String>) {
    let _parsed = mentions(source);
    let _bad = unsupported_numbers(draft, source);
    draft_or_fallback(draft, source, fallback)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trade_code_repair_and_factcheck_work_without_model() {
        // KKS: 99AAA99AA999 -> noisy O/I corrected to 0/1 in digit slots, digits fixed in letter slots
        let repaired = repair_trade_code("1O-laa-10-aa-123", CodeScheme::Kks);
        assert_eq!(repaired, "10LAA10AA123");

        let (ok_draft, passed, bad) = verify_trade_draft_facts(
            "총액 20만원 (200,000원)",
            "청구 금액 이십만 원",
            "원본 확인 필요",
        );
        assert!(passed && bad.is_empty(), "ok_draft={ok_draft}, bad={bad:?}");

        let (fb, failed_ok, hallucinated) = verify_trade_draft_facts(
            "총액 999,000원",
            "청구 금액 200,000원",
            "원본 금액 불일치",
        );
        assert!(!failed_ok && fb == "원본 금액 불일치" && !hallucinated.is_empty());
    }
}

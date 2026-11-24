/// Character-count-based multilingual document language detector (Phase 4-1 D3).
/// Replaces 1-char boolean precedence so a stray Hangul char cannot override a Japanese/English doc,
/// distinguishes Kanji-only Japanese & Traditional Chinese (`zh-tw`) vs Simplified (`zh-hans`),
/// and uses `whatlang` for Latin disambiguation.
pub fn detect_document_language(text: &str) -> String {
    let mut c_ko = 0usize;
    let mut c_kana = 0usize;
    let mut c_han = 0usize;
    let mut c_ru = 0usize;
    let mut c_ar = 0usize;
    let mut c_th = 0usize;
    let mut c_hi = 0usize;
    let mut c_bn = 0usize;
    let mut c_el = 0usize;
    let mut c_he = 0usize;
    let mut c_vi = 0usize;
    let mut c_latin = 0usize;

    for c in text.chars() {
        let u = c as u32;
        if (0xAC00..=0xD7A3).contains(&u) || (0x1100..=0x11FF).contains(&u) || (0x3130..=0x318F).contains(&u) {
            c_ko += 1;
        } else if (0x3040..=0x309F).contains(&u) || (0x30A0..=0x30FF).contains(&u) || (0xFF66..=0xFF9F).contains(&u) {
            c_kana += 1;
        } else if (0x4E00..=0x9FFF).contains(&u) || (0x3400..=0x4DBF).contains(&u) {
            c_han += 1;
        } else if (0x0400..=0x04FF).contains(&u) {
            c_ru += 1;
        } else if (0x0600..=0x06FF).contains(&u) || (0x0750..=0x077F).contains(&u) {
            c_ar += 1;
        } else if (0x0E00..=0x0E7F).contains(&u) {
            c_th += 1;
        } else if (0x0900..=0x097F).contains(&u) {
            c_hi += 1;
        } else if (0x0980..=0x09FF).contains(&u) {
            c_bn += 1;
        } else if (0x0370..=0x03FF).contains(&u) {
            c_el += 1;
        } else if (0x0590..=0x05FF).contains(&u) {
            c_he += 1;
        } else if (0x1EA0..=0x1EF9).contains(&u) {
            c_vi += 1;
        } else if (0x0041..=0x005A).contains(&u) || (0x0061..=0x007A).contains(&u) || (0x00C0..=0x024F).contains(&u) {
            c_latin += 1;
        }
    }

    // Japanese combines Kana + Han when Kana is present; standalone Han is scored separately.
    let c_ja = if c_kana > 0 { c_kana + c_han } else { 0 };
    let c_zh = if c_kana == 0 { c_han } else { 0 };
    let non_latin_counts = [
        ("ko", c_ko),
        ("ja", c_ja),
        ("zh", c_zh),
        ("th", c_th),
        ("ru", c_ru),
        ("ar", c_ar),
        ("hi", c_hi),
        ("bn", c_bn),
        ("el", c_el),
        ("he", c_he),
        ("vi", c_vi),
    ];
    let (best_nl, best_nl_count) = non_latin_counts
        .into_iter()
        .max_by_key(|&(_, n)| n)
        .unwrap_or(("en", 0));

    // Avoid 1-char stray non-Latin overriding a predominantly Latin document (e.g. 1 Hangul char in 80 Latin chars).
    let nl_wins = best_nl_count > 0 && (c_latin == 0 || best_nl_count >= 2 || best_nl_count * 5 >= c_latin);

    let local_language = if nl_wins {
        if best_nl == "zh" {
            // Disambiguate Kanji-only Japanese (Shinjitai) & Traditional Chinese (zh-tw) vs Simplified (zh-hans).
            let has_jp_kanji = text.chars().any(|c| matches!(c, '円' | '駅' | '売' | '込' | '届' | '済' | '払'));
            let has_hant = text.chars().any(|c| matches!(c, '繁' | '體' | '訂' | '貨' | '編' | '號' | '總' | '發' | '運' | '買' | '賣' | '臺' | '灣' | '國' | '際'));
            if has_jp_kanji {
                "ja".to_string()
            } else if has_hant {
                "zh-tw".to_string()
            } else {
                match whatlang::detect(text).map(|i| i.lang()) {
                    Some(whatlang::Lang::Jpn) => "ja".to_string(),
                    _ => "zh-hans".to_string(),
                }
            }
        } else {
            best_nl.to_string()
        }
    } else if c_latin > 0 {
        match whatlang::detect(text) {
            Some(info) => match info.lang() {
                whatlang::Lang::Fra => "fr",
                whatlang::Lang::Deu => "de",
                whatlang::Lang::Spa => "es",
                whatlang::Lang::Ita => "it",
                whatlang::Lang::Por => "pt",
                whatlang::Lang::Nld => "nl",
                whatlang::Lang::Ces => "cs",
                _ => "en",
            }
            .to_string(),
            None => "en".to_string(),
        }
    } else {
        "en".to_string()
    };

    local_language
}
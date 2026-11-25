//! 글자의 문자 체계(라틴 · 한글 · 가나 · 한자 · 아랍 …)와 문서 언어의 고유 문자 섞임 비율. 예전 model/lang_llm.rs 에서 옮김.

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ScriptKind {
    Latin,
    Hangul,
    Kana,
    Han,
    Cyrillic,
    Thai,
    Arabic,
    Devanagari,
    Bengali,
    Greek,
    Hebrew,
    Other,
}

pub fn script_kind(c: char) -> Option<ScriptKind> {
    if !c.is_alphabetic() {
        return None;
    }
    Some(match c as u32 {
        0x0041..=0x005A | 0x0061..=0x007A | 0x00C0..=0x024F | 0x1E00..=0x1EFF => ScriptKind::Latin,
        0x1100..=0x11FF | 0x3130..=0x318F | 0xA960..=0xA97F | 0xAC00..=0xD7FF => ScriptKind::Hangul,
        0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F => ScriptKind::Kana,
        0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xF900..=0xFAFF => ScriptKind::Han,
        0x0400..=0x052F => ScriptKind::Cyrillic,
        0x0E00..=0x0E7F => ScriptKind::Thai,
        0x0600..=0x06FF | 0x0750..=0x077F | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => ScriptKind::Arabic,
        0x0900..=0x097F => ScriptKind::Devanagari,
        0x0980..=0x09FF => ScriptKind::Bengali,
        0x0370..=0x03FF => ScriptKind::Greek,
        0x0590..=0x05FF => ScriptKind::Hebrew,
        _ => ScriptKind::Other,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LangTier {
    /// Core 12 languages supported natively across bias, Stanza, vocab plans, and Core12 trimmed artifacts.
    Core12,
    /// Extended languages supported on full-vocab artifacts (fallback to Base2B on Core12-trimmed artifacts).
    Extended,
    Unsupported,
}

#[derive(Clone, Copy, Debug)]
pub struct LangSpec {
    pub iso1: &'static str,
    pub iso3: &'static str,
    pub tier: LangTier,
    pub scripts: &'static [ScriptKind],
    pub latin_dominant: bool,
}

static LANG_SPECS: &[LangSpec] = &[
    // Core 12
    LangSpec { iso1: "ko", iso3: "kor", tier: LangTier::Core12, scripts: &[ScriptKind::Hangul], latin_dominant: false },
    LangSpec { iso1: "en", iso3: "eng", tier: LangTier::Core12, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "ja", iso3: "jpn", tier: LangTier::Core12, scripts: &[ScriptKind::Kana, ScriptKind::Han], latin_dominant: false },
    LangSpec { iso1: "zh", iso3: "zho", tier: LangTier::Core12, scripts: &[ScriptKind::Han], latin_dominant: false },
    LangSpec { iso1: "fr", iso3: "fra", tier: LangTier::Core12, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "de", iso3: "deu", tier: LangTier::Core12, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "es", iso3: "spa", tier: LangTier::Core12, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "it", iso3: "ita", tier: LangTier::Core12, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "pt", iso3: "por", tier: LangTier::Core12, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "nl", iso3: "nld", tier: LangTier::Core12, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "ar", iso3: "ara", tier: LangTier::Core12, scripts: &[ScriptKind::Arabic], latin_dominant: false },
    LangSpec { iso1: "cs", iso3: "ces", tier: LangTier::Core12, scripts: &[ScriptKind::Latin], latin_dominant: true },
    // Extended non-Latin & Latin
    LangSpec { iso1: "ru", iso3: "rus", tier: LangTier::Extended, scripts: &[ScriptKind::Cyrillic], latin_dominant: false },
    LangSpec { iso1: "uk", iso3: "ukr", tier: LangTier::Extended, scripts: &[ScriptKind::Cyrillic], latin_dominant: false },
    LangSpec { iso1: "be", iso3: "bel", tier: LangTier::Extended, scripts: &[ScriptKind::Cyrillic], latin_dominant: false },
    LangSpec { iso1: "bg", iso3: "bul", tier: LangTier::Extended, scripts: &[ScriptKind::Cyrillic], latin_dominant: false },
    LangSpec { iso1: "sr", iso3: "srp", tier: LangTier::Extended, scripts: &[ScriptKind::Cyrillic, ScriptKind::Latin], latin_dominant: false },
    LangSpec { iso1: "mk", iso3: "mkd", tier: LangTier::Extended, scripts: &[ScriptKind::Cyrillic], latin_dominant: false },
    LangSpec { iso1: "kk", iso3: "kaz", tier: LangTier::Extended, scripts: &[ScriptKind::Cyrillic], latin_dominant: false },
    LangSpec { iso1: "th", iso3: "tha", tier: LangTier::Extended, scripts: &[ScriptKind::Thai], latin_dominant: false },
    LangSpec { iso1: "hi", iso3: "hin", tier: LangTier::Extended, scripts: &[ScriptKind::Devanagari], latin_dominant: false },
    LangSpec { iso1: "mr", iso3: "mar", tier: LangTier::Extended, scripts: &[ScriptKind::Devanagari], latin_dominant: false },
    LangSpec { iso1: "ne", iso3: "nep", tier: LangTier::Extended, scripts: &[ScriptKind::Devanagari], latin_dominant: false },
    LangSpec { iso1: "bn", iso3: "ben", tier: LangTier::Extended, scripts: &[ScriptKind::Bengali], latin_dominant: false },
    LangSpec { iso1: "el", iso3: "ell", tier: LangTier::Extended, scripts: &[ScriptKind::Greek], latin_dominant: false },
    LangSpec { iso1: "he", iso3: "heb", tier: LangTier::Extended, scripts: &[ScriptKind::Hebrew], latin_dominant: false },
    LangSpec { iso1: "fa", iso3: "fas", tier: LangTier::Extended, scripts: &[ScriptKind::Arabic], latin_dominant: false },
    LangSpec { iso1: "ur", iso3: "urd", tier: LangTier::Extended, scripts: &[ScriptKind::Arabic], latin_dominant: false },
    LangSpec { iso1: "vi", iso3: "vie", tier: LangTier::Extended, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "id", iso3: "ind", tier: LangTier::Extended, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "tr", iso3: "tur", tier: LangTier::Extended, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "pl", iso3: "pol", tier: LangTier::Extended, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "sk", iso3: "slk", tier: LangTier::Extended, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "sl", iso3: "slv", tier: LangTier::Extended, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "hr", iso3: "hrv", tier: LangTier::Extended, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "ro", iso3: "ron", tier: LangTier::Extended, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "hu", iso3: "hun", tier: LangTier::Extended, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "fi", iso3: "fin", tier: LangTier::Extended, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "sv", iso3: "swe", tier: LangTier::Extended, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "da", iso3: "dan", tier: LangTier::Extended, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "no", iso3: "nob", tier: LangTier::Extended, scripts: &[ScriptKind::Latin], latin_dominant: true },
    LangSpec { iso1: "ta", iso3: "tam", tier: LangTier::Extended, scripts: &[ScriptKind::Other], latin_dominant: false },
    LangSpec { iso1: "te", iso3: "tel", tier: LangTier::Extended, scripts: &[ScriptKind::Other], latin_dominant: false },
];

/// Canonicalize BCP-47 / free-form language labels (B16 lowercase trim, B24 `"Traditional Chinese"` -> `"zh-tw"`).
pub fn canonical_lang_tag(raw: &str) -> String {
    let lower = raw.trim().to_ascii_lowercase().replace('_', "-");
    if lower.is_empty() {
        return String::new();
    }
    match lower.as_str() {
        "traditional chinese" | "chinese (traditional)" | "zh-hant" | "zh-tw" | "zh-hk" | "zh-mo" => {
            "zh-tw".to_string()
        }
        "simplified chinese" | "chinese (simplified)" | "zh-hans" | "zh-cn" | "zh-sg" => {
            "zh-hans".to_string()
        }
        "korean" => "ko".to_string(),
        "english" => "en".to_string(),
        "japanese" => "ja".to_string(),
        "chinese" => "zh".to_string(),
        "french" => "fr".to_string(),
        "german" => "de".to_string(),
        "spanish" => "es".to_string(),
        "italian" => "it".to_string(),
        "portuguese" => "pt".to_string(),
        "dutch" => "nl".to_string(),
        "arabic" => "ar".to_string(),
        "czech" => "cs".to_string(),
        "russian" => "ru".to_string(),
        "ukrainian" => "uk".to_string(),
        "thai" => "th".to_string(),
        "hindi" => "hi".to_string(),
        "bengali" => "bn".to_string(),
        "greek" => "el".to_string(),
        "hebrew" | "iw" => "he".to_string(),
        "persian" | "farsi" => "fa".to_string(),
        "urdu" => "ur".to_string(),
        "vietnamese" => "vi".to_string(),
        "indonesian" => "id".to_string(),
        "turkish" => "tr".to_string(),
        "polish" => "pl".to_string(),
        "nb" | "nn" => "no".to_string(),
        _ => lower,
    }
}

/// Static language specification lookup (replaces bias-derived script detection; fixes `ar` and `ru`/`th`/`hi`).
pub fn lang_spec(doc_lang: &str) -> Option<&'static LangSpec> {
    let canon = canonical_lang_tag(doc_lang);
    if canon.is_empty() {
        return None;
    }
    let mapped = match canon.as_str() {
        "zh-cn" | "zh-tw" | "zh-hk" | "zh-hans" | "zh-hant" => "zh",
        other => other.split(|c: char| c == '-' || c == '_').next().unwrap_or(other),
    };
    LANG_SPECS
        .iter()
        .find(|s| s.iso1 == mapped || s.iso3 == mapped)
}

pub fn native_scripts(doc_lang: &str) -> std::collections::BTreeSet<ScriptKind> {
    if let Some(spec) = lang_spec(doc_lang) {
        return spec.scripts.iter().copied().collect();
    }
    let sample = crate::nl_convert::native_script_sample(doc_lang, "", "");
    let mut set: std::collections::BTreeSet<ScriptKind> = sample.chars().filter_map(script_kind).collect();
    if set.contains(&ScriptKind::Kana) {
        set.insert(ScriptKind::Han);
    }
    set
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ScriptMix {
    pub letters: usize,
    pub native: usize,
    pub latin: usize,
    pub other: usize,
}

impl ScriptMix {
    fn share(&self, n: usize) -> f32 {
        if self.letters == 0 {
            0.0
        } else {
            n as f32 / self.letters as f32
        }
    }
    pub fn native_share(&self) -> f32 {
        self.share(self.native)
    }
    pub fn latin_share(&self) -> f32 {
        self.share(self.latin)
    }
    pub fn other_share(&self) -> f32 {
        self.share(self.other)
    }
}

pub fn script_mix(text: &str, native: &std::collections::BTreeSet<ScriptKind>) -> ScriptMix {
    let mut m = ScriptMix::default();
    for c in text.chars() {
        let k = match script_kind(c) {
            Some(k) => k,
            None => continue,
        };
        m.letters += 1;
        if native.contains(&k) {
            m.native += 1;
        } else if k == ScriptKind::Latin {
            m.latin += 1;
        } else {
            m.other += 1;
        }
    }
    m
}

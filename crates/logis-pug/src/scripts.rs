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

pub fn native_scripts(doc_lang: &str) -> std::collections::BTreeSet<ScriptKind> {
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

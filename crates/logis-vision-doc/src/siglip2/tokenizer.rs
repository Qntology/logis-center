use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;

/// 🌟 [SPLIT-GGUF] 파싱한 토크나이저. 텍스트 타워를 다시 붙일 때마다 34 MB 짜리 tokenizer.json 을 다시 읽지 않습니다.
/// 장치 메모리와 관계없는 값이라 타워를 내려도 남겨 둡니다. 파일 크기가 달라지면 다시 읽습니다.
static PARSED: Lazy<Mutex<HashMap<(PathBuf, u64), Arc<tokenizers::Tokenizer>>>> = Lazy::new(|| Mutex::new(HashMap::new()));

fn parsed(path: &Path) -> anyhow::Result<Arc<tokenizers::Tokenizer>> {
    let key = (path.to_path_buf(), std::fs::metadata(path)?.len());
    let mut cache = PARSED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(found) = cache.get(&key) {
        return Ok(Arc::clone(found));
    }
    let tokenizer = Arc::new(tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!("Failed to load SigLIP2 tokenizer: {}", e))?);
    cache.insert(key, Arc::clone(&tokenizer));
    Ok(tokenizer)
}

pub struct Siglip2Tokenizer {
    inner: Arc<tokenizers::Tokenizer>,
    pad_id: u32,
    max_len: usize,
}

impl Siglip2Tokenizer {
    pub fn from_dir(model_dir: &Path, pad_id: u32, max_len: usize) -> anyhow::Result<Self> {
        let path = model_dir.join("tokenizer.json");
        if !path.exists() {
            return Err(anyhow::anyhow!(
                "SigLIP2 tokenizer.json not found at {:?}. Please re-download the SigLIP2 model.",
                path
            ));
        }
        Ok(Self {
            inner: parsed(&path)?,
            pad_id,
            max_len: max_len.max(1),
        })
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// Unicode combining mark predicate (`Mn`/`Mc`/`Me` common blocks: Latin diacritics, Arabic harakat, Thai/Lao/Devanagari/Bengali/Tamil vowels, Hebrew nikud).
    fn is_combining_mark(c: char) -> bool {
        let u = c as u32;
        matches!(
            u,
            0x0300..=0x036F // Combining Diacritical Marks
            | 0x0591..=0x05BD | 0x05BF | 0x05C1..=0x05C2 | 0x05C4..=0x05C7 // Hebrew
            | 0x0610..=0x061A | 0x064B..=0x065F | 0x0670 | 0x06D6..=0x06ED // Arabic harakat & Quranic marks
            | 0x0900..=0x0903 | 0x093A..=0x094F | 0x0951..=0x0957 | 0x0962..=0x0963 // Devanagari
            | 0x0981..=0x0983 | 0x09BC | 0x09BE..=0x09CD | 0x09D7 | 0x09E2..=0x09E3 // Bengali
            | 0x0E31 | 0x0E34..=0x0E3A | 0x0E47..=0x0E4E // Thai combining vowels & tone marks
            | 0x1AB0..=0x1AFF | 0x1DC0..=0x1DFF | 0x20D0..=0x20FF | 0xFE20..=0xFE2F
        )
    }

    pub fn canonicalize(text: &str) -> String {
        // Phase 4-1 D2: Preserve combining marks (Arabic harakat, Thai/Hindi vowels) & strip Arabic tatweel U+0640.
        let mut out = String::with_capacity(text.len());
        for c in text.chars() {
            if c == '\u{0640}' {
                continue;
            }
            if c.is_alphanumeric() {
                for lc in c.to_lowercase() {
                    out.push(lc);
                }
            } else if Self::is_combining_mark(c) {
                out.push(c);
            } else {
                out.push(' ');
            }
        }
        out.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    pub fn encode_one(&self, text: &str) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
        let canon = Self::canonicalize(text);
        let enc = self
            .inner
            .encode(canon.as_str(), true)
            .map_err(|e| anyhow::anyhow!("SigLIP2 tokenize failed: {}", e))?;

        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        if ids.len() > self.max_len {
            ids.truncate(self.max_len);
        }
        let real = ids.len();
        while ids.len() < self.max_len {
            ids.push(self.pad_id);
        }

        let mut mask = vec![0u32; self.max_len];
        for i in 0..real {
            mask[i] = 1;
        }

        Ok((ids, mask))
    }

    pub fn encode_batch(&self, texts: &[String]) -> anyhow::Result<Vec<(Vec<u32>, Vec<u32>)>> {
        let mut out = Vec::with_capacity(texts.len());
        for t in texts {
            out.push(self.encode_one(t)?);
        }
        Ok(out)
    }
}
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

    fn canonicalize(text: &str) -> String {
        text.chars()
            .map(|c| {
                if c.is_alphanumeric() || c.is_whitespace() {
                    c.to_lowercase().next().unwrap_or(c)
                } else {
                    ' '
                }
            })
            .collect::<String>()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
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
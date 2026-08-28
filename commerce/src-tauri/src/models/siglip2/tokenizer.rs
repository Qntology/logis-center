use std::path::Path;

/// 🌟 [SigLIP2 TEXT TOKENIZER]
///
///  ── 왜 별도 래퍼인가 ──
///   SigLIP 계열은 토크나이즈 전에 **canonicalize** 를 반드시 수행합니다.
///   (소문자화 → 구두점 제거 → 공백 정규화)
///   그리고 시퀀스를 항상 max_len(64) 으로 패딩/절단한 뒤
///   마지막 위치를 풀링합니다. 이 계약을 코드로 고정합니다.
///
///  ── 필요한 파일 ──
///   {model_dir}/tokenizer.json  (gemma sentencepiece, vocab 256000)
pub struct Siglip2Tokenizer {
    inner: tokenizers::Tokenizer,
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
        let inner = tokenizers::Tokenizer::from_file(&path)
            .map_err(|e| anyhow::anyhow!("Failed to load SigLIP2 tokenizer: {}", e))?;
        Ok(Self {
            inner,
            pad_id,
            max_len: max_len.max(1),
        })
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }

    /// SigLIP canonicalize_text 와 동일한 정규화.
    ///  · 소문자화
    ///  · 알파벳/숫자/공백을 제외한 모든 문자를 공백으로 치환
    ///  · 연속 공백 압축
    ///
    /// 어떤 언어의 어휘도 코드에 등장하지 않습니다. 문자 클래스 판정만 사용합니다.
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

    /// 텍스트 하나를 (ids, attention_mask) 로 변환합니다.
    /// 길이는 항상 max_len 으로 고정됩니다.
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
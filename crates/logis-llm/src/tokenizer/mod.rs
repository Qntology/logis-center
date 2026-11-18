use anyhow::{Ok, Result, anyhow};
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

pub struct TokenizerModel {
    pub tokenizer: Tokenizer,
}

impl TokenizerModel {
    pub fn new(tokenizer: Tokenizer) -> Self {
        Self { tokenizer }
    }

    pub fn init(path: &str) -> Result<Self> {
        let dir = std::path::Path::new(path);
        if !dir.exists() {
            return Err(anyhow!("model path not found: {}", path));
        }
        let tokenizer_file = dir.join("tokenizer.json");
        if !tokenizer_file.exists() {
            return Err(anyhow!("tokenizer.json not found in model path: {:?}", tokenizer_file));
        }
        let tokenizer = Tokenizer::from_file(tokenizer_file)
            .map_err(|e| anyhow!(format!("tokenizer from file error{}", e)))?;
        Ok(Self { tokenizer })
    }

    pub fn text_encode_vec(&self, text: String, add_special_token: bool) -> Result<Vec<u32>> {
        let token_id = self
            .tokenizer
            .encode(text, add_special_token)
            .map_err(|e| anyhow!(format!("tokenizer encode error: {}", e)))?
            .get_ids()
            .to_vec();
        Ok(token_id)
    }
    pub fn text_encode(&self, text: String, device: &Device) -> Result<Tensor> {
        let token_id = self.text_encode_vec(text, true)?;
        let token_tensor = Tensor::from_slice(&token_id, (1, token_id.len()), device)?;
        Ok(token_tensor)
    }

    pub fn token_decode(&self, tokens: Vec<u32>) -> Result<String> {
        let decode = self
            .tokenizer
            .decode(&tokens, true)
            .map_err(|e| anyhow!(format!("tokenizer encode error{}", e)))?;
        Ok(decode)
    }
}

/// 🌟 [UTF8-STREAM] 토큰을 하나씩 받아 '완성된' UTF-8 글자만 표준출력용으로 내보냅니다.
///
/// ByteLevel BPE 는 한 글자(예: 한글 3바이트)를 여러 토큰으로 나눌 수 있어, 토큰 하나만 디코드하면
/// 미완성 바이트가 U+FFFD(�) 로 바뀝니다('총����액'). 여기서는 아직 끝나지 않은 토큰을 붙잡아 두었다가
/// 다음 토큰과 함께 디코드해 글자가 완성될 때 한 번에 내보냅니다. 최종 결과 텍스트(전체 시퀀스 디코드)와
/// 생성 휴리스틱용 버퍼는 건드리지 않습니다. `LOGIS_STREAM_UTF8=0` 이면 예전처럼 토큰마다 바로 디코드합니다.
#[derive(Default)]
pub struct Utf8TokenStream {
    pending: Vec<u32>,
}

impl Utf8TokenStream {
    /// 미완성으로 붙잡아 둘 최대 토큰 수. 넘으면(정말로 깨진 바이트열) 손실 디코드로 내보냅니다.
    const MAX_PENDING: usize = 8;

    pub fn new() -> Self {
        Self::default()
    }

    fn enabled() -> bool {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *ON.get_or_init(|| {
            let v = std::env::var("LOGIS_STREAM_UTF8").unwrap_or_default().trim().to_ascii_lowercase();
            !matches!(v.as_str(), "0" | "off" | "false" | "no")
        })
    }

    /// 토큰 하나를 받아, 이번에 완성된 텍스트를 돌려줍니다. (아직 미완성이면 빈 문자열)
    pub fn push(&mut self, tok: &TokenizerModel, token: u32) -> String {
        if !Self::enabled() {
            return tok.token_decode(vec![token]).unwrap_or_default();
        }
        self.pending.push(token);
        let text = match tok.token_decode(self.pending.clone()) {
            std::result::Result::Ok(t) => t,
            Err(_) => {
                self.pending.clear();
                return String::new();
            }
        };
        if text.ends_with('\u{FFFD}') && self.pending.len() < Self::MAX_PENDING {
            return String::new();
        }
        self.pending.clear();
        text
    }

    /// 남아 있는 미완성 토큰을 (손실 디코드로) 모두 내보냅니다.
    pub fn finish(&mut self, tok: &TokenizerModel) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let text = tok.token_decode(std::mem::take(&mut self.pending)).unwrap_or_default();
        text
    }
}

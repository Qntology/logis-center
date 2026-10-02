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

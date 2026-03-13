use anyhow::{Ok, Result, anyhow};
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

pub struct TokenizerModel {
    pub tokenizer: Tokenizer,
}

impl TokenizerModel {
    pub fn init(path: &str) -> Result<Self> {
        let path = path.to_string();
        assert!(
            std::path::Path::new(&path).exists(),
            "model path file not exists"
        );
        let tokenizer_file = std::path::Path::new(&path).join("tokenizer.json");
        assert!(
            tokenizer_file.exists(),
            "tokenizer.json not exists in model path: {:?}",
            tokenizer_file
        );
        let tokenizer = Tokenizer::from_file(tokenizer_file)
            .map_err(|e| anyhow!(format!("tokenizer from file error{}", e)))?;
        Ok(Self { tokenizer })
    }

    pub fn text_encode_vec(&self, text: String, add_special_token: bool) -> Result<Vec<u32>> {
        // [FIX] Use encode with 'is_pretokenized=false' but ensure special tokens are handled.
        // Some tokenizers require a specific call to handle added tokens correctly.
        let encoding = self
            .tokenizer
            .encode(text, add_special_token)
            .map_err(|e| anyhow!(format!("tokenizer encode error: {}", e)))?;
        
        Ok(encoding.get_ids().to_vec())
    }
    pub fn text_encode(&self, text: String, device: &Device) -> Result<Tensor> {
        let token_id = self.text_encode_vec(text, true)?;
        let token_tensor = Tensor::from_slice(&token_id, (1, token_id.len()), device)?.to_dtype(candle_core::DType::U32)?;
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

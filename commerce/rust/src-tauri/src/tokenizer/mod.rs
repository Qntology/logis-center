use anyhow::{Result, anyhow};
use tokenizers::Tokenizer;
use std::path::Path;

#[derive(Clone)]
pub struct TokenizerModel {
    pub tokenizer: Tokenizer,
}

impl TokenizerModel {
    pub fn init(path: &str) -> Result<Self> {
        let tokenizer_path = Path::new(path).join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!(e))?;
        Ok(Self { tokenizer })
    }

    pub fn text_encode_vec(&self, text: String, add_special: bool) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, add_special).map_err(|e| anyhow!(e))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn token_decode(&self, ids: Vec<u32>) -> Result<String> {
        self.tokenizer.decode(&ids, true).map_err(|e| anyhow!(e))
    }
}
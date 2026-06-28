use crate::models::granite::model::MambaLayerCache;

use candle_core::{Result, Tensor, D, IndexOp};
use crate::models::granite::model::{GraniteMoeHybrid, GraniteHybridCache};
use tokenizers::Tokenizer;
use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub struct GraniteGenerateModel {
    pub model: GraniteMoeHybrid,
    pub tokenizer: Tokenizer,
    pub cache: Option<GraniteHybridCache>,
}

impl GraniteGenerateModel {
    pub fn new(model: GraniteMoeHybrid, tokenizer: Tokenizer) -> Self {
        Self {
            model,
            tokenizer,
            cache: None,
        }
    }

    pub fn clear_kv_cache(&mut self) {
        self.cache = None;
    }

    pub fn get_cache_snapshot(&self) -> Option<GraniteHybridCache> {
        self.cache.clone()
    }

    pub fn set_cache_snapshot(&mut self, cache: Option<GraniteHybridCache>) {
        self.cache = cache;
    }

    pub fn prefill(
        &mut self,
        prompt: &str,
        device: &candle_core::Device,
    ) -> Result<()> {
        let tokens = self.tokenizer.encode(prompt, true).map_err(candle_core::Error::msg)?;
        let tokens = tokens.get_ids().to_vec();

        if self.cache.is_none() {
            let mut attention_caches = Vec::new();
            let mut mamba_caches = Vec::new();
            let dtype = self.model.wte.embeddings().dtype();
            for layer_type in &self.model.cfg.layer_types {
                if layer_type == "attention" {
                    attention_caches.push((
                        Tensor::zeros((1, self.model.cfg.num_key_value_heads(), 0, self.model.cfg.head_dim()), dtype, device)?,
                        Tensor::zeros((1, self.model.cfg.num_key_value_heads(), 0, self.model.cfg.head_dim()), dtype, device)?,
                    ));
                } else {
                    mamba_caches.push(MambaLayerCache::new(1, &self.model.cfg, device, dtype)?);
                }
            }
            self.cache = Some(GraniteHybridCache { attention_caches, mamba_caches });
        }

        let cache = self.cache.as_mut().unwrap();
        let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
        let _ = self.model.forward(&input, cache)?;

        Ok(())
    }

    pub fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        device: &candle_core::Device,
        cancel_token: Option<Arc<AtomicBool>>,
    ) -> Result<String> {
        let add_special = self.cache.is_none();
        let tokens = self.tokenizer.encode(prompt, add_special).map_err(candle_core::Error::msg)?;
        let mut tokens = tokens.get_ids().to_vec();
        let mut generated_tokens = Vec::new();
        let mut prev_text_len = 0;

        // Initialize cache only if it's currently None (Stateful maintenance)
        if self.cache.is_none() {
            let mut attention_caches = Vec::new();
            let mut mamba_caches = Vec::new();
            let dtype = self.model.wte.embeddings().dtype();
            for layer_type in &self.model.cfg.layer_types {
                if layer_type == "attention" {
                    attention_caches.push((
                        Tensor::zeros((1, self.model.cfg.num_key_value_heads(), 0, self.model.cfg.head_dim()), dtype, device)?,
                        Tensor::zeros((1, self.model.cfg.num_key_value_heads(), 0, self.model.cfg.head_dim()), dtype, device)?,
                    ));
                } else {
                    mamba_caches.push(MambaLayerCache::new(1, &self.model.cfg, device, dtype)?);
                }
            }
            self.cache = Some(GraniteHybridCache { attention_caches, mamba_caches });
        }

        let cache = self.cache.as_mut().unwrap();

        // Prefill prompt
        let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
        let mut logits = self.model.forward(&input, cache)?;

        for _ in 0..max_tokens {
            if let Some(flag) = &cancel_token {
                if flag.load(Ordering::Relaxed) {
                    break;
                }
            }

            let logits_last = logits.i((0, logits.dim(1)? - 1, ..))?;
            
            // Greedy sampling
            let next_token = logits_last.argmax(D::Minus1)?.to_scalar::<u32>()?;
            
            if next_token == self.tokenizer.token_to_id("<|end_of_text|>").unwrap_or(u32::MAX) {
                break;
            }

            tokens.push(next_token);
            generated_tokens.push(next_token);
            
            // 실시간 로그 출력 로직 (한글 깨짐 방지 처리 포함)
            let current_text = self.tokenizer.decode(&generated_tokens, true).unwrap_or_default();
            // 새로 추가된 텍스트가 있고, 마지막 문자가 디코딩 실패로 인한 깨진 문자가 아닐 때만 출력
            if current_text.len() > prev_text_len && !current_text.ends_with('\u{FFFD}') {
                let new_text = &current_text[prev_text_len..];
                print!("{}", new_text);
                // I/O 병목을 줄이기 위해 매 토큰마다 화면에 밀어내지 않고, 100개의 토큰이 쌓일 때마다 한 번씩 출력합니다.
                if generated_tokens.len() % 100 == 0 {
                    std::io::stdout().flush().unwrap();
                }
                prev_text_len = current_text.len();
            }
            
            // Next input is just the single new token
            let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
            logits = self.model.forward(&input, cache)?;
        }
        
        let generated_text = self.tokenizer.decode(&generated_tokens, true).map_err(candle_core::Error::msg)?;
        
        // 루프가 끝난 후, 미처 출력되지 못한 마지막 조각이 있다면 마저 출력
        if generated_text.len() > prev_text_len {
            print!("{}", &generated_text[prev_text_len..]);
        }
        // 루프가 모두 끝난 후 무조건 버퍼를 한 번 비워줍니다.
        std::io::stdout().flush().unwrap();
        println!(); // 마지막에 콘솔 줄바꿈 추가

        Ok(generated_text)
    }
}

use candle_core::{Result, Tensor, D, IndexOp};
use crate::model::{GraniteMoeHybrid, GraniteHybridCache};
use tokenizers::Tokenizer;
use std::io::Write;

pub fn generate(
    model: &GraniteMoeHybrid,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    device: &candle_core::Device,
) -> Result<String> {
    let tokens = tokenizer.encode(prompt, true).map_err(candle_core::Error::msg)?;
    let mut tokens = tokens.get_ids().to_vec();
    let mut generated_tokens = Vec::new();
    let mut prev_text_len = 0;

    // Initialize cache
    let mut attention_caches = Vec::new();
    let mut mamba_caches = Vec::new();
    for layer_type in &model.cfg.layer_types {
        if layer_type == "attention" {
            attention_caches.push((
                Tensor::zeros((1, model.cfg.num_key_value_heads(), 0, model.cfg.head_dim()), candle_core::DType::F32, device)?,
                Tensor::zeros((1, model.cfg.num_key_value_heads(), 0, model.cfg.head_dim()), candle_core::DType::F32, device)?,
            ));
        } else {
            mamba_caches.push(crate::model::MambaLayerCache::new(1, &model.cfg, device, candle_core::DType::F32)?);
        }
    }
    let mut cache = GraniteHybridCache { attention_caches, mamba_caches };

    // Prefill prompt
    let input = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
    let mut logits = model.forward(&input, &mut cache)?;

    for _ in 0..max_tokens {
        let logits_last = logits.i((0, logits.dim(1)? - 1, ..))?;
        
        // Greedy sampling
        let next_token = logits_last.argmax(D::Minus1)?.to_scalar::<u32>()?;
        
        if next_token == tokenizer.token_to_id("<|end_of_text|>").unwrap_or(u32::MAX) {
            break;
        }

        tokens.push(next_token);
        generated_tokens.push(next_token);
        
        // 실시간 로그 출력 로직 (한글 깨짐 방지 처리 포함)
        let current_text = tokenizer.decode(&generated_tokens, true).unwrap_or_default();
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
        logits = model.forward(&input, &mut cache)?;
    }
    
    let generated_text = tokenizer.decode(&generated_tokens, true).map_err(candle_core::Error::msg)?;
    
    // 루프가 끝난 후, 미처 출력되지 못한 마지막 조각이 있다면 마저 출력
    if generated_text.len() > prev_text_len {
        print!("{}", &generated_text[prev_text_len..]);
    }
    // 루프가 모두 끝난 후 무조건 버퍼를 한 번 비워줍니다.
    std::io::stdout().flush().unwrap();
    println!(); // 마지막에 콘솔 줄바꿈 추가

    Ok(generated_text)
}

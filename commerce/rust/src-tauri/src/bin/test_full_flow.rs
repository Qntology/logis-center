use anyhow::Result;
use candle_core::{Device, DType};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

// Correct crate name from Cargo.toml
use tauri_app_lib::models::qwen3vl::generate::Qwen3VLGenerateModel;

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    println!("[TEST] Using device: {:?}", device);

    // Path relative to src-tauri
    let model_path = "models/Qwen3-0.6B-Instruct-gguf";
    
    println!("[TEST] Initializing 0.6B Model in Baking Mode...");
    let mut model = Qwen3VLGenerateModel::init_with_config(
        model_path,
        None,
        None,
        Some(&device),
        0,
        Some(&device),
        0,
        Some(DType::BF16),
        None,
        true, // force_text_only
        true, // baking_only
        false
    )?;

    println!("[TEST] Model Loaded: {}", model.model_name);

    // Generate simulated PUG content (Approx 10k tokens for testing)
    let large_text = "tag.class | some text node content inside pug template structure ".repeat(2000); 
    println!("[TEST] Input Text Length: {} chars", large_text.len());

    let cancel_flag = Arc::new(AtomicBool::new(false));
    
    println!("[TEST] Starting Prefill (Real Rust Logic)...");
    let start = std::time::Instant::now();
    
    // Test the prefill_text_only method which uses our Chunked Masking
    match model.prefill_text_only(
        &large_text,
        Some(cancel_flag),
        None,
        Some(Path::new("tmp/kv/test_session"))
    ) {
        Ok(_) => {
            println!("[TEST] SUCCESS! Prefill completed in {:?}", start.elapsed());
            println!("[TEST] Final KV Length: {}", model.get_kv_len());
        },
        Err(e) => {
            println!("[TEST] FAILED during prefill: {:?}", e);
        }
    }

    Ok(())
}
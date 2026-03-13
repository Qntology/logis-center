use anyhow::{Result, anyhow};
use serde_json;
use candle_core::{DType, Device, Tensor, D};
use tauri_app_lib::models::qwen3_5::generate::Qwen3_5GenerateModel;
use tauri_app_lib::openai_types::ChatCompletionParameters;

pub async fn run_test() -> Result<()> {
    println!("🚀 Starting Qwen3.5 Hybrid Engine Final Test (BF16)...");

    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    println!("💻 Using device: {:?}", device);

    let model_path = "models/Qwen3.5-0.8B-Split".to_string();
    println!("📂 Loading model from {}...", model_path);

    // Using BF16 for better precision with Qwen3.5
    let mut model = Qwen3_5GenerateModel::init(&model_path, Some(&device), Some(DType::BF16), true)?;

    println!("\n✨ Hybrid Model initialized successfully (BF16).");

    let message_json = r#"{
        "model": "qwen3.5",
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of France?"
            }
        ],
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9
    }"#;

    let params: ChatCompletionParameters = serde_json::from_str(message_json)?;
    println!("❓ Asking: What is the capital of France?");

    let result = model.generate(params, None, None, None).await?;
    println!("\n\n✅ Final Result:\n{}", result);

    println!("\n\n🤖 Final Result:\n{}", result);
    println!("\n✅ Test completed successfully!");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    run_test().await
}

use anyhow::Result;
use candle_core::{DType, Device};
// [FIX] Use the correct library crate name 'tauri_app_lib'
use tauri_app_lib::models::qwen3_5::generate::Qwen3_5GenerateModel;
use tauri_app_lib::openai_types::ChatCompletionParameters;

pub async fn run_test() -> Result<()> {
    println!("🚀 Starting Qwen3.5 Hybrid Engine Test...");

    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    println!("💻 Using device: {:?}", device);

    // Use simple relative path
    let model_path = "src-tauri/models/Qwen3.5-0.8B-Full".to_string();

    println!("📂 Loading model from {}...", model_path);
    
    // Using BF16 for original model for better precision
    let mut model = Qwen3_5GenerateModel::init(&model_path, Some(&device), Some(DType::BF16), true)?;

    println!("\n✨ Hybrid Model initialized successfully.");

    let message_json = r#"{
        "model": "qwen3.5",
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ],
        "max_tokens": 64,
        "temperature": 0.0,
        "top_p": 1.0
    }"#;

    let mes: ChatCompletionParameters = serde_json::from_str(message_json)?;
    println!("❓ Asking: Hello, how are you?");

    let result = model.generate(mes, None, None, None).await?;

    println!("\n\n🤖 Result:\n{}", result);
    println!("\n✅ Test completed successfully!");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    run_test().await
}

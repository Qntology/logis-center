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

    // Ensure we use the correct model path without extra 'Q'
    let base_path = std::fs::canonicalize("src-tauri/models").or_else(|_| std::fs::canonicalize("models"))?;
    let model_path = base_path.join("Qwen3.5-0.8B-Split").to_string_lossy().to_string();

    println!("📂 Loading model from {}...", model_path);
    
    let mut model = Qwen3_5GenerateModel::init(&model_path, Some(&device), Some(DType::F16), true)?;

    println!("\n✨ Hybrid Model initialized successfully.");

    let message_json = r#"{
        "model": "qwen3.5",
        "messages": [
            {
                "role": "user",
                "content": "Explain Newton's first law in one sentence."
            }
        ],
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9
    }"#;

    let mes: ChatCompletionParameters = serde_json::from_str(message_json)?;
    println!("❓ Asking: Explain Newton's first law in one sentence.");

    let result = model.generate(mes, None, None, None).await?;

    println!("\n\n🤖 Result:\n{}", result);
    println!("\n✅ Test completed successfully!");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    run_test().await
}

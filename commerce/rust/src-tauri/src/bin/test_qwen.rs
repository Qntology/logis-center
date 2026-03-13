use anyhow::{Result, anyhow};
use serde_json;
use candle_core::{DType, Device, Tensor, D};
use tauri_app_lib::models::qwen3_5::generate::Qwen3_5GenerateModel;
use tauri_app_lib::openai_types::ChatCompletionParameters;

pub async fn run_test() -> Result<()> {
    println!("🚀 Starting Qwen3.5 Hybrid Engine Final Test (F16)...");

    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    println!("💻 Using device: {:?}", device);

    let model_path = "models/Qwen3.5-0.8B-Full".to_string();
    println!("📂 Loading model from {}...", model_path);
    
    // Try F16 for better compatibility with some CUDA kernels/hardware
    let mut model = Qwen3_5GenerateModel::init(&model_path, Some(&device), Some(DType::F16), true)?;

    println!("\n✨ Hybrid Model initialized successfully (F16).");

    let message_json = r#"{
        "model": "qwen3.5",
        "messages": [
            {
                "role": "user",
                "content": "Hello!"
            }
        ],
        "max_tokens": 20,
        "temperature": 0.0,
        "top_p": 1.0
    }"#;

    let params: ChatCompletionParameters = serde_json::from_str(message_json)?;
    println!("❓ Asking: Hello!");

    let result = model.generate(params, None, None, None).await?;

    println!("\n\n🤖 Final Result:\n{}", result);
    println!("\n✅ Test completed successfully!");

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    run_test().await
}

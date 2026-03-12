// src-tauri/src/bin/test_qwen.rs
use anyhow::Result;
use candle_core::{DType, Device};
use tauri_app_lib::models::qwen3_5::Qwen3_5GenerateModel;
use tauri_app_lib::openai_types::{ChatCompletionParameters, ChatCompletionMessage};

#[tokio::main]
pub async fn main() -> Result<()> {
    println!("🚀 Starting Qwen3.5 Hybrid Engine Test...");
    println!("📂 Current working directory: {:?}", std::env::current_dir()?);

    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    println!("💻 Using device: {:?}", device);

    // Find model path by checking common locations
    let base_path = if std::path::Path::new("src-tauri/models").exists() {
        std::path::PathBuf::from("src-tauri/models")
    } else if std::path::Path::new("models").exists() {
        std::path::PathBuf::from("models")
    } else {
        anyhow::bail!("Could not find models directory in 'src-tauri/models' or 'models'");
    };

    let model_path = base_path.join("Qwen3.5-0.8B-Full").to_string_lossy().to_string();
    println!("📂 Loading model from {}...", model_path);

    // Using BF16 for original model for better precision
    let mut model = Qwen3_5GenerateModel::init(&model_path, Some(&device), Some(DType::BF16), true)?;

    let params = ChatCompletionParameters {
        messages: vec![
            ChatCompletionMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
            },
            ChatCompletionMessage {
                role: "user".to_string(),
                content: "Tell me about Newton's Laws of Motion.".to_string(),
            },
        ],
        temperature: Some(0.7),
        max_tokens: Some(128),
        ..Default::default()
    };

    println!("\n🤖 Model Response:");
    println!("-------------------");
    let response = model.generate(params, None, None, None).await?;
    println!("\n-------------------");
    println!("✅ Test completed successfully!");

    Ok(())
}

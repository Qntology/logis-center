use anyhow::Result;
use candle_core::{Device, DType};
use std::path::Path;

/*
 * [TEST BINARY] Qwen3.5-0.8B Standalone Test
 * Purpose: Verify if 1+1=? result can be generated without 'unexpected dtype' error.
 */

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Starting Qwen3.5 Standalone Test...");

    // Adjust path based on execution location (assumes running from workspace root or src-tauri)
    let model_path = if Path::new("models/Qwen3.5-0.8B-Split").exists() {
        "models/Qwen3.5-0.8B-Split"
    } else {
        "src-tauri/models/Qwen3.5-0.8B-Split"
    };

    if !Path::new(model_path).exists() {
        println!("❌ Model path not found: {}", model_path);
        return Ok(());
    }

    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("💻 Using device: {:?}", device);

    // [INIT] Initialize Generator
    // Note: We use the existing Qwen3_5GenerateModel from the crate
    use tauri_app_lib::models::qwen35::generate::Qwen3_5GenerateModel;
    use tauri_app_lib::openai_types::{ChatCompletionParameters, ChatCompletionRequestMessage, ChatCompletionRequestUserMessage, ChatCompletionRequestUserMessageContent};

    let mut model = Qwen3_5GenerateModel::init(model_path, Some(&device), Some(DType::BF16), true)?;
    println!("✅ Model initialized successfully.");

    let params = ChatCompletionParameters {
        messages: vec![
            ChatCompletionRequestMessage::User(ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text("1+1=?".to_string()),
                name: None,
            }),
        ],
        temperature: Some(0.1),
        max_tokens: Some(50),
        ..Default::default()
    };

    println!("🤔 Asking: 1+1=?");
    let response = model.generate(params, None).await?;

    println!("\n✨ Result: {}", response);
    println!("\n✅ Test completed successfully!");

    Ok(())
}

use anyhow::Result;
use candle_core::{Device, DType};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Starting Qwen3.5 Hybrid Engine Test...");

    let model_path = if Path::new("models/Qwen3.5-0.8B-Split").exists() {
        "models/Qwen3.5-0.8B-Split"
    } else {
        "src-tauri/models/Qwen3.5-0.8B-Split"
    };

    let device = Device::new_cuda(1)
        .or_else(|_| Device::new_cuda(0))
        .unwrap_or(Device::Cpu);
    println!("💻 Using device: {:?}", device);

    use tauri_app_lib::models::qwen35::generate::Qwen3_5GenerateModel;
    use tauri_app_lib::openai_types::ChatCompletionParameters;

    let mut model_files = vec![];
    for entry in std::fs::read_dir(model_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("st") {
            model_files.push(path.to_str().unwrap().to_string());
        }
    }
    
    if model_files.is_empty() {
        anyhow::bail!("No .st files found in {}", model_path);
    }

    println!("📚 Loading {} model files...", model_files.len());
    let mut model = Qwen3_5GenerateModel::init_with_files(&model_files, model_path, Some(&device), Some(DType::F16))?;
    
    println!("✅ Hybrid Model initialized successfully.");

    let message_json = r#"
    {
        "model": "qwen3.5",
        "messages": [
            {
                "role": "user",
                "content": "1+1="
            }
        ],
        "max_tokens": 10,
        "temperature": 0.0,
        "top_p": 1.0
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message_json)?;
    
    println!("🤔 Asking: 1+1=?");
    let response = model.generate(mes, None, None, None).await?;

    println!("\n✨ Result: {}", response);
    println!("\n✅ Test completed successfully!");

    Ok(())
}

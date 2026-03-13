use vllm_rs::api::{EngineBuilder, ModelRepo};
use vllm_rs::server::{ChatMessage, MessageContentType};
use vllm_rs::utils::{config::SamplingParams, log_throughput};

fn main() -> anyhow::Result<()> {
    // Need to leak the string to satisfy &'static str requirement in ModelRepo
    let model_path: &'static str = Box::leak("../../../src-tauri/models/Qwen3.5-0.8B-Full".to_string().into_boxed_str());
    
    let mut engine = EngineBuilder::new(ModelRepo::ModelPath(model_path)).build()?;

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(MessageContentType::PureText("Explain Newton's first law in one sentence.".to_string())),
        tool_call_id: None,
        tool_calls: None,
    }];

    let params = SamplingParams::default();
    // generate now requires tools argument
    let output = engine.generate(params, messages, Vec::new())?;
    println!("\n\n{}", output.decode_output);

    log_throughput(&vec![output]);
    Ok(())
}

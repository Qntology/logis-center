use crate::openai_types::ChatCompletionParameters;
use anyhow::{Result, anyhow};
use minijinja::{Environment, Value as MiniJinjaValue, context};

use crate::utils::string_to_static_str;

pub fn get_template(path: String) -> Result<String> {
    let tokenizer_config_file = std::path::Path::new(&path).join("tokenizer_config.json");
    if !tokenizer_config_file.exists() {
        return Err(anyhow!(
            "tokenizer_config.json not exists in model path: {:?}",
            tokenizer_config_file
        ));
    }
    let tokenizer_config: serde_json::Value = 
        serde_json::from_slice(&std::fs::read(&tokenizer_config_file)?)
            .map_err(|e| anyhow!(format!("load tokenizer_config file error:{}",e)))
            .unwrap();
    let chat_template = tokenizer_config["chat_template"]
        .as_str()
        .ok_or(anyhow!(format!("chat_template to str error")))?;
    
    let fixed_template = chat_template
        // 변수명이나 따옴표 종류에 상관없이 모두 잡아내도록 변경
        .replace(".startswith(", " is startingwith(")
        .replace(".endswith(", " is endingwith(")
        // 아래는 기존 코드 유지
        .replace(
            "content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')",
            "((content | split('</think>'))[0] | rstrip('\n') | split('<think>'))[-1] | lstrip('\n')", 
        )
        .replace(
            "content.split('</think>')[-1].lstrip('\n')",
            "(content | split('</think>'))[-1] | lstrip('\n')", 
        )
        .replace(
            "reasoning_content.strip('\n')",
            "reasoning_content | strip('\n')", 
        )
        .replace(
            "content.lstrip('\n')",
            "content | lstrip('\n')", 
        );
        
    Ok(fixed_template)
}

pub struct ChatTemplate {
    env: Environment<'static>,
}

impl ChatTemplate {
    pub fn init(path: &str) -> Result<Self> {
        let path: String = path.to_string();
        if !std::path::Path::new(&path).exists() {
            return Err(anyhow!("model path not found"));
        }
        let template = match get_template(path.clone()) {
            Ok(template) => template,
            Err(e) => {
                let jinja_path = std::path::Path::new(&path).join("chat_template.jinja");
                if !jinja_path.exists() {
                    return Err(anyhow!(
                        "get_template err {e} and chat_template.jinja not found at {:?}",
                        jinja_path
                    ));
                }
                std::fs::read_to_string(&jinja_path)
                    .map_err(|e| anyhow!("Failed to read chat_template.jinja: {}", e))?
            }
        };
        let template = string_to_static_str(template);
        let mut env = Environment::new();
        env.add_filter("tojson", |v: MiniJinjaValue| {
            serde_json::to_string(&v).unwrap()
        });

        env.add_filter("split", |s: String, delimiter: String| {
            s.split(&delimiter)
                .map(|s| s.to_string())
                .collect::<Vec<String>>()
        });

        env.add_filter("lstrip", |s: String, chars: Option<String>| match chars {
            Some(chars_str) => s.trim_start_matches(chars_str.as_str()).to_string(),
            None => s.trim_start().to_string(),
        });

        env.add_filter("rstrip", |s: String, chars: Option<String>| match chars {
            Some(chars_str) => s.trim_end_matches(chars_str.as_str()).to_string(),
            None => s.trim_end().to_string(),
        });
        
        let _ = env.add_template("chat", template);

        Ok(Self { env })
    }

    pub fn apply_chat_template(&self, messages: &ChatCompletionParameters) -> Result<String> {
        // [FIX] Flatten User Message Content Arrays to String for Jinja Compatibility
        // Qwen chat template expects 'content' to be a string, not an array.
        let mut flattened_messages = messages.messages.clone();
        for msg in &mut flattened_messages {
            if let crate::openai_types::ChatCompletionRequestMessage::User(user_msg) = msg {
                if let crate::openai_types::ChatCompletionRequestUserMessageContent::Array(parts) = &user_msg.content {
                    let mut text_content = String::new();
                    
                    for part in parts {
                        if let crate::openai_types::ChatCompletionRequestMessageContentPart::Text(text_part) = part {
                            text_content.push_str(&text_part.text);
                        } else if let crate::openai_types::ChatCompletionRequestMessageContentPart::ImageURL(_) = part {
                            // Qwen-VL 모델이 이미지를 인식할 수 있도록 플레이스홀더를 주입합니다.
                            text_content.push_str("<|vision_start|><|image_pad|><|vision_end|>\n");
                        }
                    }
                    
                    user_msg.content = crate::openai_types::ChatCompletionRequestUserMessageContent::Text(text_content);
                }
            }
        }

        let context = context! {
            messages => flattened_messages,
            tools => &messages.tools.as_ref(),
            add_generation_prompt => true,
        };
        let template = self
            .env
            .get_template("chat")
            .map_err(|e| anyhow!(format!("render template error {}", e)))?;
        let message_str = template
            .render(context)
            .map_err(|e| anyhow!(format!("render template error {}", e)))?;
        Ok(message_str)
    }
}

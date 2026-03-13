use crate::utils::chat_template::Message;
use crate::utils::image::{ImageData, ImageProcessConfig, ImageProcessTrait};
use crate::utils::config::EngineConfig;
use crate::core::engine::LLMEngine;
use candle_core::{Tensor, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use parking_lot::RwLock;
use crate::utils::image::{load_image_from_base64, load_image_from_url, get_tensor_raw_data, compute_tokens_per_image};

pub const IMAGE_PLACEHOLDER: &str = "<|image_pad|>";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: MessageContentType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::tools::ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum MessageContentType {
    PureText(String),
    Single(MessageContent),
    Multi(Vec<MessageContent>),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MessageContent {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ImageUrl,
    },
    ImageBase64 {
        image_base64: String,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ImageUrl {
    pub url: String,
    pub detail: Option<String>,
}

impl ImageUrl {
    pub fn url(&self) -> &str {
        &self.url
    }
}

pub fn build_messages_and_images(
    messages: &[ChatMessage],
    img_cfg: Option<&ImageProcessConfig>,
) -> Result<(Vec<Message>, Option<ImageData>)> {
    use crate::models::qwen3_vl::input::Qwen3VLImageProcessor;
    use crate::utils::config::ModelType;
    use crate::utils::image::ImageProcessor;

    let mut processor: Option<Box<dyn ImageProcessTrait + Send>> = if let Some(cfg) = img_cfg {
        if matches!(cfg.model_type, ModelType::Qwen3VL) {
            Some(Box::new(Qwen3VLImageProcessor::default(cfg)))
        } else {
            Some(Box::new(ImageProcessor::new(cfg)))
        }
    } else {
        None
    };

    let mut images: Vec<(Tensor, Vec<(usize, usize)>)> = vec![];

    let messages: Vec<Message> = messages
        .iter()
        .map(|m| convert_chat_message(m, &mut processor, &mut images))
        .collect::<Result<Vec<_>>>()?;

    let image_data = if !images.is_empty() && img_cfg.is_some() {
        let mut image_sizes = Vec::new();
        let mut image_tensors = Vec::new();
        for (t, s) in &images {
            image_tensors.push(t);
            image_sizes.extend(s);
        }
        let images_tensor = Tensor::cat(&image_tensors, 0)?;
        let (images_raw, images_shape) = get_tensor_raw_data(&images_tensor)?;
        crate::log_info!(
            "{} images detected in the chat message, combined image shape {:?}",
            images_shape[0],
            images_shape
        );
        let cfg = img_cfg.unwrap();
        let tokens_per_image = compute_tokens_per_image(cfg, &image_sizes);
        Some(ImageData {
            raw: images_raw,
            shape: images_shape,
            patches: image_sizes,
            image_idx: 0,
            image_token_offset: 0,
            tokens_per_image,
            image_token_id: cfg.image_token_id,
        })
    } else {
        None
    };

    Ok((messages, image_data))
}

fn convert_chat_message(
    m: &ChatMessage,
    processor: &mut Option<Box<dyn ImageProcessTrait + Send>>,
    images_tensors: &mut Vec<(Tensor, Vec<(usize, usize)>)>,
) -> Result<Message> {
    let role = m.role.clone();
    let mut prompt = String::new();
    let mut images = Vec::new();

    match &m.content {
        MessageContentType::PureText(text) => {
            prompt.push_str(text);
        }
        MessageContentType::Single(item) => {
            append_message_item(item, &mut prompt, &mut images)?;
        }
        MessageContentType::Multi(items) => {
            for item in items {
                append_message_item(item, &mut prompt, &mut images)?;
            }
        }
    }

    if !images.is_empty() && processor.is_some() {
        if let Some(processor) = processor.as_mut() {
            let (images_tensor, image_sizes) = processor.process_inputs(&mut prompt, &images)?;
            images_tensors.push((images_tensor, image_sizes));
        }
    }

    let mut message = Message::new(role, prompt.trim().to_owned(), images.len());
    
    if let Some(calls) = &m.tool_calls {
        let template_calls: Vec<serde_json::Value> = calls.iter().map(to_template_tool_call).collect();
        message.set_tool_calls(template_calls);
    }
    
    if let Some(id) = &m.tool_call_id {
        message.set_tool_call_id(id.clone());
    }

    Ok(message)
}

fn append_message_item(
    item: &MessageContent,
    prompt: &mut String,
    images: &mut Vec<image::DynamicImage>,
) -> Result<()> {
    match item {
        MessageContent::Text { text } => {
            prompt.push_str(text);
        }
        MessageContent::ImageUrl { image_url } => {
            let url = image_url.url();
            let img = if url.starts_with("data:") {
                let img = load_image_from_base64(url)?;
                img
            } else {
                let img = load_image_from_url(url)?;
                img
            };
            prompt.push_str(IMAGE_PLACEHOLDER);
            images.push(img);
        }
        MessageContent::ImageBase64 { image_base64 } => {
            let img = load_image_from_base64(image_base64)?;
            prompt.push_str(IMAGE_PLACEHOLDER);
            images.push(img);
        }
    }
    Ok(())
}

fn to_template_tool_call(call: &crate::tools::ToolCall) -> serde_json::Value {
    let args = parse_template_tool_arguments(call.function.arguments.as_deref());

    serde_json::json!({
        "id": call.id.clone(),
        "type": call.tool_type.clone(),
        "function": {
            "name": call.function.name.clone(),
            "arguments": args
        }
    })
}

fn parse_template_tool_arguments(arguments: Option<&str>) -> serde_json::Value {
    let Some(raw) = arguments.map(str::trim).filter(|s| !s.is_empty()) else {
        return serde_json::json!({});
    };

    match serde_json::from_str::<serde_json::Value>(raw).ok() {
        Some(serde_json::Value::Object(obj)) => serde_json::Value::Object(obj),
        Some(serde_json::Value::String(inner)) => {
            match serde_json::from_str::<serde_json::Value>(inner.trim()).ok() {
                Some(serde_json::Value::Object(obj)) => serde_json::Value::Object(obj),
                _ => serde_json::json!({}),
            }
        }
        _ => serde_json::json!({}),
    }
}

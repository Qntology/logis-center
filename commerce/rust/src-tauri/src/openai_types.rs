use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct ChatCompletionParameters {
    pub messages: Vec<ChatCompletionRequestMessage>,
    pub model: String,
    pub frequency_penalty: Option<f32>,
    pub logit_bias: Option<Value>,
    pub logprobs: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub max_tokens: Option<u32>,
    pub n: Option<u32>,
    pub presence_penalty: Option<f32>,
    pub response_format: Option<Value>,
    pub seed: Option<u32>,
    pub stop: Option<Vec<String>>,
    pub stream: Option<bool>,
    pub stream_options: Option<Value>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub tools: Option<Vec<ChatCompletionTool>>,
    pub tool_choice: Option<Value>,
    pub user: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "role")]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionRequestMessage {
    System(ChatCompletionRequestSystemMessage),
    User(ChatCompletionRequestUserMessage),
    Assistant(ChatCompletionRequestAssistantMessage),
    Tool(ChatCompletionRequestToolMessage),
    Function(ChatCompletionRequestFunctionMessage),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequestSystemMessage {
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequestUserMessage {
    pub content: ChatCompletionRequestUserMessageContent,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum ChatCompletionRequestUserMessageContent {
    Text(String),
    Array(Vec<ChatCompletionRequestMessageContentPart>),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum ChatCompletionRequestMessageContentPart {
    Text(ChatCompletionRequestMessageContentPartText),
    ImageURL(ChatCompletionRequestMessageContentPartImage),
    #[serde(rename = "video_url")]
    VideoURL(ChatCompletionRequestMessageContentPartVideo),
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequestMessageContentPartText {
    pub text: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequestMessageContentPartImage {
    pub image_url: ImageURL,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequestMessageContentPartVideo {
    pub video_url: VideoURL,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ImageURL {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct VideoURL {
    pub url: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, Default)]
pub struct ChatCompletionRequestAssistantMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ChatCompletionMessageToolCall>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequestToolMessage {
    pub content: String,
    pub tool_call_id: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatCompletionRequestFunctionMessage {
    pub content: Option<String>,
    pub name: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatCompletionTool {
    #[serde(rename = "type")]
    pub r#type: String, // "function"
    pub function: FunctionObject,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FunctionObject {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatCompletionMessageToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub r#type: String, // "function"
    pub function: FunctionCall,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

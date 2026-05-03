//! Canned chat-template registry covering the five major open-weight
//! instruction-tuned families.
//!
//! The templates are modelled after the official `tokenizer_config.json`
//! `chat_template` fields for each family as of late 2024 / early 2025.
//! They are intentionally small, hand-transcribed Jinja-lite snippets rather
//! than direct copies of the vendor templates so that they render
//! deterministically on this crate's minimal evaluator.
//!
//! | Family   | Roles supported                  | Reference tag(s)                            |
//! |----------|----------------------------------|---------------------------------------------|
//! | ChatML   | system / user / assistant / tool | `<|im_start|>` / `<|im_end|>`               |
//! | Llama3   | system / user / assistant        | `<|start_header_id|>` / `<|eot_id|>`        |
//! | Mistral  | user / assistant                 | `[INST]` / `[/INST]`                        |
//! | Gemma    | user / assistant                 | `<start_of_turn>` / `<end_of_turn>`         |
//! | Qwen     | system / user / assistant        | `<|im_start|>` / `<|im_end|>` + `<|endoftext|>` |

use crate::{error::TokenizerResult, utils::render_template};

// ── ChatMessage ──────────────────────────────────────────────────────────────

/// A single chat-turn used by [`ChatTemplateKind::render`].
///
/// The lifetime ties the message to the caller's storage so no extra
/// allocations are needed during rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatMessage<'a> {
    /// Conventional role: `"system"`, `"user"`, `"assistant"`, or `"tool"`.
    pub role: &'a str,
    /// Message content (raw, not pre-tokenized).
    pub content: &'a str,
}

impl<'a> ChatMessage<'a> {
    /// Convenience constructor.
    pub fn new(role: &'a str, content: &'a str) -> Self {
        Self { role, content }
    }

    /// Short-hand for a user message.
    pub fn user(content: &'a str) -> Self {
        Self::new("user", content)
    }

    /// Short-hand for an assistant message.
    pub fn assistant(content: &'a str) -> Self {
        Self::new("assistant", content)
    }

    /// Short-hand for a system message.
    pub fn system(content: &'a str) -> Self {
        Self::new("system", content)
    }
}

// ── ChatTemplateKind ─────────────────────────────────────────────────────────

/// Identifies one of the built-in chat-template families.
///
/// Use [`Self::render`] to format a sequence of [`ChatMessage`]s into the
/// canonical prompt string for that family.  The returned string is ready to
/// be passed to [`crate::OxiTokenizer::encode`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ChatTemplateKind {
    /// OpenAI-style ChatML: `<|im_start|>role\ncontent<|im_end|>\n…`.
    ChatML,
    /// Llama-3 Instruct: `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`.
    Llama3,
    /// Mistral Instruct: `<s>[INST] user [/INST] assistant </s>`.
    Mistral,
    /// Gemma Instruct: `<start_of_turn>role\ncontent<end_of_turn>`.
    Gemma,
    /// Qwen series: Same tags as ChatML plus `<|endoftext|>` trailer.
    Qwen,
}

impl ChatTemplateKind {
    /// Return the raw Jinja-lite template string for this family.
    pub fn template(&self) -> &'static str {
        match self {
            Self::ChatML => CHATML_TEMPLATE,
            Self::Llama3 => LLAMA3_TEMPLATE,
            Self::Mistral => MISTRAL_TEMPLATE,
            Self::Gemma => GEMMA_TEMPLATE,
            Self::Qwen => QWEN_TEMPLATE,
        }
    }

    /// Render a list of [`ChatMessage`]s into a prompt string.
    pub fn render(&self, messages: &[ChatMessage<'_>]) -> String {
        let pairs: Vec<(&str, &str)> = messages.iter().map(|m| (m.role, m.content)).collect();
        render_template(self.template(), &pairs)
    }

    /// Render messages and append an assistant-generation prompt (the opener
    /// that tells the model "your turn").  For ChatML this is
    /// `<|im_start|>assistant\n`; for Llama-3 `<|start_header_id|>assistant<|end_header_id|>\n\n`.
    pub fn render_with_generation_prompt(&self, messages: &[ChatMessage<'_>]) -> String {
        let mut out = self.render(messages);
        out.push_str(self.generation_prompt());
        out
    }

    /// The tag that follows the last user message to invite an assistant
    /// response.  Empty for families that don't need one.
    pub fn generation_prompt(&self) -> &'static str {
        match self {
            Self::ChatML | Self::Qwen => "<|im_start|>assistant\n",
            Self::Llama3 => "<|start_header_id|>assistant<|end_header_id|>\n\n",
            Self::Mistral => "",
            Self::Gemma => "<start_of_turn>model\n",
        }
    }

    /// Tokenize this family's prompt directly with a tokenizer.
    ///
    /// Equivalent to `tok.encode(&kind.render(msgs))` but kept as a helper
    /// method for symmetry with the HF `apply_chat_template` API.
    pub fn encode(
        &self,
        tokenizer: &crate::OxiTokenizer,
        messages: &[ChatMessage<'_>],
    ) -> TokenizerResult<Vec<u32>> {
        tokenizer.encode(&self.render(messages))
    }

    /// List all kinds that this crate knows about.  Useful for testing and
    /// for building UI pickers.
    pub fn all() -> &'static [ChatTemplateKind] {
        &[
            Self::ChatML,
            Self::Llama3,
            Self::Mistral,
            Self::Gemma,
            Self::Qwen,
        ]
    }

    /// Infer a template kind from a model name heuristic.  Returns `None` if
    /// no family is recognised.
    pub fn infer_from_name(name: &str) -> Option<Self> {
        let n = name.to_ascii_lowercase();
        if n.contains("llama-3") || n.contains("llama3") {
            Some(Self::Llama3)
        } else if n.contains("mistral") {
            Some(Self::Mistral)
        } else if n.contains("gemma") {
            Some(Self::Gemma)
        } else if n.contains("qwen") {
            Some(Self::Qwen)
        } else if n.contains("chatml") {
            Some(Self::ChatML)
        } else {
            None
        }
    }
}

// ── Template strings ─────────────────────────────────────────────────────────

const CHATML_TEMPLATE: &str =
    "{% for message in messages %}<|im_start|>{{ role }}\n{{ content }}<|im_end|>\n{% endfor %}";

const LLAMA3_TEMPLATE: &str = concat!(
    "<|begin_of_text|>",
    "{% for message in messages %}",
    "<|start_header_id|>{{ role }}<|end_header_id|>\n\n",
    "{{ content }}<|eot_id|>",
    "{% endfor %}"
);

// Mistral needs an if/else to distinguish user from assistant turns.
// The minimal evaluator supports `{% if role == "user" %} … {% else %} … {% endif %}`.
const MISTRAL_TEMPLATE: &str = concat!(
    "{% for message in messages %}",
    "{% if role == \"user\" %}<s>[INST] {{ content }} [/INST]{% else %} {{ content }}</s>{% endif %}",
    "{% endfor %}"
);

const GEMMA_TEMPLATE: &str = concat!(
    "{% for message in messages %}",
    "<start_of_turn>{{ role }}\n{{ content }}<end_of_turn>\n",
    "{% endfor %}"
);

const QWEN_TEMPLATE: &str =
    "{% for message in messages %}<|im_start|>{{ role }}\n{{ content }}<|im_end|>\n{% endfor %}";

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_kinds_yield_a_template() {
        for k in ChatTemplateKind::all() {
            assert!(!k.template().is_empty(), "template for {k:?} empty");
        }
    }

    #[test]
    fn chatml_renders_basic() {
        let out = ChatTemplateKind::ChatML.render(&[ChatMessage::user("hi")]);
        assert!(out.contains("<|im_start|>user"));
        assert!(out.contains("hi"));
        assert!(out.contains("<|im_end|>"));
    }

    #[test]
    fn llama3_renders_basic() {
        let out = ChatTemplateKind::Llama3.render(&[ChatMessage::user("hi")]);
        assert!(out.contains("<|begin_of_text|>"));
        assert!(out.contains("<|start_header_id|>user<|end_header_id|>"));
        assert!(out.contains("<|eot_id|>"));
    }

    #[test]
    fn mistral_renders_basic() {
        let out = ChatTemplateKind::Mistral
            .render(&[ChatMessage::user("hi"), ChatMessage::assistant("there")]);
        assert!(out.contains("[INST] hi [/INST]"));
        assert!(out.contains("there"));
    }

    #[test]
    fn gemma_renders_basic() {
        let out = ChatTemplateKind::Gemma.render(&[ChatMessage::user("hi")]);
        assert!(out.contains("<start_of_turn>user"));
        assert!(out.contains("<end_of_turn>"));
    }

    #[test]
    fn qwen_renders_basic() {
        let out = ChatTemplateKind::Qwen.render(&[ChatMessage::user("hi")]);
        assert!(out.contains("<|im_start|>user"));
        assert!(out.contains("<|im_end|>"));
    }

    #[test]
    fn generation_prompt_chatml() {
        let p = ChatTemplateKind::ChatML.generation_prompt();
        assert!(p.contains("assistant"));
    }

    #[test]
    fn render_with_generation_prompt() {
        let out =
            ChatTemplateKind::ChatML.render_with_generation_prompt(&[ChatMessage::user("hi")]);
        assert!(out.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn infer_from_name_known() {
        assert_eq!(
            ChatTemplateKind::infer_from_name("Qwen3-1.7B"),
            Some(ChatTemplateKind::Qwen)
        );
        assert_eq!(
            ChatTemplateKind::infer_from_name("Meta-Llama-3-8B-Instruct"),
            Some(ChatTemplateKind::Llama3)
        );
        assert_eq!(
            ChatTemplateKind::infer_from_name("mistral-7b"),
            Some(ChatTemplateKind::Mistral)
        );
        assert_eq!(
            ChatTemplateKind::infer_from_name("gemma-2b"),
            Some(ChatTemplateKind::Gemma)
        );
    }

    #[test]
    fn infer_from_name_unknown() {
        assert_eq!(ChatTemplateKind::infer_from_name("bert-base"), None);
    }

    #[test]
    fn encode_works_with_stub() {
        let tok = crate::OxiTokenizer::char_level_stub(256);
        let ids = ChatTemplateKind::ChatML
            .encode(&tok, &[ChatMessage::user("hi")])
            .expect("encode ok");
        assert!(!ids.is_empty());
    }

    #[test]
    fn chat_message_constructors() {
        let u = ChatMessage::user("x");
        assert_eq!(u.role, "user");
        let a = ChatMessage::assistant("y");
        assert_eq!(a.role, "assistant");
        let s = ChatMessage::system("z");
        assert_eq!(s.role, "system");
    }
}

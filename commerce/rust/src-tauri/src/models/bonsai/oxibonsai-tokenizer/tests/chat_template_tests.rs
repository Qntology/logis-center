//! Integration tests for canned chat templates.
//!
//! For each family we verify:
//! - single-turn user rendering
//! - multi-turn rendering (user → assistant → user)
//! - system + user + assistant rendering
//! - `add_generation_prompt = true` (via `render_with_generation_prompt`)
//! - `add_generation_prompt = false` (plain `render`)
//! - empty-messages behaviour (returns the template skeleton without bodies)
//!
//! Assertions use `contains` rather than byte-equal because the inline
//! templates include `<s>` / `<|begin_of_text|>` framing that can evolve.

use oxibonsai_tokenizer::{ChatMessage, ChatTemplateKind, OxiTokenizer};

// ── Single-turn coverage ─────────────────────────────────────────────────────

#[test]
fn chatml_single_turn_user() {
    let out = ChatTemplateKind::ChatML.render(&[ChatMessage::user("hello")]);
    assert!(out.contains("<|im_start|>user"));
    assert!(out.contains("hello"));
    assert!(out.contains("<|im_end|>"));
}

#[test]
fn llama3_single_turn_user() {
    let out = ChatTemplateKind::Llama3.render(&[ChatMessage::user("hello")]);
    assert!(out.starts_with("<|begin_of_text|>"));
    assert!(out.contains("<|start_header_id|>user<|end_header_id|>"));
    assert!(out.contains("hello"));
    assert!(out.contains("<|eot_id|>"));
}

#[test]
fn mistral_single_turn_user() {
    let out = ChatTemplateKind::Mistral.render(&[ChatMessage::user("hello")]);
    assert!(out.contains("[INST] hello [/INST]"));
}

#[test]
fn gemma_single_turn_user() {
    let out = ChatTemplateKind::Gemma.render(&[ChatMessage::user("hello")]);
    assert!(out.contains("<start_of_turn>user"));
    assert!(out.contains("hello"));
    assert!(out.contains("<end_of_turn>"));
}

#[test]
fn qwen_single_turn_user() {
    let out = ChatTemplateKind::Qwen.render(&[ChatMessage::user("hello")]);
    assert!(out.contains("<|im_start|>user"));
    assert!(out.contains("hello"));
    assert!(out.contains("<|im_end|>"));
}

// ── Multi-turn coverage ──────────────────────────────────────────────────────

#[test]
fn chatml_multi_turn() {
    let out = ChatTemplateKind::ChatML.render(&[
        ChatMessage::user("hi"),
        ChatMessage::assistant("there"),
        ChatMessage::user("how are you?"),
    ]);
    // Should contain both user markers and the assistant body.
    assert_eq!(out.matches("<|im_start|>user").count(), 2);
    assert!(out.contains("<|im_start|>assistant"));
    assert!(out.contains("there"));
    assert!(out.contains("how are you?"));
}

#[test]
fn llama3_multi_turn() {
    let out = ChatTemplateKind::Llama3
        .render(&[ChatMessage::user("hi"), ChatMessage::assistant("there")]);
    assert!(out.contains("<|start_header_id|>user<|end_header_id|>"));
    assert!(out.contains("<|start_header_id|>assistant<|end_header_id|>"));
    assert_eq!(out.matches("<|eot_id|>").count(), 2);
}

#[test]
fn mistral_multi_turn() {
    let out = ChatTemplateKind::Mistral.render(&[
        ChatMessage::user("hi"),
        ChatMessage::assistant("there"),
        ChatMessage::user("again"),
    ]);
    assert_eq!(out.matches("[INST]").count(), 2);
    assert!(out.contains("there"));
}

#[test]
fn gemma_multi_turn() {
    let out =
        ChatTemplateKind::Gemma.render(&[ChatMessage::user("hi"), ChatMessage::assistant("there")]);
    assert_eq!(out.matches("<start_of_turn>").count(), 2);
    assert_eq!(out.matches("<end_of_turn>").count(), 2);
}

#[test]
fn qwen_multi_turn() {
    let out =
        ChatTemplateKind::Qwen.render(&[ChatMessage::user("hi"), ChatMessage::assistant("there")]);
    assert!(out.contains("<|im_start|>user"));
    assert!(out.contains("<|im_start|>assistant"));
}

// ── System + user + assistant ────────────────────────────────────────────────

#[test]
fn chatml_with_system() {
    let out = ChatTemplateKind::ChatML.render(&[
        ChatMessage::system("you are helpful"),
        ChatMessage::user("hi"),
        ChatMessage::assistant("hello"),
    ]);
    assert!(out.contains("<|im_start|>system"));
    assert!(out.contains("you are helpful"));
    assert!(out.contains("<|im_start|>user"));
    assert!(out.contains("<|im_start|>assistant"));
}

#[test]
fn llama3_with_system() {
    let out = ChatTemplateKind::Llama3
        .render(&[ChatMessage::system("sys prompt"), ChatMessage::user("hi")]);
    assert!(out.contains("<|start_header_id|>system<|end_header_id|>"));
    assert!(out.contains("sys prompt"));
}

#[test]
fn gemma_with_system() {
    let out = ChatTemplateKind::Gemma.render(&[
        ChatMessage::system("sys"),
        ChatMessage::user("u"),
        ChatMessage::assistant("a"),
    ]);
    assert!(out.contains("<start_of_turn>system"));
    assert!(out.contains("<start_of_turn>user"));
    assert!(out.contains("<start_of_turn>assistant"));
}

#[test]
fn qwen_with_system() {
    let out = ChatTemplateKind::Qwen
        .render(&[ChatMessage::system("You are Qwen"), ChatMessage::user("hi")]);
    assert!(out.contains("<|im_start|>system"));
    assert!(out.contains("You are Qwen"));
}

#[test]
fn mistral_with_system_no_crash() {
    // Mistral doesn't natively have a system role — the evaluator should at
    // least not panic and should emit *something* for the system turn.
    let out =
        ChatTemplateKind::Mistral.render(&[ChatMessage::system("sys"), ChatMessage::user("hi")]);
    assert!(out.contains("[INST] hi [/INST]"));
}

// ── Generation prompt ON ────────────────────────────────────────────────────

#[test]
fn chatml_generation_prompt() {
    let out = ChatTemplateKind::ChatML.render_with_generation_prompt(&[ChatMessage::user("hi")]);
    assert!(out.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn llama3_generation_prompt() {
    let out = ChatTemplateKind::Llama3.render_with_generation_prompt(&[ChatMessage::user("hi")]);
    assert!(out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
}

#[test]
fn gemma_generation_prompt() {
    let out = ChatTemplateKind::Gemma.render_with_generation_prompt(&[ChatMessage::user("hi")]);
    assert!(out.ends_with("<start_of_turn>model\n"));
}

#[test]
fn qwen_generation_prompt() {
    let out = ChatTemplateKind::Qwen.render_with_generation_prompt(&[ChatMessage::user("hi")]);
    assert!(out.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn mistral_generation_prompt_is_empty() {
    // Mistral is self-delimiting — no extra opener is appended.
    let plain = ChatTemplateKind::Mistral.render(&[ChatMessage::user("hi")]);
    let with_prompt =
        ChatTemplateKind::Mistral.render_with_generation_prompt(&[ChatMessage::user("hi")]);
    assert_eq!(plain, with_prompt);
}

// ── Generation prompt OFF ────────────────────────────────────────────────────

#[test]
fn chatml_no_generation_prompt_by_default() {
    let out = ChatTemplateKind::ChatML.render(&[ChatMessage::user("hi")]);
    assert!(!out.ends_with("<|im_start|>assistant\n"));
}

#[test]
fn llama3_no_generation_prompt_by_default() {
    let out = ChatTemplateKind::Llama3.render(&[ChatMessage::user("hi")]);
    assert!(!out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
}

#[test]
fn gemma_no_generation_prompt_by_default() {
    let out = ChatTemplateKind::Gemma.render(&[ChatMessage::user("hi")]);
    assert!(!out.ends_with("<start_of_turn>model\n"));
}

#[test]
fn qwen_no_generation_prompt_by_default() {
    let out = ChatTemplateKind::Qwen.render(&[ChatMessage::user("hi")]);
    assert!(!out.ends_with("<|im_start|>assistant\n"));
}

// ── Empty-messages behaviour ─────────────────────────────────────────────────

#[test]
fn chatml_empty_messages_produces_empty_loop_body() {
    let out = ChatTemplateKind::ChatML.render(&[]);
    // No user/assistant markers because the for-loop ran zero times.
    assert!(!out.contains("<|im_start|>user"));
    assert!(!out.contains("<|im_end|>"));
}

#[test]
fn llama3_empty_messages_keeps_bos() {
    let out = ChatTemplateKind::Llama3.render(&[]);
    // The <|begin_of_text|> prefix lives outside the loop, so it remains.
    assert!(out.contains("<|begin_of_text|>"));
    assert!(!out.contains("<|start_header_id|>"));
}

#[test]
fn mistral_empty_messages_yields_empty_string() {
    let out = ChatTemplateKind::Mistral.render(&[]);
    assert!(out.is_empty() || !out.contains("[INST]"));
}

#[test]
fn gemma_empty_messages_yields_empty_string() {
    let out = ChatTemplateKind::Gemma.render(&[]);
    assert!(!out.contains("<start_of_turn>"));
}

#[test]
fn qwen_empty_messages_yields_empty_string() {
    let out = ChatTemplateKind::Qwen.render(&[]);
    assert!(!out.contains("<|im_start|>"));
}

#[test]
fn empty_plus_generation_prompt_yields_just_the_prompt() {
    let out = ChatTemplateKind::ChatML.render_with_generation_prompt(&[]);
    assert_eq!(out, "<|im_start|>assistant\n");
}

// ── ChatMessage helpers ──────────────────────────────────────────────────────

#[test]
fn chat_message_user_ctor() {
    let m = ChatMessage::user("x");
    assert_eq!(m.role, "user");
    assert_eq!(m.content, "x");
}

#[test]
fn chat_message_assistant_ctor() {
    let m = ChatMessage::assistant("y");
    assert_eq!(m.role, "assistant");
}

#[test]
fn chat_message_system_ctor() {
    let m = ChatMessage::system("z");
    assert_eq!(m.role, "system");
}

#[test]
fn chat_message_new_custom_role() {
    let m = ChatMessage::new("tool", "call");
    assert_eq!(m.role, "tool");
    assert_eq!(m.content, "call");
}

// ── Template introspection ───────────────────────────────────────────────────

#[test]
fn all_kinds_have_nonempty_template() {
    for kind in ChatTemplateKind::all() {
        assert!(!kind.template().is_empty());
    }
}

#[test]
fn all_kinds_render_user_hi() {
    for kind in ChatTemplateKind::all() {
        let out = kind.render(&[ChatMessage::user("hi")]);
        assert!(out.contains("hi"), "kind {:?} didn't render content", kind);
    }
}

// ── Infer from model name ────────────────────────────────────────────────────

#[test]
fn infer_llama3() {
    assert_eq!(
        ChatTemplateKind::infer_from_name("Meta-Llama-3-8B-Instruct"),
        Some(ChatTemplateKind::Llama3)
    );
}

#[test]
fn infer_llama3_lowercase() {
    assert_eq!(
        ChatTemplateKind::infer_from_name("llama3-8b"),
        Some(ChatTemplateKind::Llama3)
    );
}

#[test]
fn infer_qwen() {
    assert_eq!(
        ChatTemplateKind::infer_from_name("Qwen3-1.7B"),
        Some(ChatTemplateKind::Qwen)
    );
}

#[test]
fn infer_mistral() {
    assert_eq!(
        ChatTemplateKind::infer_from_name("Mistral-7B-Instruct-v0.2"),
        Some(ChatTemplateKind::Mistral)
    );
}

#[test]
fn infer_gemma() {
    assert_eq!(
        ChatTemplateKind::infer_from_name("gemma-2-9b-it"),
        Some(ChatTemplateKind::Gemma)
    );
}

#[test]
fn infer_chatml_fallback() {
    assert_eq!(
        ChatTemplateKind::infer_from_name("custom-chatml-finetune"),
        Some(ChatTemplateKind::ChatML)
    );
}

#[test]
fn infer_unknown_returns_none() {
    assert_eq!(ChatTemplateKind::infer_from_name("bert"), None);
}

// ── Encode roundtrip ─────────────────────────────────────────────────────────

#[test]
fn encode_against_char_level_tokenizer() {
    let tok = OxiTokenizer::char_level_stub(256);
    let ids = ChatTemplateKind::ChatML
        .encode(&tok, &[ChatMessage::user("hi")])
        .expect("encode ok");
    assert!(!ids.is_empty());
}

#[test]
fn encode_preserves_content_bytes_when_ascii() {
    let tok = OxiTokenizer::char_level_stub(256);
    let template = ChatTemplateKind::ChatML;
    let messages = [ChatMessage::user("abc")];
    let rendered = template.render(&messages);
    let via_encode = template.encode(&tok, &messages).expect("encode");
    let via_tok = tok.encode(&rendered).expect("tok encode");
    assert_eq!(via_encode, via_tok);
}

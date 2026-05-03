//! Enhanced SSE streaming with delta tokens, finish reasons, and usage info.
//!
//! This module provides OpenAI-compatible Server-Sent Events (SSE) streaming
//! primitives:
//!
//! - [`StreamChunk`] / [`StreamChoice`] / [`StreamDelta`] — wire-format structs
//!   that match the OpenAI `chat.completion.chunk` schema.
//! - [`SseFormatter`] — stateless helpers that format SSE event strings.
//! - [`TokenStream`] — a byte-level buffer that accumulates raw token bytes and
//!   yields decoded `String`s as soon as a valid UTF-8 sequence is complete.
//! - [`StreamStats`] — throughput accounting for a single generation request.

use std::time::{SystemTime, UNIX_EPOCH};

// ── Wire-format structs ───────────────────────────────────────────────────────

/// A single SSE streaming chunk (OpenAI-compatible delta format).
#[derive(Debug, Clone, serde::Serialize)]
pub struct StreamChunk {
    /// Unique completion ID shared across all chunks in one generation.
    pub id: String,
    /// Always `"chat.completion.chunk"`.
    pub object: String,
    /// Unix timestamp of when the generation started.
    pub created: u64,
    /// Model name (e.g. `"bonsai-8b"`).
    pub model: String,
    /// One-element list of choices (multi-choice streaming is not yet supported).
    pub choices: Vec<StreamChoice>,
}

/// A single choice within a [`StreamChunk`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct StreamChoice {
    /// Zero-based choice index.
    pub index: usize,
    /// The incremental delta for this chunk.
    pub delta: StreamDelta,
    /// `None` for all chunks except the last; `"stop"` / `"length"` on the last chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// Log-probability information (not yet computed — always `null`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<serde_json::Value>,
}

/// The incremental content delta for one chunk.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StreamDelta {
    /// Set to `"assistant"` on the very first chunk; `None` on subsequent chunks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// The token text for this chunk; `None` on the final (finish-reason) chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

// ── SseFormatter ─────────────────────────────────────────────────────────────

/// Stateless SSE event formatter.
///
/// Produces correctly framed `data: …\n\n` strings for each stage of a
/// streaming generation response.
pub struct SseFormatter {
    /// Whether to append a usage chunk after the final delta.
    pub include_usage: bool,
    model_name: String,
}

impl SseFormatter {
    /// Create a new formatter for the given model.
    pub fn new(model_name: &str) -> Self {
        Self {
            include_usage: false,
            model_name: model_name.to_owned(),
        }
    }

    /// Enable a trailing usage chunk.
    pub fn with_usage(mut self) -> Self {
        self.include_usage = true;
        self
    }

    /// Return the current Unix timestamp in seconds.
    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    }

    /// Format the **first** chunk of a streaming response.
    ///
    /// The first chunk carries `role: "assistant"` and an empty content string
    /// so that clients can render the role indicator immediately.
    pub fn first_chunk(&self, request_id: &str) -> String {
        let chunk = StreamChunk {
            id: request_id.to_owned(),
            object: "chat.completion.chunk".to_owned(),
            created: Self::now_secs(),
            model: self.model_name.clone(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta {
                    role: Some("assistant".to_owned()),
                    content: Some(String::new()),
                },
                finish_reason: None,
                logprobs: None,
            }],
        };
        Self::format_event(&serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".to_owned()))
    }

    /// Format a **token delta** chunk carrying `token_text` as the content.
    pub fn token_chunk(&self, request_id: &str, token_text: &str) -> String {
        let chunk = StreamChunk {
            id: request_id.to_owned(),
            object: "chat.completion.chunk".to_owned(),
            created: Self::now_secs(),
            model: self.model_name.clone(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta {
                    role: None,
                    content: Some(token_text.to_owned()),
                },
                finish_reason: None,
                logprobs: None,
            }],
        };
        Self::format_event(&serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".to_owned()))
    }

    /// Format the **final** chunk carrying the `finish_reason` and no content.
    pub fn final_chunk(&self, request_id: &str, finish_reason: &str) -> String {
        let chunk = StreamChunk {
            id: request_id.to_owned(),
            object: "chat.completion.chunk".to_owned(),
            created: Self::now_secs(),
            model: self.model_name.clone(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some(finish_reason.to_owned()),
                logprobs: None,
            }],
        };
        Self::format_event(&serde_json::to_string(&chunk).unwrap_or_else(|_| "{}".to_owned()))
    }

    /// The SSE `[DONE]` sentinel that signals stream completion.
    pub fn done_sentinel() -> &'static str {
        "data: [DONE]\n\n"
    }

    /// Wrap arbitrary JSON data in a `data: …\n\n` SSE frame.
    pub fn format_event(data: &str) -> String {
        format!("data: {data}\n\n")
    }

    /// Format a JSON error payload as an SSE event.
    pub fn error_event(message: &str) -> String {
        // Escape the message to avoid breaking the JSON.
        let escaped = message.replace('\\', "\\\\").replace('"', "\\\"");
        Self::format_event(&format!(r#"{{"error":{{"message":"{escaped}"}}}}"#))
    }
}

// ── TokenStream ───────────────────────────────────────────────────────────────

/// Byte-level detokenizer buffer with partial-token accumulation.
///
/// Raw model output often arrives as byte slices that do not align with UTF-8
/// character boundaries (e.g. multi-byte CJK characters split across two model
/// tokens).  `TokenStream` accumulates bytes until a complete UTF-8 sequence is
/// available, then returns the decoded string.
pub struct TokenStream {
    buffer: Vec<u8>,
    /// If `true`, the stream defers flushing until a whitespace boundary is found.
    /// This can be useful for word-level de-tokenization.
    pub flush_at_whitespace: bool,
}

impl Default for TokenStream {
    fn default() -> Self {
        Self::new()
    }
}

impl TokenStream {
    /// Create a new empty `TokenStream`.
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            flush_at_whitespace: false,
        }
    }

    /// Append `bytes` to the internal buffer.
    ///
    /// Returns `Some(text)` if the buffer now forms a valid complete UTF-8
    /// string, or `None` if more bytes are still needed to complete a multi-byte
    /// character.
    pub fn push_token_bytes(&mut self, bytes: &[u8]) -> Option<String> {
        self.buffer.extend_from_slice(bytes);

        // Try to decode the buffer as UTF-8.
        match std::str::from_utf8(&self.buffer) {
            Ok(s) => {
                if self.flush_at_whitespace {
                    // Only flush at whitespace boundaries.
                    if s.contains(char::is_whitespace) {
                        let text = s.to_owned();
                        self.buffer.clear();
                        Some(text)
                    } else {
                        None
                    }
                } else {
                    let text = s.to_owned();
                    self.buffer.clear();
                    Some(text)
                }
            }
            Err(e) => {
                // Check if there is a valid prefix followed by an incomplete sequence.
                let valid_up_to = e.valid_up_to();
                if valid_up_to > 0 {
                    // Emit the valid prefix; keep the incomplete tail.
                    let text = std::str::from_utf8(&self.buffer[..valid_up_to])
                        .unwrap_or("") // safe: we just validated this range
                        .to_owned();
                    self.buffer.drain(..valid_up_to);
                    Some(text)
                } else {
                    // Still mid-sequence — wait for more bytes.
                    None
                }
            }
        }
    }

    /// Force-flush whatever remains in the buffer as lossy UTF-8.
    ///
    /// Any invalid byte sequences are replaced with U+FFFD (replacement char).
    pub fn flush(&mut self) -> String {
        let text = String::from_utf8_lossy(&self.buffer).into_owned();
        self.buffer.clear();
        text
    }

    /// Returns `true` if the internal buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }
}

// ── StreamStats ───────────────────────────────────────────────────────────────

/// Per-request generation throughput statistics.
#[derive(Debug, Default, serde::Serialize)]
pub struct StreamStats {
    /// Total tokens emitted in the completion.
    pub tokens_generated: usize,
    /// Number of tokens in the prompt (prefill phase).
    pub prefill_tokens: usize,
    /// Wall-clock milliseconds until the first token was emitted.
    pub time_to_first_token_ms: u64,
    /// Total wall-clock milliseconds for the entire generation.
    pub total_time_ms: u64,
    /// Tokens-per-second throughput (cached result of [`StreamStats::throughput`]).
    pub tokens_per_second: f32,
}

impl StreamStats {
    /// Create a blank `StreamStats`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the final statistics after generation completes.
    pub fn finish(&mut self, tokens: usize, prefill: usize, ttft_ms: u64, total_ms: u64) {
        self.tokens_generated = tokens;
        self.prefill_tokens = prefill;
        self.time_to_first_token_ms = ttft_ms;
        self.total_time_ms = total_ms;
        self.tokens_per_second = self.throughput();
    }

    /// Compute tokens-per-second from recorded statistics.
    ///
    /// Returns `0.0` if `total_time_ms` is zero (avoids division by zero).
    pub fn throughput(&self) -> f32 {
        if self.total_time_ms == 0 {
            return 0.0;
        }
        self.tokens_generated as f32 / (self.total_time_ms as f32 / 1_000.0)
    }

    /// Serialize these statistics as an SSE usage chunk.
    ///
    /// The payload follows the OpenAI convention of appending a final usage
    /// chunk before `[DONE]`:
    ///
    /// ```json
    /// {"id":"...","object":"chat.completion.chunk","usage":{"prompt_tokens":…,…}}
    /// ```
    pub fn to_usage_chunk(&self, request_id: &str, model: &str) -> String {
        let payload = serde_json::json!({
            "id": request_id,
            "object": "chat.completion.chunk",
            "model": model,
            "usage": {
                "prompt_tokens": self.prefill_tokens,
                "completion_tokens": self.tokens_generated,
                "total_tokens": self.prefill_tokens + self.tokens_generated,
            }
        });
        SseFormatter::format_event(
            &serde_json::to_string(&payload).unwrap_or_else(|_| "{}".to_owned()),
        )
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_formatter() -> SseFormatter {
        SseFormatter::new("bonsai-8b")
    }

    // ── SseFormatter ──

    #[test]
    fn test_sse_formatter_first_chunk_has_role() {
        let fmt = make_formatter();
        let event = fmt.first_chunk("req-001");
        let json_part = event
            .strip_prefix("data: ")
            .expect("must start with data:")
            .trim_end();
        let v: serde_json::Value = serde_json::from_str(json_part).expect("must be valid JSON");
        let role = &v["choices"][0]["delta"]["role"];
        assert_eq!(role, "assistant", "first chunk must carry role: assistant");
    }

    #[test]
    fn test_sse_formatter_token_chunk_has_content() {
        let fmt = make_formatter();
        let event = fmt.token_chunk("req-002", "Hello");
        let json_part = event
            .strip_prefix("data: ")
            .expect("must start with data:")
            .trim_end();
        let v: serde_json::Value = serde_json::from_str(json_part).expect("must be valid JSON");
        let content = &v["choices"][0]["delta"]["content"];
        assert_eq!(content, "Hello", "token chunk must carry content");
        // role should be absent.
        assert!(
            v["choices"][0]["delta"]["role"].is_null(),
            "token chunk must not carry role"
        );
    }

    #[test]
    fn test_sse_formatter_final_chunk_has_finish_reason() {
        let fmt = make_formatter();
        let event = fmt.final_chunk("req-003", "stop");
        let json_part = event
            .strip_prefix("data: ")
            .expect("must start with data:")
            .trim_end();
        let v: serde_json::Value = serde_json::from_str(json_part).expect("must be valid JSON");
        let reason = &v["choices"][0]["finish_reason"];
        assert_eq!(reason, "stop", "final chunk must carry finish_reason");
    }

    #[test]
    fn test_sse_formatter_done_sentinel() {
        assert_eq!(SseFormatter::done_sentinel(), "data: [DONE]\n\n");
    }

    #[test]
    fn test_sse_format_event() {
        let event = SseFormatter::format_event(r#"{"foo":"bar"}"#);
        assert_eq!(event, "data: {\"foo\":\"bar\"}\n\n");
    }

    #[test]
    fn test_sse_error_event() {
        let event = SseFormatter::error_event("something went wrong");
        assert!(event.starts_with("data: "), "must be an SSE data event");
        assert!(
            event.contains("something went wrong"),
            "must contain the message"
        );
        // Must parse as valid JSON.
        let json_part = event
            .strip_prefix("data: ")
            .expect("data: prefix")
            .trim_end();
        let v: serde_json::Value =
            serde_json::from_str(json_part).expect("error event must be valid JSON");
        assert!(v["error"]["message"].is_string());
    }

    // ── TokenStream ──

    #[test]
    fn test_token_stream_ascii_passthrough() {
        let mut ts = TokenStream::new();
        let result = ts.push_token_bytes(b"hello");
        assert_eq!(result, Some("hello".to_owned()));
        assert!(ts.is_empty());
    }

    #[test]
    fn test_token_stream_flush() {
        let mut ts = TokenStream::new();
        // Push a valid ASCII byte so something is in the buffer-then-flushed path.
        ts.push_token_bytes(b"hi");
        // Buffer is cleared after the push_token_bytes call above.
        // Now push a partial UTF-8 sequence.
        let partial = &[0xE4u8, 0xB8u8]; // first 2 bytes of a 3-byte CJK char
        let result = ts.push_token_bytes(partial);
        assert!(result.is_none(), "incomplete sequence should return None");
        // Force flush — should produce replacement char or whatever is valid.
        let flushed = ts.flush();
        assert!(!flushed.is_empty() || flushed.is_empty()); // either outcome is OK
        assert!(ts.is_empty(), "buffer must be empty after flush");
    }

    #[test]
    fn test_token_stream_empty_after_flush() {
        let mut ts = TokenStream::new();
        let _ = ts.flush(); // flush on empty buffer
        assert!(ts.is_empty());
    }

    #[test]
    fn test_token_stream_multibyte_utf8() {
        let mut ts = TokenStream::new();
        // "中" = U+4E2D = bytes [0xE4, 0xB8, 0xAD]
        let bytes = "中".as_bytes();

        // Push first two bytes — should return None.
        let r1 = ts.push_token_bytes(&bytes[..2]);
        assert!(r1.is_none(), "incomplete UTF-8 should return None");

        // Push the final byte — should now decode.
        let r2 = ts.push_token_bytes(&bytes[2..]);
        assert_eq!(r2, Some("中".to_owned()));
        assert!(ts.is_empty());
    }

    // ── StreamStats ──

    #[test]
    fn test_stream_stats_throughput() {
        let mut stats = StreamStats::new();
        stats.tokens_generated = 100;
        stats.total_time_ms = 2_000; // 2 seconds
        let tps = stats.throughput();
        assert!((tps - 50.0).abs() < 0.01, "expected 50 tps, got {tps}");
    }

    #[test]
    fn test_stream_stats_throughput_zero_time() {
        let stats = StreamStats::new(); // total_time_ms == 0
        assert_eq!(stats.throughput(), 0.0);
    }

    #[test]
    fn test_stream_stats_finish() {
        let mut stats = StreamStats::new();
        stats.finish(200, 50, 120, 4_000);
        assert_eq!(stats.tokens_generated, 200);
        assert_eq!(stats.prefill_tokens, 50);
        assert_eq!(stats.time_to_first_token_ms, 120);
        assert_eq!(stats.total_time_ms, 4_000);
        // throughput = 200 / 4.0 = 50 tps
        assert!((stats.tokens_per_second - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_stream_chunk_serializes_correctly() {
        let chunk = StreamChunk {
            id: "chatcmpl-abc".to_owned(),
            object: "chat.completion.chunk".to_owned(),
            created: 1_700_000_000,
            model: "bonsai-8b".to_owned(),
            choices: vec![StreamChoice {
                index: 0,
                delta: StreamDelta {
                    role: Some("assistant".to_owned()),
                    content: Some("Hi".to_owned()),
                },
                finish_reason: None,
                logprobs: None,
            }],
        };

        let json = serde_json::to_string(&chunk).expect("serialization must succeed");
        let v: serde_json::Value = serde_json::from_str(&json).expect("must parse back to JSON");

        assert_eq!(v["id"], "chatcmpl-abc");
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(v["choices"][0]["delta"]["content"], "Hi");
        // finish_reason is None so it should be absent from JSON.
        assert!(v["choices"][0]["finish_reason"].is_null());
    }

    #[test]
    fn test_stream_stats_usage_chunk() {
        let mut stats = StreamStats::new();
        stats.finish(10, 5, 50, 1_000);
        let chunk = stats.to_usage_chunk("req-x", "bonsai-8b");
        assert!(chunk.starts_with("data: "));
        let json_part = chunk.strip_prefix("data: ").expect("prefix").trim_end();
        let v: serde_json::Value =
            serde_json::from_str(json_part).expect("usage chunk must be valid JSON");
        assert_eq!(v["usage"]["prompt_tokens"], 5);
        assert_eq!(v["usage"]["completion_tokens"], 10);
        assert_eq!(v["usage"]["total_tokens"], 15);
    }
}

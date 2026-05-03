//! UTF-8-safe streaming decoder.
//!
//! When a server emits tokens one at a time, naive `decode(&[id])` can return
//! strings with invalid UTF-8 because a single BPE token may hold *part* of a
//! multi-byte codepoint (common for CJK / emoji output).  The decoder in this
//! module keeps a small byte buffer across calls and only flushes characters
//! that form a complete UTF-8 sequence.
//!
//! ## Usage
//!
//! ```rust
//! use oxibonsai_tokenizer::OxiTokenizer;
//!
//! let tok = OxiTokenizer::char_level_stub(256);
//! let ids = tok.encode("Hello!").expect("encode");
//! let mut dec = tok.streaming_decoder();
//! let mut out = String::new();
//! for id in &ids {
//!     if let Some(piece) = dec.push_token(*id) {
//!         out.push_str(&piece);
//!     }
//! }
//! out.push_str(&dec.finish().expect("stream must end on a UTF-8 boundary"));
//! assert_eq!(out, "Hello!");
//! ```

use crate::{
    error::{TokenizerError, TokenizerResult},
    tokenizer::OxiTokenizer,
};

/// A streaming decoder that yields well-formed UTF-8 slices as tokens arrive.
///
/// The decoder holds a reference to its parent [`OxiTokenizer`] so that
/// special-token handling, vocabulary lookup and byte-level decoding remain
/// consistent with [`OxiTokenizer::decode`].
pub struct StreamingDecoder<'a> {
    tokenizer: &'a OxiTokenizer,
    /// Bytes that have been decoded but not yet emitted because they are
    /// part of an incomplete UTF-8 sequence.
    pending: Vec<u8>,
    /// Total bytes the decoder has seen across the stream (for diagnostics).
    total_bytes: usize,
    /// Total tokens the decoder has seen across the stream.
    total_tokens: usize,
}

impl<'a> StreamingDecoder<'a> {
    /// Create a fresh decoder tied to `tokenizer`.
    pub fn new(tokenizer: &'a OxiTokenizer) -> Self {
        Self {
            tokenizer,
            pending: Vec::with_capacity(8),
            total_bytes: 0,
            total_tokens: 0,
        }
    }

    /// Push a single token ID and return the next well-formed UTF-8 slice, if
    /// any.  Returns `None` when the token's bytes do not extend any
    /// previously-pending prefix into a full UTF-8 character.
    ///
    /// The returned `String` contains all characters that became complete as
    /// a result of this push — may be multiple characters if the token
    /// carries several whole code points.
    pub fn push_token(&mut self, id: u32) -> Option<String> {
        self.total_tokens += 1;
        let mut scratch: Vec<u8> = Vec::with_capacity(8);
        self.tokenizer.decode_id_into(id, &mut scratch);
        if scratch.is_empty() {
            return None;
        }
        self.total_bytes += scratch.len();
        self.pending.extend_from_slice(&scratch);
        self.flush_complete()
    }

    /// Push many tokens at once.  Equivalent to repeatedly calling
    /// [`Self::push_token`] but only returns once, with all complete
    /// characters concatenated.
    pub fn push_tokens(&mut self, ids: &[u32]) -> Option<String> {
        let mut out = String::new();
        for &id in ids {
            if let Some(piece) = self.push_token(id) {
                out.push_str(&piece);
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(out)
        }
    }

    /// Finish the stream and return any remaining bytes as a `String`.
    ///
    /// Returns an error if the pending buffer still contains an incomplete
    /// UTF-8 sequence (strict mode).  If lossy finishing is desired, use
    /// [`Self::finish_lossy`] instead.
    pub fn finish(mut self) -> TokenizerResult<String> {
        if self.pending.is_empty() {
            return Ok(String::new());
        }
        match String::from_utf8(std::mem::take(&mut self.pending)) {
            Ok(s) => Ok(s),
            Err(_) => Err(TokenizerError::IncompleteUtf8),
        }
    }

    /// Finish the stream, replacing any trailing invalid bytes with
    /// `\u{FFFD}`.  Never fails.
    pub fn finish_lossy(mut self) -> String {
        if self.pending.is_empty() {
            return String::new();
        }
        let bytes = std::mem::take(&mut self.pending);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Number of bytes currently held in the pending buffer.
    ///
    /// A non-zero value after a `push_token` call indicates that the last
    /// token ended mid-UTF-8-sequence.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Reset the decoder state without destroying the `OxiTokenizer`
    /// reference — useful when processing multiple independent streams.
    pub fn reset(&mut self) {
        self.pending.clear();
        self.total_bytes = 0;
        self.total_tokens = 0;
    }

    /// Total bytes processed since construction or last [`Self::reset`].
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Total tokens processed since construction or last [`Self::reset`].
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Pull all complete UTF-8 characters out of `pending`, leaving any
    /// trailing incomplete sequence behind.
    fn flush_complete(&mut self) -> Option<String> {
        if self.pending.is_empty() {
            return None;
        }

        // Find the longest UTF-8-valid prefix of `pending`.
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                // Entire buffer is valid.
                let owned = s.to_owned();
                self.pending.clear();
                if owned.is_empty() {
                    None
                } else {
                    Some(owned)
                }
            }
            Err(e) => {
                let valid_up_to = e.valid_up_to();
                if valid_up_to == 0 {
                    return None;
                }
                // Extract the complete prefix.
                let prefix_bytes = self.pending[..valid_up_to].to_vec();
                self.pending.drain(..valid_up_to);
                match String::from_utf8(prefix_bytes) {
                    Ok(s) if !s.is_empty() => Some(s),
                    _ => None,
                }
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use crate::OxiTokenizer;

    #[test]
    fn ascii_passthrough() {
        let tok = OxiTokenizer::char_level_stub(256);
        let ids = tok.encode("abc").expect("encode");
        let mut dec = tok.streaming_decoder();
        let mut out = String::new();
        for id in &ids {
            if let Some(piece) = dec.push_token(*id) {
                out.push_str(&piece);
            }
        }
        out.push_str(&dec.finish().expect("finish ok"));
        assert_eq!(out, "abc");
    }

    #[test]
    fn reset_clears_state() {
        let tok = OxiTokenizer::char_level_stub(256);
        let mut dec = tok.streaming_decoder();
        let ids = tok.encode("abc").expect("encode");
        for id in &ids {
            dec.push_token(*id);
        }
        dec.reset();
        assert_eq!(dec.pending_len(), 0);
        assert_eq!(dec.total_bytes(), 0);
        assert_eq!(dec.total_tokens(), 0);
    }

    #[test]
    fn push_tokens_batch() {
        let tok = OxiTokenizer::char_level_stub(256);
        let mut dec = tok.streaming_decoder();
        let ids = tok.encode("hello").expect("encode");
        let out = dec.push_tokens(&ids).unwrap_or_default();
        // Non-empty because char-level stub emits one char per token.
        assert!(!out.is_empty());
    }

    #[test]
    fn finish_on_empty_is_ok() {
        let tok = OxiTokenizer::char_level_stub(256);
        let dec = tok.streaming_decoder();
        let out = dec.finish().expect("empty finish ok");
        assert_eq!(out, "");
    }

    #[test]
    fn finish_lossy_never_fails() {
        let tok = OxiTokenizer::char_level_stub(256);
        let dec = tok.streaming_decoder();
        let out = dec.finish_lossy();
        assert_eq!(out, "");
    }

    #[test]
    fn counters_advance() {
        let tok = OxiTokenizer::char_level_stub(256);
        let mut dec = tok.streaming_decoder();
        let ids = tok.encode("ab").expect("encode");
        for id in &ids {
            dec.push_token(*id);
        }
        assert!(dec.total_tokens() >= ids.len());
        assert!(dec.total_bytes() > 0);
    }
}

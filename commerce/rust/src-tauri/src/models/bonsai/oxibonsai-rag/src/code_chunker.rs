//! Language-aware chunking for source files.
//!
//! [`CodeChunker`] splits code by language-specific structural markers so
//! that retrieval returns syntactically meaningful units rather than
//! arbitrary character windows.  Supported languages are enumerated by
//! [`Language`]; anything outside that set falls back to
//! [`RecursiveCharSplitter`].
//!
//! # Language splitters
//!
//! - [`Language::Rust`] — splits on top-level `fn `, `impl `, `struct `,
//!   `enum `, `mod `, and `pub fn ` forms that occur at the start of a
//!   line (preceded by `\n`).
//! - [`Language::Python`] — splits on `\nclass ` and `\ndef ` at the start
//!   of a line.
//! - [`Language::Json`] — if the document parses as a JSON array/object,
//!   each depth-1 child becomes a chunk.  Malformed JSON falls back to
//!   [`RecursiveCharSplitter`].
//! - [`Language::Plain`] — delegates to [`RecursiveCharSplitter`] with a
//!   configurable window size.

use serde::{Deserialize, Serialize};

use crate::advanced_chunker::{ChunkStrategy, RecursiveCharSplitter};
use crate::chunker::Chunk;
use crate::error::RagError;

// ─────────────────────────────────────────────────────────────────────────────
// Language enum
// ─────────────────────────────────────────────────────────────────────────────

/// Source-code languages recognised by [`CodeChunker`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    /// Rust source.
    Rust,
    /// Python source.
    Python,
    /// JSON document (depth-1 child splitter).
    Json,
    /// Anything else — delegates to the character-recursive fallback.
    #[default]
    Plain,
}

impl Language {
    /// Classify a file based on its lower-cased extension.  Unknown
    /// extensions map to [`Language::Plain`].
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_ascii_lowercase().as_str() {
            "rs" => Self::Rust,
            "py" => Self::Python,
            "json" => Self::Json,
            _ => Self::Plain,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CodeChunker
// ─────────────────────────────────────────────────────────────────────────────

/// Language-aware code chunker.
pub struct CodeChunker {
    language: Language,
    fallback_window: usize,
    min_chunk_chars: usize,
}

impl Default for CodeChunker {
    fn default() -> Self {
        Self {
            language: Language::default(),
            fallback_window: 1024,
            min_chunk_chars: 16,
        }
    }
}

impl CodeChunker {
    /// Create a chunker for `language` with default parameters.
    pub fn new(language: Language) -> Self {
        Self {
            language,
            ..Self::default()
        }
    }

    /// Configure the fallback [`RecursiveCharSplitter`] window size.
    #[must_use]
    pub fn with_fallback_window(mut self, window: usize) -> Self {
        self.fallback_window = window.max(64);
        self
    }

    /// Discard chunks shorter than `min` characters.
    #[must_use]
    pub fn with_min_chunk_chars(mut self, min: usize) -> Self {
        self.min_chunk_chars = min;
        self
    }

    /// The language this chunker was constructed with.
    pub fn language(&self) -> Language {
        self.language
    }

    /// Split `text` into chunks appropriate for its language.
    pub fn chunk(&self, text: &str, doc_id: usize) -> Result<Vec<Chunk>, RagError> {
        if text.trim().is_empty() {
            return Ok(Vec::new());
        }

        let raw_chunks: Vec<(usize, String)> = match self.language {
            Language::Rust => split_rust(text),
            Language::Python => split_python(text),
            Language::Json => split_json(text).unwrap_or_else(|| self.split_plain(text)),
            Language::Plain => self.split_plain(text),
        };

        let mut out = Vec::with_capacity(raw_chunks.len());
        for (char_offset, body) in raw_chunks {
            let trimmed = body.trim();
            if trimmed.chars().count() < self.min_chunk_chars {
                continue;
            }
            let chunk_idx = out.len();
            out.push(Chunk::new(
                trimmed.to_string(),
                doc_id,
                chunk_idx,
                char_offset,
            ));
        }

        // Fallback: if we produced nothing (e.g. a short file with no
        // structural markers) treat the whole document as one chunk.
        if out.is_empty() {
            out.push(Chunk::new(text.trim().to_string(), doc_id, 0, 0));
        }

        Ok(out)
    }

    fn split_plain(&self, text: &str) -> Vec<(usize, String)> {
        let splitter = RecursiveCharSplitter::new(self.fallback_window);
        splitter
            .chunk(text)
            .into_iter()
            .map(|rc| (rc.char_start, rc.text))
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Language-specific splitters
// ─────────────────────────────────────────────────────────────────────────────

/// Markers that signal the start of a new top-level Rust item.
const RUST_MARKERS: &[&str] = &[
    "\nfn ",
    "\npub fn ",
    "\nimpl ",
    "\nstruct ",
    "\nenum ",
    "\nmod ",
    "\npub mod ",
    "\ntrait ",
    "\npub struct ",
    "\npub enum ",
    "\npub trait ",
];

fn split_rust(text: &str) -> Vec<(usize, String)> {
    split_by_line_prefixes(text, RUST_MARKERS)
}

/// Markers that signal the start of a new top-level Python definition.
const PYTHON_MARKERS: &[&str] = &["\nclass ", "\ndef ", "\nasync def "];

fn split_python(text: &str) -> Vec<(usize, String)> {
    split_by_line_prefixes(text, PYTHON_MARKERS)
}

/// Split on the supplied line-prefix markers.  Each chunk starts at the
/// `\n` that precedes a marker (so the marker itself is included) and runs
/// up to the next marker.
fn split_by_line_prefixes(text: &str, markers: &[&str]) -> Vec<(usize, String)> {
    // Collect byte offsets of every marker occurrence
    let mut boundaries: Vec<usize> = Vec::new();
    for marker in markers {
        let mut start = 0usize;
        while let Some(idx) = text[start..].find(marker) {
            let absolute = start + idx + 1; // skip the leading '\n'
            boundaries.push(absolute);
            start = absolute + marker.len() - 1;
        }
    }
    boundaries.sort_unstable();
    boundaries.dedup();

    // Prepend 0 so the prologue (everything before the first marker) is
    // treated as its own chunk.
    let mut starts = Vec::with_capacity(boundaries.len() + 1);
    starts.push(0usize);
    starts.extend(boundaries);
    starts.dedup();

    let mut out = Vec::with_capacity(starts.len());
    for i in 0..starts.len() {
        let begin = starts[i];
        let end = starts.get(i + 1).copied().unwrap_or(text.len());
        let body = &text[begin..end];
        if !body.trim().is_empty() {
            out.push((begin, body.to_string()));
        }
    }
    out
}

fn split_json(text: &str) -> Option<Vec<(usize, String)>> {
    let value: serde_json::Value = serde_json::from_str(text).ok()?;
    match value {
        serde_json::Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for (idx, item) in items.into_iter().enumerate() {
                if let Ok(text) = serde_json::to_string_pretty(&item) {
                    out.push((idx, text));
                }
            }
            Some(out)
        }
        serde_json::Value::Object(obj) => {
            let mut out = Vec::with_capacity(obj.len());
            for (idx, (key, value)) in obj.into_iter().enumerate() {
                let body = match serde_json::to_string_pretty(&value) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                out.push((idx, format!("\"{key}\": {body}")));
            }
            Some(out)
        }
        // Scalar JSON values aren't splittable — signal failure so we fall
        // back to the character splitter.
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Inline tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_splits_on_fn_markers() {
        let source = "\nfn one() {}\nfn two() {}\nfn three() {}\n";
        let chunker = CodeChunker::new(Language::Rust).with_min_chunk_chars(1);
        let chunks = chunker.chunk(source, 0).expect("chunk");
        assert!(chunks.len() >= 3, "got {} chunks", chunks.len());
    }

    #[test]
    fn python_splits_on_def_and_class() {
        let source =
            "\nclass A:\n    pass\n\ndef foo():\n    return 1\n\ndef bar():\n    return 2\n";
        let chunker = CodeChunker::new(Language::Python).with_min_chunk_chars(1);
        let chunks = chunker.chunk(source, 0).expect("chunk");
        assert!(chunks.len() >= 3, "got {} chunks", chunks.len());
    }

    #[test]
    fn json_array_splits_by_element() {
        let source = "[1, 2, 3, 4]";
        let chunker = CodeChunker::new(Language::Json).with_min_chunk_chars(1);
        let chunks = chunker.chunk(source, 0).expect("chunk");
        assert_eq!(chunks.len(), 4);
    }

    #[test]
    fn plain_delegates_to_splitter() {
        let text = "a".repeat(4096);
        let chunker = CodeChunker::new(Language::Plain)
            .with_fallback_window(512)
            .with_min_chunk_chars(1);
        let chunks = chunker.chunk(&text, 0).expect("chunk");
        assert!(chunks.len() > 1);
    }
}

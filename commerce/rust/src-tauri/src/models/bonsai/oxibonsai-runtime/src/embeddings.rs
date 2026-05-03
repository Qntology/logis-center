//! OpenAI v1 embeddings endpoint.
//!
//! Implements `POST /v1/embeddings` — an OpenAI-compatible embedding API that
//! converts text (or token ID arrays) into dense float vectors.
//!
//! # Backends
//!
//! The [`EmbedderRegistry`] manages two complementary backends:
//!
//! - **[`TfIdfEmbedder`]** — fitted on-the-fly from the texts seen so far;
//!   becomes active after the first call (or explicit [`EmbedderRegistry::fit_tfidf`]).
//! - **[`IdentityEmbedder`]** — byte-hash fallback used before TF-IDF is fitted,
//!   always available and fully deterministic.
//!
//! # Encoding formats
//!
//! - `"float"` (default) — embedding returned as a JSON array of `f32` values.
//! - `"base64"` — embedding encoded as a hex string; each `f32` is serialised
//!   as four little-endian bytes rendered as lowercase hex.
//!
//! # Dimensions
//!
//! Setting `dimensions` in the request truncates the embedding vectors to that
//! length before returning them.  If `dimensions` exceeds the natural embedding
//! size the full vector is returned unmodified.

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json, Response},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use oxibonsai_rag::embedding::{Embedder, IdentityEmbedder, TfIdfEmbedder};

// ─── Request / Response types ─────────────────────────────────────────────────

/// Input accepted by the embeddings endpoint.
///
/// All four variants are deserialized from untagged JSON, so the format is
/// inferred from the structure of the value supplied in the `"input"` field:
///
/// | JSON value | Variant |
/// |---|---|
/// | `"some text"` | `Single` |
/// | `["text one", "text two"]` | `Batch` |
/// | `[42, 1337]` | `TokenIds` |
/// | `[[42, 1337], [9, 99]]` | `BatchTokenIds` |
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    /// A single text string.
    Single(String),
    /// A batch of text strings.
    Batch(Vec<String>),
    /// A single token-ID sequence (converted to a space-joined string).
    TokenIds(Vec<u32>),
    /// A batch of token-ID sequences.
    BatchTokenIds(Vec<Vec<u32>>),
}

impl EmbeddingInput {
    /// Convert all inputs to `String` form for embedding.
    ///
    /// Token-ID sequences are rendered as space-separated decimal numbers so
    /// they can be passed through the text-based embedder.
    pub fn as_strings(&self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s.clone()],
            EmbeddingInput::Batch(v) => v.clone(),
            EmbeddingInput::TokenIds(ids) => {
                vec![ids
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")]
            }
            EmbeddingInput::BatchTokenIds(batch) => batch
                .iter()
                .map(|ids| {
                    ids.iter()
                        .map(|id| id.to_string())
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .collect(),
        }
    }

    /// Number of distinct inputs.
    pub fn len(&self) -> usize {
        match self {
            EmbeddingInput::Single(_) => 1,
            EmbeddingInput::Batch(v) => v.len(),
            EmbeddingInput::TokenIds(_) => 1,
            EmbeddingInput::BatchTokenIds(v) => v.len(),
        }
    }

    /// Whether the input contains no items.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// `POST /v1/embeddings` request body.
#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    /// The model name (accepted but ignored — the registry selects the backend).
    pub model: Option<String>,
    /// The text(s) or token sequence(s) to embed.
    pub input: EmbeddingInput,
    /// Encoding format: `"float"` (default) or `"base64"`.
    pub encoding_format: Option<String>,
    /// If set, truncate each embedding to this many dimensions.
    pub dimensions: Option<usize>,
    /// Opaque caller identifier (not processed).
    pub user: Option<String>,
}

/// The serialised form of a single embedding.
///
/// When the request specifies `encoding_format = "base64"` the `Base64` variant
/// is used; otherwise `Float`.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum EmbeddingData {
    /// Embedding as a JSON array of `f32` values.
    Float(Vec<f32>),
    /// Embedding encoded as a hex string (see [`EmbedderRegistry::encode_base64`]).
    Base64(String),
}

/// A single embedding object in the response.
#[derive(Debug, Serialize)]
pub struct EmbeddingObject {
    /// Always `"embedding"`.
    pub object: String,
    /// The dense vector (or its encoded form).
    pub embedding: EmbeddingData,
    /// Zero-based position of this item among all inputs.
    pub index: usize,
}

/// Token usage reported in the embeddings response.
#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    /// Total tokens consumed by the prompt(s).
    pub prompt_tokens: usize,
    /// Same as `prompt_tokens` (there are no completion tokens for embeddings).
    pub total_tokens: usize,
}

/// `POST /v1/embeddings` response body.
#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    /// Always `"list"`.
    pub object: String,
    /// One [`EmbeddingObject`] per input.
    pub data: Vec<EmbeddingObject>,
    /// The model / backend used.
    pub model: String,
    /// Token usage statistics.
    pub usage: EmbeddingUsage,
}

// ─── EmbedderRegistry ─────────────────────────────────────────────────────────

/// Thread-safe registry that holds the active embedding backends.
///
/// On creation the TF-IDF slot is empty and the `IdentityEmbedder` is used as
/// a deterministic fall-back.  Once enough documents have been seen (or
/// [`EmbedderRegistry::fit_tfidf`] is called explicitly) the TF-IDF embedder
/// is installed and used for all subsequent requests.
pub struct EmbedderRegistry {
    default_dim: usize,
    tfidf: std::sync::Mutex<Option<TfIdfEmbedder>>,
    identity: IdentityEmbedder,
}

impl EmbedderRegistry {
    /// Create a new registry.
    ///
    /// `default_dim` controls the dimensionality of the `IdentityEmbedder`
    /// fallback and is also used as `max_features` when fitting TF-IDF.
    pub fn new(default_dim: usize) -> Self {
        let dim = default_dim.max(1);
        // `dim` is guaranteed ≥ 1 by the `.max(1)` clamp above, so
        // `IdentityEmbedder::new` cannot fail here.  We still handle the
        // `Err` branch explicitly to avoid `.expect()` in production code.
        let identity = match IdentityEmbedder::new(dim) {
            Ok(embedder) => embedder,
            Err(_) => unreachable!("dim ≥ 1 was guaranteed by max(1) above"),
        };
        Self {
            default_dim: dim,
            tfidf: std::sync::Mutex::new(None),
            identity,
        }
    }

    /// Embed a slice of text strings, returning one dense vector per input.
    ///
    /// Uses the TF-IDF backend when it has been fitted; falls back to
    /// `IdentityEmbedder` otherwise.  Texts that fail to embed are silently
    /// replaced with a zero vector of the appropriate dimension.
    pub fn embed_texts(&self, texts: &[String]) -> Vec<Vec<f32>> {
        let guard = self.tfidf.lock().expect("embedder registry mutex poisoned");
        if let Some(ref tfidf) = *guard {
            texts
                .iter()
                .map(|t| {
                    tfidf
                        .embed(t)
                        .unwrap_or_else(|_| vec![0.0; tfidf.embedding_dim()])
                })
                .collect()
        } else {
            texts
                .iter()
                .map(|t| {
                    self.identity
                        .embed(t)
                        .unwrap_or_else(|_| vec![0.0; self.default_dim])
                })
                .collect()
        }
    }

    /// Fit the TF-IDF backend from `corpus`.
    ///
    /// After this call [`embed_texts`](Self::embed_texts) will use TF-IDF for
    /// all subsequent requests.  Subsequent calls replace the existing model.
    pub fn fit_tfidf(&self, corpus: &[String]) {
        if corpus.is_empty() {
            return;
        }
        let refs: Vec<&str> = corpus.iter().map(String::as_str).collect();
        let fitted = TfIdfEmbedder::fit(&refs, self.default_dim);
        let mut guard = self.tfidf.lock().expect("embedder registry mutex poisoned");
        *guard = Some(fitted);
    }

    /// Return the current embedding dimension.
    ///
    /// Returns the TF-IDF vocabulary size when a fitted model is present,
    /// otherwise the configured `default_dim`.
    pub fn embedding_dim(&self) -> usize {
        let guard = self.tfidf.lock().expect("embedder registry mutex poisoned");
        if let Some(ref tfidf) = *guard {
            tfidf.embedding_dim()
        } else {
            self.default_dim
        }
    }

    /// Encode an embedding vector as a hex string (pure Rust, no external deps).
    ///
    /// Each `f32` is serialised as four bytes in little-endian order, with each
    /// byte represented as two lowercase hex digits.  The result is therefore
    /// `8 * embedding.len()` characters long.
    pub fn encode_base64(embedding: &[f32]) -> String {
        let mut out = String::with_capacity(embedding.len() * 8);
        for value in embedding {
            let bytes = value.to_le_bytes();
            for byte in bytes {
                use std::fmt::Write as _;
                let _ = write!(out, "{byte:02x}");
            }
        }
        out
    }
}

// ─── App state ────────────────────────────────────────────────────────────────

/// Axum application state for the embeddings sub-router.
pub struct EmbeddingAppState {
    /// The active embedding registry.
    pub registry: EmbedderRegistry,
}

impl EmbeddingAppState {
    /// Create a new state with the given embedding dimensionality.
    pub fn new(dim: usize) -> Self {
        Self {
            registry: EmbedderRegistry::new(dim),
        }
    }
}

// ─── Handler ──────────────────────────────────────────────────────────────────

/// Handler for `POST /v1/embeddings`.
///
/// Computes dense vector representations for all supplied inputs and returns
/// an OpenAI-compatible response.
#[tracing::instrument(skip(state))]
pub async fn create_embeddings(
    State(state): State<Arc<EmbeddingAppState>>,
    Json(req): Json<EmbeddingRequest>,
) -> Result<Response, StatusCode> {
    if req.input.is_empty() {
        return Err(StatusCode::UNPROCESSABLE_ENTITY);
    }

    let texts = req.input.as_strings();
    let use_base64 = req
        .encoding_format
        .as_deref()
        .map(|f| f == "base64")
        .unwrap_or(false);

    // Fit TF-IDF on the fly when the caller provides a meaningful corpus
    // (≥ 2 documents).  Single-document batches are not large enough to build
    // a useful IDF weighting, so we fall back to the IdentityEmbedder in that
    // case to keep embedding dimensions stable across truncation scenarios.
    if texts.len() >= 2 {
        state.registry.fit_tfidf(&texts);
    }

    let raw_embeddings = state.registry.embed_texts(&texts);

    // Count tokens for usage: approximate as whitespace-split word count.
    let prompt_tokens: usize = texts
        .iter()
        .map(|t| t.split_whitespace().count().max(1))
        .sum();

    let model_name = req.model.unwrap_or_else(|| "bonsai-embeddings".to_string());

    let data: Vec<EmbeddingObject> = raw_embeddings
        .into_iter()
        .enumerate()
        .map(|(index, mut vec)| {
            // Optionally truncate to requested dimensions.
            if let Some(dim) = req.dimensions {
                vec.truncate(dim);
            }

            let embedding = if use_base64 {
                EmbeddingData::Base64(EmbedderRegistry::encode_base64(&vec))
            } else {
                EmbeddingData::Float(vec)
            };

            EmbeddingObject {
                object: "embedding".to_owned(),
                embedding,
                index,
            }
        })
        .collect();

    let response = EmbeddingResponse {
        object: "list".to_owned(),
        data,
        model: model_name,
        usage: EmbeddingUsage {
            prompt_tokens,
            total_tokens: prompt_tokens,
        },
    };

    Ok(Json(response).into_response())
}

// ─── Router factory ───────────────────────────────────────────────────────────

/// Build a standalone Axum router for the embeddings endpoint.
///
/// Mount this at the root with [`Router::merge`] or nest it under a path
/// prefix with [`Router::nest`].  The router exposes a single route:
///
/// ```text
/// POST /v1/embeddings
/// ```
pub fn create_embeddings_router(dim: usize) -> Router {
    let state = Arc::new(EmbeddingAppState::new(dim));
    Router::new()
        .route("/v1/embeddings", axum::routing::post(create_embeddings))
        .with_state(state)
}

// ─── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── EmbeddingInput ────────────────────────────────────────────────────────

    #[test]
    fn embedding_input_single_as_strings() {
        let input = EmbeddingInput::Single("hello world".to_string());
        assert_eq!(input.as_strings(), vec!["hello world"]);
        assert_eq!(input.len(), 1);
        assert!(!input.is_empty());
    }

    #[test]
    fn embedding_input_batch_as_strings() {
        let input = EmbeddingInput::Batch(vec!["foo".to_string(), "bar".to_string()]);
        let strings = input.as_strings();
        assert_eq!(strings.len(), 2);
        assert_eq!(strings[0], "foo");
        assert_eq!(strings[1], "bar");
        assert_eq!(input.len(), 2);
    }

    #[test]
    fn embedding_input_token_ids_as_strings() {
        let input = EmbeddingInput::TokenIds(vec![1u32, 2, 3]);
        let strings = input.as_strings();
        assert_eq!(strings.len(), 1);
        assert_eq!(strings[0], "1 2 3");
    }

    #[test]
    fn embedding_input_batch_token_ids_as_strings() {
        let input = EmbeddingInput::BatchTokenIds(vec![vec![10u32, 20], vec![30u32]]);
        let strings = input.as_strings();
        assert_eq!(strings.len(), 2);
        assert_eq!(strings[0], "10 20");
        assert_eq!(strings[1], "30");
    }

    #[test]
    fn embedding_input_empty_batch_is_empty() {
        let input = EmbeddingInput::Batch(vec![]);
        assert!(input.is_empty());
        assert_eq!(input.len(), 0);
    }

    // ── EmbedderRegistry ─────────────────────────────────────────────────────

    #[test]
    fn embedder_registry_basic_embed() {
        let registry = EmbedderRegistry::new(32);
        let texts = vec!["hello world".to_string(), "foo bar baz".to_string()];
        let embeddings = registry.embed_texts(&texts);
        assert_eq!(embeddings.len(), 2);
        // Each embedding must have exactly `default_dim` elements.
        for emb in &embeddings {
            assert_eq!(emb.len(), 32, "expected 32 dimensions, got {}", emb.len());
        }
    }

    #[test]
    fn embedder_registry_tfidf_fit_changes_dim() {
        let registry = EmbedderRegistry::new(64);
        let corpus: Vec<String> = (0..20)
            .map(|i| format!("document number {i} with some unique words term{i}"))
            .collect();
        registry.fit_tfidf(&corpus);
        // After fitting the dimension comes from the TF-IDF vocabulary.
        let dim = registry.embedding_dim();
        assert!(dim > 0, "expected positive dimension after fit");
    }

    #[test]
    fn embedder_registry_fit_empty_corpus_is_noop() {
        let registry = EmbedderRegistry::new(16);
        registry.fit_tfidf(&[]);
        // Should still use IdentityEmbedder (dim == default_dim).
        assert_eq!(registry.embedding_dim(), 16);
    }

    #[test]
    fn embedder_registry_embed_after_fit() {
        let registry = EmbedderRegistry::new(32);
        let corpus: Vec<String> = vec![
            "the quick brown fox".to_string(),
            "jumped over the lazy dog".to_string(),
            "the fox and the dog".to_string(),
        ];
        registry.fit_tfidf(&corpus);
        let embeddings = registry.embed_texts(&corpus);
        for emb in &embeddings {
            assert!(!emb.is_empty(), "embedding must not be empty after fit");
        }
    }

    // ── encode_base64 ─────────────────────────────────────────────────────────

    #[test]
    fn encode_base64_non_empty() {
        let vec = vec![1.0f32, 0.5f32, -1.0f32];
        let encoded = EmbedderRegistry::encode_base64(&vec);
        // Each f32 → 4 bytes → 8 hex chars; 3 values → 24 chars.
        assert_eq!(
            encoded.len(),
            24,
            "expected 24 hex chars for 3 f32 values, got {}",
            encoded.len()
        );
        assert!(!encoded.is_empty());
    }

    #[test]
    fn encode_base64_empty_input() {
        let encoded = EmbedderRegistry::encode_base64(&[]);
        assert!(encoded.is_empty());
    }

    #[test]
    fn encode_base64_deterministic() {
        let vec = vec![std::f32::consts::PI, 2.71f32];
        let a = EmbedderRegistry::encode_base64(&vec);
        let b = EmbedderRegistry::encode_base64(&vec);
        assert_eq!(a, b, "encoding must be deterministic");
    }

    #[test]
    fn encode_base64_known_value() {
        // f32::from_le_bytes([0x00, 0x00, 0x80, 0x3f]) == 1.0
        let vec = vec![1.0f32];
        let encoded = EmbedderRegistry::encode_base64(&vec);
        assert_eq!(encoded, "0000803f");
    }

    // ── EmbeddingResponse serialisation ──────────────────────────────────────

    #[test]
    fn embedding_response_serialises_correctly() {
        let resp = EmbeddingResponse {
            object: "list".to_owned(),
            data: vec![EmbeddingObject {
                object: "embedding".to_owned(),
                embedding: EmbeddingData::Float(vec![0.1, 0.2]),
                index: 0,
            }],
            model: "bonsai-embeddings".to_owned(),
            usage: EmbeddingUsage {
                prompt_tokens: 3,
                total_tokens: 3,
            },
        };
        let json = serde_json::to_string(&resp).expect("serialisation must succeed");
        assert!(json.contains("\"object\":\"list\""));
        assert!(json.contains("\"object\":\"embedding\""));
        assert!(json.contains("\"index\":0"));
    }
}

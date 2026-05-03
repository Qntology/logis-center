//! End-to-end RAG pipeline: index → retrieve → build prompt.
//!
//! [`RagPipeline`] composes a [`Retriever`] with prompt-templating logic.
//! It is the top-level object most applications will interact with.

use tracing::debug;

use crate::chunker::ChunkConfig;
use crate::embedding::Embedder;
use crate::error::RagError;
use crate::retriever::{Retriever, RetrieverConfig};

// ─────────────────────────────────────────────────────────────────────────────
// RagConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the full RAG pipeline.
///
/// Marked `#[non_exhaustive]`; use [`RagConfig::default`] combined with the
/// `with_*` builders from external crates.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RagConfig {
    /// How to chunk documents before indexing.
    pub chunk_config: ChunkConfig,
    /// How to configure the retriever (top-k, min-score, re-rank).
    pub retriever_config: RetrieverConfig,
    /// Maximum total characters of retrieved context to include in the prompt.
    /// Retrieved chunks are concatenated in order; once this limit would be
    /// exceeded, remaining chunks are dropped.
    pub max_context_chars: usize,
    /// String placed between adjacent retrieved chunks in the context block.
    pub context_separator: String,
    /// Prompt template.  Two placeholders are expanded:
    ///
    /// - `{context}` — the retrieved context string.
    /// - `{query}` — the raw query string.
    pub prompt_template: String,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            chunk_config: ChunkConfig::default(),
            retriever_config: RetrieverConfig::default(),
            max_context_chars: 4096,
            context_separator: "\n---\n".to_string(),
            prompt_template: "{context}\n\nQuestion: {query}\n\nAnswer:".to_string(),
        }
    }
}

impl RagConfig {
    /// Set [`RagConfig::chunk_config`] (builder).
    #[must_use]
    pub fn with_chunk_config(mut self, chunk_config: ChunkConfig) -> Self {
        self.chunk_config = chunk_config;
        self
    }

    /// Set [`RagConfig::retriever_config`] (builder).
    #[must_use]
    pub fn with_retriever_config(mut self, retriever_config: RetrieverConfig) -> Self {
        self.retriever_config = retriever_config;
        self
    }

    /// Set [`RagConfig::max_context_chars`] (builder).
    #[must_use]
    pub fn with_max_context_chars(mut self, max_context_chars: usize) -> Self {
        self.max_context_chars = max_context_chars;
        self
    }

    /// Set [`RagConfig::context_separator`] (builder).
    #[must_use]
    pub fn with_context_separator(mut self, context_separator: impl Into<String>) -> Self {
        self.context_separator = context_separator.into();
        self
    }

    /// Set [`RagConfig::prompt_template`] (builder).
    #[must_use]
    pub fn with_prompt_template(mut self, prompt_template: impl Into<String>) -> Self {
        self.prompt_template = prompt_template.into();
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PipelineStats
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics for a [`RagPipeline`] instance.
#[derive(Debug, Clone)]
pub struct PipelineStats {
    /// Number of documents that have been indexed.
    pub documents_indexed: usize,
    /// Number of chunks currently in the vector store.
    pub chunks_indexed: usize,
    /// Embedding dimensionality used by this pipeline.
    pub embedding_dim: usize,
    /// Approximate heap bytes used by the vector store.
    pub store_memory_bytes: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// RagPipeline
// ─────────────────────────────────────────────────────────────────────────────

/// Full RAG pipeline: document indexing + context retrieval + prompt building.
///
/// # Example
///
/// ```rust
/// use oxibonsai_rag::embedding::IdentityEmbedder;
/// use oxibonsai_rag::pipeline::{RagConfig, RagPipeline};
///
/// let embedder = IdentityEmbedder::new(64).expect("valid dim");
/// let mut pipeline = RagPipeline::new(embedder, RagConfig::default());
///
/// pipeline.index_document("Rust is a systems programming language.").expect("failed to index document");
/// let prompt = pipeline.build_prompt("What is Rust?").expect("failed to build prompt");
/// assert!(prompt.contains("Question: What is Rust?"));
/// ```
pub struct RagPipeline<E: Embedder> {
    retriever: Retriever<E>,
    config: RagConfig,
}

impl<E: Embedder> RagPipeline<E> {
    /// Create a new pipeline with `embedder` and `config`.
    pub fn new(embedder: E, config: RagConfig) -> Self {
        let retriever = Retriever::new(embedder, config.retriever_config.clone());
        Self { retriever, config }
    }

    /// Index a single document.
    ///
    /// Returns the number of chunks that were stored, or an error if the
    /// document is empty or embedding fails.
    pub fn index_document(&mut self, text: &str) -> Result<usize, RagError> {
        self.retriever.add_document(text, &self.config.chunk_config)
    }

    /// Index multiple documents, returning per-document chunk counts.
    pub fn index_documents(&mut self, texts: &[&str]) -> Result<Vec<usize>, RagError> {
        self.retriever
            .add_documents(texts, &self.config.chunk_config)
    }

    /// Retrieve the most relevant context for `query` as a single string.
    ///
    /// Chunks are concatenated with `config.context_separator`.  The total
    /// length is capped at `config.max_context_chars`; chunks that would
    /// exceed this limit are dropped entirely (no partial truncation).
    pub fn retrieve_context(&self, query: &str) -> Result<String, RagError> {
        if query.trim().is_empty() {
            return Err(RagError::EmptyQuery);
        }

        let results = self.retriever.retrieve(query)?;
        let mut parts: Vec<&str> = Vec::with_capacity(results.len());
        let sep = &self.config.context_separator;
        let mut total_chars = 0usize;

        for result in &results {
            let text_len = result.chunk.text.len();
            let sep_len = if parts.is_empty() { 0 } else { sep.len() };
            if total_chars + sep_len + text_len > self.config.max_context_chars && !parts.is_empty()
            {
                break;
            }
            total_chars += sep_len + text_len;
            parts.push(&result.chunk.text);
        }

        debug!(
            chunks_used = parts.len(),
            context_chars = total_chars,
            "context assembled"
        );
        Ok(parts.join(sep))
    }

    /// Build a prompt by filling in `{context}` and `{query}` in the template.
    ///
    /// Returns [`RagError::EmptyQuery`] for blank queries.  If the vector store
    /// is empty, the context placeholder is replaced with an empty string
    /// (allowing the model to answer from prior knowledge).
    pub fn build_prompt(&self, query: &str) -> Result<String, RagError> {
        if query.trim().is_empty() {
            return Err(RagError::EmptyQuery);
        }

        let context = match self.retrieve_context(query) {
            Ok(ctx) => ctx,
            Err(RagError::NoDocumentsIndexed) => String::new(),
            Err(e) => return Err(e),
        };

        let prompt = self
            .config
            .prompt_template
            .replace("{context}", &context)
            .replace("{query}", query);

        Ok(prompt)
    }

    /// Retrieve a snapshot of pipeline statistics.
    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            documents_indexed: self.retriever.document_count(),
            chunks_indexed: self.retriever.chunk_count(),
            embedding_dim: self.retriever.embedder().embedding_dim(),
            store_memory_bytes: self.retriever.store().memory_usage_bytes(),
        }
    }

    /// Borrow the underlying retriever.
    pub fn retriever(&self) -> &Retriever<E> {
        &self.retriever
    }
}

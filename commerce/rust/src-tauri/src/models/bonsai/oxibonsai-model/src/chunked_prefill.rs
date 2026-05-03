//! Chunked prefill: process long prompts in smaller chunks.
//!
//! Instead of processing the entire prompt in one forward pass,
//! chunked prefill splits it into chunks and processes each sequentially.
//! This reduces peak memory from O(seq_len²) to O(chunk_size²).

/// Configuration for chunked prefill.
#[derive(Debug, Clone)]
pub struct ChunkedPrefillConfig {
    /// Maximum tokens per chunk (default: 512).
    pub chunk_size: usize,
    /// Whether to overlap chunks for better context.
    pub overlap: usize,
    /// Priority of prefill vs decode in scheduling.
    pub priority: PrefillPriority,
}

/// Scheduling priority between prefill and decode phases.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrefillPriority {
    /// Prefill all chunks before any decode (default).
    PrefillFirst,
    /// Interleave prefill chunks with decode steps.
    Interleaved,
    /// Decode has priority; prefill when idle.
    DecodePriority,
}

impl Default for ChunkedPrefillConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            overlap: 0,
            priority: PrefillPriority::PrefillFirst,
        }
    }
}

impl ChunkedPrefillConfig {
    /// Create a new config with the given chunk size.
    pub fn new(chunk_size: usize) -> Self {
        Self {
            chunk_size,
            ..Default::default()
        }
    }

    /// Set the overlap between consecutive chunks.
    pub fn with_overlap(mut self, overlap: usize) -> Self {
        self.overlap = overlap;
        self
    }

    /// Set the scheduling priority.
    pub fn with_priority(mut self, priority: PrefillPriority) -> Self {
        self.priority = priority;
        self
    }
}

/// A chunk of the prompt to prefill.
#[derive(Debug, Clone)]
pub struct PrefillChunk {
    /// Token IDs in this chunk.
    pub tokens: Vec<u32>,
    /// Start position in the original sequence.
    pub start_pos: usize,
    /// End position (exclusive) in the original sequence.
    pub end_pos: usize,
    /// Zero-based index of this chunk.
    pub chunk_index: usize,
    /// Whether this is the last chunk.
    pub is_last: bool,
}

impl PrefillChunk {
    /// Number of tokens in this chunk.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether this chunk is empty.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Split a prompt into prefill chunks.
///
/// When `overlap > 0`, consecutive chunks share `overlap` tokens at the boundary
/// so the model has context continuity. The stride is `chunk_size - overlap`.
pub fn create_prefill_chunks(
    prompt_tokens: &[u32],
    config: &ChunkedPrefillConfig,
) -> Vec<PrefillChunk> {
    if prompt_tokens.is_empty() {
        return vec![];
    }

    let chunk_size = config.chunk_size.max(1);
    let overlap = config.overlap.min(chunk_size.saturating_sub(1));
    let stride = chunk_size - overlap;

    let mut chunks = Vec::new();
    let total = prompt_tokens.len();
    let mut start = 0usize;
    let mut index = 0usize;

    while start < total {
        let end = (start + chunk_size).min(total);
        let tokens = prompt_tokens[start..end].to_vec();

        chunks.push(PrefillChunk {
            tokens,
            start_pos: start,
            end_pos: end,
            chunk_index: index,
            is_last: false, // fixed up below
        });

        index += 1;

        // Advance by stride, but if stride would not make progress (e.g.
        // overlap >= chunk_size), force at least 1 token forward.
        let advance = stride.max(1);
        start += advance;
    }

    // Mark the last chunk.
    if let Some(last) = chunks.last_mut() {
        last.is_last = true;
    }

    chunks
}

/// Action returned by the prefill scheduler.
#[derive(Debug, Clone)]
pub enum PrefillAction {
    /// Process the next prefill chunk.
    Prefill(PrefillChunk),
    /// All prefill done, proceed with decode.
    StartDecode,
    /// Yield to decode for one step (interleaved mode).
    YieldToDecode,
}

/// Prefill scheduling: determines the order of chunks and decode steps.
pub struct PrefillScheduler {
    config: ChunkedPrefillConfig,
    chunks: Vec<PrefillChunk>,
    current_chunk: usize,
    prefill_complete: bool,
    /// Tracks whether the last action was a prefill (for interleaved mode).
    last_was_prefill: bool,
}

impl PrefillScheduler {
    /// Create a new scheduler for the given prompt.
    pub fn new(prompt_tokens: &[u32], config: ChunkedPrefillConfig) -> Self {
        let chunks = create_prefill_chunks(prompt_tokens, &config);
        Self {
            config,
            chunks,
            current_chunk: 0,
            prefill_complete: false,
            last_was_prefill: false,
        }
    }

    /// Get the next action to perform.
    pub fn next_action(&mut self) -> PrefillAction {
        if self.prefill_complete || self.current_chunk >= self.chunks.len() {
            self.prefill_complete = true;
            return PrefillAction::StartDecode;
        }

        match self.config.priority {
            PrefillPriority::PrefillFirst => {
                let chunk = self.chunks[self.current_chunk].clone();
                self.current_chunk += 1;
                if self.current_chunk >= self.chunks.len() {
                    self.prefill_complete = true;
                }
                self.last_was_prefill = true;
                PrefillAction::Prefill(chunk)
            }
            PrefillPriority::Interleaved => {
                if self.last_was_prefill && self.current_chunk < self.chunks.len() {
                    // Yield after each prefill chunk.
                    self.last_was_prefill = false;
                    PrefillAction::YieldToDecode
                } else {
                    let chunk = self.chunks[self.current_chunk].clone();
                    self.current_chunk += 1;
                    if self.current_chunk >= self.chunks.len() {
                        self.prefill_complete = true;
                    }
                    self.last_was_prefill = true;
                    PrefillAction::Prefill(chunk)
                }
            }
            PrefillPriority::DecodePriority => {
                // In decode-priority mode, we still prefill but always yield
                // between chunks to let decode run first.
                if self.last_was_prefill {
                    self.last_was_prefill = false;
                    PrefillAction::YieldToDecode
                } else {
                    let chunk = self.chunks[self.current_chunk].clone();
                    self.current_chunk += 1;
                    if self.current_chunk >= self.chunks.len() {
                        self.prefill_complete = true;
                    }
                    self.last_was_prefill = true;
                    PrefillAction::Prefill(chunk)
                }
            }
        }
    }

    /// Report that decode can be performed (for interleaved mode).
    pub fn decode_available(&self) -> bool {
        !self.prefill_complete && self.config.priority != PrefillPriority::PrefillFirst
    }

    /// Whether all prefill chunks have been processed.
    pub fn is_complete(&self) -> bool {
        self.prefill_complete
    }

    /// Progress as fraction (0.0 - 1.0).
    pub fn progress(&self) -> f32 {
        if self.chunks.is_empty() {
            return 1.0;
        }
        self.current_chunk as f32 / self.chunks.len() as f32
    }

    /// Total number of chunks.
    pub fn total_chunks(&self) -> usize {
        self.chunks.len()
    }

    /// Estimate memory savings compared to full prefill.
    ///
    /// The dominant memory consumer in self-attention is the attention score
    /// matrix of shape `[num_heads, seq_len, seq_len]` (FP32). With chunked
    /// prefill the largest matrix is `[num_heads, chunk_size, chunk_size]`.
    pub fn memory_savings(&self, hidden_dim: usize) -> f32 {
        if self.chunks.is_empty() {
            return 0.0;
        }
        let total_tokens: usize = self.chunks.iter().map(|c| c.end_pos).max().unwrap_or(0);
        let chunk_size = self.config.chunk_size;
        if total_tokens == 0 || chunk_size == 0 {
            return 0.0;
        }
        let full = total_tokens as f64 * total_tokens as f64 * hidden_dim as f64;
        let chunked = chunk_size as f64 * chunk_size as f64 * hidden_dim as f64;
        if full == 0.0 {
            return 0.0;
        }
        1.0 - (chunked / full) as f32
    }
}

/// Estimate peak memory for chunked vs full prefill.
///
/// The estimate focuses on the attention score matrix which dominates memory
/// in transformer forward passes: `num_heads * seq_len * seq_len * 4` bytes
/// (FP32).
pub fn peak_memory_estimate(
    seq_len: usize,
    chunk_size: usize,
    _hidden_dim: usize,
    num_heads: usize,
) -> PrefillMemoryEstimate {
    let bytes_per_element = 4usize; // f32
    let full_prefill_bytes = num_heads * seq_len * seq_len * bytes_per_element;
    let effective_chunk = chunk_size.min(seq_len);
    let chunked_prefill_bytes = num_heads * effective_chunk * effective_chunk * bytes_per_element;

    let memory_savings_ratio = if full_prefill_bytes == 0 {
        0.0
    } else {
        1.0 - (chunked_prefill_bytes as f32 / full_prefill_bytes as f32)
    };

    let num_chunks = if chunk_size == 0 {
        0
    } else {
        seq_len.div_ceil(chunk_size)
    };

    PrefillMemoryEstimate {
        full_prefill_bytes,
        chunked_prefill_bytes,
        memory_savings_ratio,
        num_chunks,
    }
}

/// Memory estimate comparing full vs chunked prefill.
#[derive(Debug, Clone)]
pub struct PrefillMemoryEstimate {
    /// Peak memory for full (non-chunked) prefill in bytes.
    pub full_prefill_bytes: usize,
    /// Peak memory for chunked prefill in bytes.
    pub chunked_prefill_bytes: usize,
    /// Ratio of memory saved (0.0 = no savings, 1.0 = all saved).
    pub memory_savings_ratio: f32,
    /// Number of chunks needed.
    pub num_chunks: usize,
}

impl PrefillMemoryEstimate {
    /// Human-readable summary of the estimate.
    pub fn summary(&self) -> String {
        let full_mb = self.full_prefill_bytes as f64 / (1024.0 * 1024.0);
        let chunked_mb = self.chunked_prefill_bytes as f64 / (1024.0 * 1024.0);
        let pct = self.memory_savings_ratio * 100.0;
        format!(
            "Full prefill: {full_mb:.1} MB, Chunked: {chunked_mb:.1} MB \
             ({pct:.1}% savings, {n} chunks)",
            n = self.num_chunks,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default() {
        let cfg = ChunkedPrefillConfig::default();
        assert_eq!(cfg.chunk_size, 512);
        assert_eq!(cfg.overlap, 0);
        assert_eq!(cfg.priority, PrefillPriority::PrefillFirst);
    }

    #[test]
    fn config_builder() {
        let cfg = ChunkedPrefillConfig::new(256)
            .with_overlap(32)
            .with_priority(PrefillPriority::Interleaved);
        assert_eq!(cfg.chunk_size, 256);
        assert_eq!(cfg.overlap, 32);
        assert_eq!(cfg.priority, PrefillPriority::Interleaved);
    }

    #[test]
    fn empty_prompt() {
        let chunks = create_prefill_chunks(&[], &ChunkedPrefillConfig::default());
        assert!(chunks.is_empty());
    }
}

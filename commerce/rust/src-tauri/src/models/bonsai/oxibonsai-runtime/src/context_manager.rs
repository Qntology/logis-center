//! Context window management for multi-turn inference.
//!
//! Manages the token budget across conversation turns, supporting multiple
//! truncation strategies when context exceeds the model's maximum sequence length.
//!
//! ## Truncation Strategies
//!
//! - [`TruncationStrategy::TruncateLeft`] — drops the oldest conversation tokens (default).
//!   System prompt tokens are always preserved.
//! - [`TruncationStrategy::TruncateRight`] — drops the newest conversation tokens.
//! - [`TruncationStrategy::SlidingWindow`] — keeps system prompt + most recent tokens.
//!   Equivalent to `TruncateLeft` in this implementation.
//! - [`TruncationStrategy::Summarize`] — placeholder; falls back to `TruncateLeft`.
//!
//! ## Usage
//!
//! ```rust
//! use oxibonsai_runtime::context_manager::{ContextWindow, TruncationStrategy};
//!
//! let mut window = ContextWindow::new(2048, TruncationStrategy::TruncateLeft);
//! window.set_system_prompt(vec![1, 2, 3]).expect("system prompt fits");
//! window.append(&[10, 20, 30]);
//! let tokens = window.tokens();
//! assert!(tokens.len() <= 2048);
//! ```

// ──────────────────────────────────────────────────────────────────
// Truncation strategy
// ──────────────────────────────────────────────────────────────────

/// Strategy for handling context that exceeds the maximum token budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TruncationStrategy {
    /// Drop oldest conversation tokens (default). System prompt is never removed.
    TruncateLeft,
    /// Drop newest conversation tokens. System prompt is never removed.
    TruncateRight,
    /// Keep system prompt plus the most recent conversation tokens.
    /// In practice identical to `TruncateLeft` for system-prompt-first layouts.
    SlidingWindow,
    /// Placeholder for future LLM-based summarisation. Falls back to `TruncateLeft`.
    Summarize,
}

// ──────────────────────────────────────────────────────────────────
// Context error
// ──────────────────────────────────────────────────────────────────

/// Error type for context window operations.
#[derive(Debug)]
pub struct ContextError(String);

impl std::fmt::Display for ContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ContextError: {}", self.0)
    }
}

impl std::error::Error for ContextError {}

// ──────────────────────────────────────────────────────────────────
// ContextWindow
// ──────────────────────────────────────────────────────────────────

/// A fixed-capacity token window with configurable truncation.
///
/// The window stores a protected *system prompt* (never truncated) and
/// a mutable *conversation* segment. Together they must fit within
/// `max_tokens`.
pub struct ContextWindow {
    /// Maximum total token count (system + conversation).
    pub max_tokens: usize,
    /// Tokens belonging to the system prompt — never truncated.
    pub system_tokens: Vec<u32>,
    /// Accumulated conversation tokens.
    pub conversation: Vec<u32>,
    /// How to truncate when the window is full.
    pub strategy: TruncationStrategy,
}

impl ContextWindow {
    /// Create a new empty context window.
    pub fn new(max_tokens: usize, strategy: TruncationStrategy) -> Self {
        Self {
            max_tokens,
            system_tokens: Vec::new(),
            conversation: Vec::new(),
            strategy,
        }
    }

    /// Set the system prompt tokens.
    ///
    /// Returns an error if the system prompt alone exceeds `max_tokens`.
    pub fn set_system_prompt(&mut self, tokens: Vec<u32>) -> Result<(), ContextError> {
        if tokens.len() > self.max_tokens {
            return Err(ContextError(format!(
                "system prompt ({} tokens) exceeds max_tokens ({})",
                tokens.len(),
                self.max_tokens
            )));
        }
        self.system_tokens = tokens;
        // Truncate conversation if system prompt now leaves no room
        self.truncate_to_fit();
        Ok(())
    }

    /// Append tokens to the conversation.
    ///
    /// If the new tokens cause the window to overflow, truncation is applied
    /// *before* appending as much of the new tokens as will fit.
    ///
    /// Returns the number of tokens actually appended.
    pub fn append(&mut self, tokens: &[u32]) -> usize {
        self.conversation.extend_from_slice(tokens);
        let removed = self.truncate_to_fit();
        // How many of the newly added tokens survived truncation
        tokens.len().saturating_sub(removed)
    }

    /// Truncate the conversation segment to make the total fit within `max_tokens`.
    ///
    /// Applies the configured [`TruncationStrategy`].
    /// Returns the number of tokens removed from the conversation.
    pub fn truncate_to_fit(&mut self) -> usize {
        let capacity_for_conv = self.max_tokens.saturating_sub(self.system_tokens.len());
        if self.conversation.len() <= capacity_for_conv {
            return 0;
        }
        let excess = self.conversation.len() - capacity_for_conv;

        match self.strategy {
            TruncationStrategy::TruncateLeft
            | TruncationStrategy::SlidingWindow
            | TruncationStrategy::Summarize => {
                // Remove from the front (oldest tokens)
                self.conversation.drain(0..excess);
            }
            TruncationStrategy::TruncateRight => {
                // Remove from the back (newest tokens)
                let new_len = self.conversation.len() - excess;
                self.conversation.truncate(new_len);
            }
        }

        excess
    }

    /// Concatenate system tokens and conversation tokens into a single flat vector.
    ///
    /// The result is always within `max_tokens`.
    pub fn tokens(&self) -> Vec<u32> {
        let mut result = Vec::with_capacity(self.system_tokens.len() + self.conversation.len());
        result.extend_from_slice(&self.system_tokens);
        result.extend_from_slice(&self.conversation);
        result
    }

    /// Total token count (system + conversation).
    pub fn len(&self) -> usize {
        self.system_tokens.len() + self.conversation.len()
    }

    /// Returns `true` if both system and conversation are empty.
    pub fn is_empty(&self) -> bool {
        self.system_tokens.is_empty() && self.conversation.is_empty()
    }

    /// Number of additional tokens that can be appended before truncation.
    pub fn remaining_capacity(&self) -> usize {
        self.max_tokens.saturating_sub(self.len())
    }

    /// Returns `true` if the window is at or beyond its maximum capacity.
    pub fn is_at_limit(&self) -> bool {
        self.len() >= self.max_tokens
    }

    /// Clear all conversation tokens (system prompt is preserved).
    pub fn clear_conversation(&mut self) {
        self.conversation.clear();
    }

    /// Fraction of `max_tokens` currently in use: `len / max_tokens`.
    ///
    /// Returns 0.0 if `max_tokens` is zero.
    pub fn utilization(&self) -> f32 {
        if self.max_tokens == 0 {
            return 0.0;
        }
        self.len() as f32 / self.max_tokens as f32
    }
}

// ──────────────────────────────────────────────────────────────────
// ConversationTurn
// ──────────────────────────────────────────────────────────────────

/// A single turn in a multi-turn conversation.
pub struct ConversationTurn {
    /// Role identifier (e.g., `"user"`, `"assistant"`, `"system"`).
    pub role: String,
    /// Raw text content of this turn.
    pub content: String,
    /// Pre-tokenised representation of `content`.
    pub token_ids: Vec<u32>,
}

// ──────────────────────────────────────────────────────────────────
// ConversationContext
// ──────────────────────────────────────────────────────────────────

/// A multi-turn conversation with automatic context window management.
///
/// Each added turn is stored with its role, content, and token ids.
/// `build_tokens()` concatenates all turn token ids in order,
/// respecting the underlying [`ContextWindow`]'s token budget.
pub struct ConversationContext {
    window: ContextWindow,
    turns: Vec<ConversationTurn>,
}

impl ConversationContext {
    /// Create a new conversation context with the given maximum token budget.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            window: ContextWindow::new(max_tokens, TruncationStrategy::TruncateLeft),
            turns: Vec::new(),
        }
    }

    /// Add a conversation turn.
    ///
    /// The turn's token ids are appended to the context window.
    pub fn add_turn(&mut self, role: &str, content: &str, token_ids: Vec<u32>) {
        self.window.append(&token_ids);
        self.turns.push(ConversationTurn {
            role: role.to_string(),
            content: content.to_string(),
            token_ids,
        });
    }

    /// Build a flat token sequence from all turns, respecting the window budget.
    ///
    /// Concatenates token ids in turn order. The result is always within
    /// `max_tokens` after truncation.
    pub fn build_tokens(&self) -> Vec<u32> {
        self.window.tokens()
    }

    /// Number of turns added to this conversation.
    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    /// Total token count across the current context window (after truncation).
    pub fn total_tokens(&self) -> usize {
        self.window.len()
    }

    /// Returns `true` if the context window is at its maximum capacity.
    pub fn is_full(&self) -> bool {
        self.window.is_at_limit()
    }

    /// Clear all turns and reset the context window.
    pub fn clear(&mut self) {
        self.turns.clear();
        self.window.clear_conversation();
    }

    /// Reference to the most recently added turn, if any.
    pub fn last_turn(&self) -> Option<&ConversationTurn> {
        self.turns.last()
    }

    /// Utilisation of the token budget: `total_tokens / max_tokens`.
    pub fn utilization(&self) -> f32 {
        self.window.utilization()
    }
}

// ──────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_window_append_within_limit() {
        let mut window = ContextWindow::new(100, TruncationStrategy::TruncateLeft);
        let appended = window.append(&[1, 2, 3, 4, 5]);
        assert!(appended > 0, "should append tokens when within limit");
        assert_eq!(window.conversation.len(), 5);
        assert_eq!(window.len(), 5);
    }

    #[test]
    fn test_context_window_truncate_left() {
        let mut window = ContextWindow::new(5, TruncationStrategy::TruncateLeft);
        // Fill to capacity
        window.append(&[1, 2, 3, 4, 5]);
        assert_eq!(window.conversation.len(), 5);

        // Append more — oldest should be dropped
        window.append(&[6, 7]);
        assert_eq!(
            window.conversation.len(),
            5,
            "should still be at max after truncation"
        );
        // The newest tokens (6, 7) should be at the end
        let last = *window.conversation.last().expect("must have tokens");
        assert_eq!(last, 7, "newest token should be 7");
        // The oldest tokens (1, 2) should be gone
        assert!(
            !window.conversation.contains(&1),
            "token 1 should have been truncated"
        );
    }

    #[test]
    fn test_context_window_truncate_right() {
        let mut window = ContextWindow::new(5, TruncationStrategy::TruncateRight);
        window.append(&[1, 2, 3, 4, 5]);
        window.append(&[6, 7]);
        // Newest tokens (6, 7) should be dropped, oldest retained
        assert_eq!(window.conversation.len(), 5);
        assert_eq!(
            window.conversation[0], 1,
            "token 1 should be preserved with TruncateRight"
        );
        assert!(
            !window.conversation.contains(&6),
            "token 6 should have been truncated"
        );
    }

    #[test]
    fn test_context_window_system_prompt_preserved() {
        let mut window = ContextWindow::new(10, TruncationStrategy::TruncateLeft);
        window
            .set_system_prompt(vec![100, 200, 300])
            .expect("system prompt should fit");

        // Fill remaining capacity (7 slots)
        window.append(&[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(window.len(), 10);

        // Add more — system tokens must survive
        window.append(&[8, 9]);
        let tokens = window.tokens();
        assert_eq!(tokens.len(), 10);
        assert_eq!(tokens[0], 100, "system token 0 must be preserved");
        assert_eq!(tokens[1], 200, "system token 1 must be preserved");
        assert_eq!(tokens[2], 300, "system token 2 must be preserved");
    }

    #[test]
    fn test_context_window_remaining_capacity() {
        let mut window = ContextWindow::new(20, TruncationStrategy::TruncateLeft);
        assert_eq!(window.remaining_capacity(), 20);
        window.append(&[1, 2, 3]);
        assert_eq!(window.remaining_capacity(), 17);
        window.set_system_prompt(vec![10, 20]).expect("fits");
        // system (2) + conversation (3) = 5; remaining = 15
        assert_eq!(window.remaining_capacity(), 15);
    }

    #[test]
    fn test_context_window_system_prompt_too_large() {
        let mut window = ContextWindow::new(5, TruncationStrategy::TruncateLeft);
        let result = window.set_system_prompt(vec![1, 2, 3, 4, 5, 6]);
        assert!(
            result.is_err(),
            "system prompt larger than max_tokens should error"
        );
    }

    #[test]
    fn test_conversation_context_add_turn() {
        let mut ctx = ConversationContext::new(200);
        ctx.add_turn("user", "Hello!", vec![10, 20, 30]);
        ctx.add_turn("assistant", "Hi there!", vec![40, 50, 60, 70]);

        assert_eq!(ctx.turn_count(), 2);
        assert_eq!(ctx.total_tokens(), 7, "3 + 4 = 7 tokens total");

        let last = ctx.last_turn().expect("must have a last turn");
        assert_eq!(last.role, "assistant");
        assert_eq!(last.content, "Hi there!");
    }

    #[test]
    fn test_conversation_context_build_tokens() {
        let mut ctx = ConversationContext::new(100);
        ctx.add_turn("user", "A", vec![1, 2]);
        ctx.add_turn("assistant", "B", vec![3, 4, 5]);

        let tokens = ctx.build_tokens();
        assert_eq!(
            tokens,
            vec![1, 2, 3, 4, 5],
            "tokens should be in turn order"
        );
    }

    #[test]
    fn test_context_utilization() {
        let mut window = ContextWindow::new(100, TruncationStrategy::TruncateLeft);
        assert!(
            (window.utilization() - 0.0).abs() < f32::EPSILON,
            "empty window has 0.0 utilization"
        );
        window.append(&(0u32..50).collect::<Vec<_>>());
        assert!(
            (window.utilization() - 0.5).abs() < f32::EPSILON,
            "50/100 = 0.5 utilization"
        );
        window.append(&(0u32..50).collect::<Vec<_>>());
        assert!(
            (window.utilization() - 1.0).abs() < f32::EPSILON,
            "full window = 1.0 utilization"
        );
    }
}

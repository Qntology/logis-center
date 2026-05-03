//! Model hot-reload: update model weights without server restart.
//!
//! Uses a generation counter and atomic swap to enable zero-downtime
//! model updates. In-flight requests complete with the old model while the new
//! model is swapped in atomically.
//!
//! # Design
//!
//! The [`HotReloadCoordinator`] holds:
//! - An [`AtomicU64`] generation counter, advanced on every reload.
//! - An [`RwLock`]-protected history of [`ModelVersion`] snapshots.
//!
//! Callers that need to check whether the model has changed since they last
//! read it can compare their saved generation against [`HotReloadCoordinator::current_generation`].
//!
//! # Example
//!
//! ```rust
//! use oxibonsai_runtime::hot_reload::HotReloadCoordinator;
//!
//! let coord = HotReloadCoordinator::new();
//! assert_eq!(coord.current_generation(), 0);
//!
//! let gen = coord.record_reload("v1 weights loaded", Some("/models/v1.bin".to_string()));
//! assert_eq!(gen, 1);
//! assert_eq!(coord.current_generation(), 1);
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Instant;

// ─────────────────────────────────────────────────────────────────────────────
// Type alias
// ─────────────────────────────────────────────────────────────────────────────

/// The current generation of the loaded model.  Starts at 0 (no model loaded)
/// and is incremented by one on every successful reload.
type Generation = u64;

// ─────────────────────────────────────────────────────────────────────────────
// ModelVersion
// ─────────────────────────────────────────────────────────────────────────────

/// Metadata snapshot for a single model version.
#[derive(Debug, Clone)]
pub struct ModelVersion {
    /// Monotonically increasing generation counter for this version.
    pub generation: Generation,
    /// Filesystem path to the model weights file, if known.
    pub path: Option<String>,
    /// Wall-clock time at which this version was recorded.
    pub loaded_at: Instant,
    /// Free-form description (e.g. checkpoint name, commit hash).
    pub description: String,
}

impl ModelVersion {
    /// Create a new version snapshot with `loaded_at` set to now.
    pub fn new(generation: Generation, description: impl Into<String>) -> Self {
        Self {
            generation,
            path: None,
            loaded_at: Instant::now(),
            description: description.into(),
        }
    }

    /// Seconds elapsed since this version was loaded.
    pub fn age_seconds(&self) -> f64 {
        self.loaded_at.elapsed().as_secs_f64()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HotReloadCoordinator
// ─────────────────────────────────────────────────────────────────────────────

/// Hot-reload coordinator: manages atomic model generation swapping.
///
/// This type is cheap to clone (all fields are reference-counted) and
/// `Send + Sync`, so it can be shared freely across threads.
pub struct HotReloadCoordinator {
    /// Atomically readable current generation.
    current_generation: Arc<AtomicU64>,
    /// Full history of loaded versions, most-recent last internally,
    /// reversed on read via [`Self::version_history`].
    version_history: Arc<RwLock<Vec<ModelVersion>>>,
    /// Maximum number of version records to retain.
    max_history: usize,
}

impl HotReloadCoordinator {
    /// Create a coordinator with default max history (32 entries).
    pub fn new() -> Self {
        Self::with_max_history(32)
    }

    /// Create a coordinator that retains at most `max_history` version records.
    pub fn with_max_history(max_history: usize) -> Self {
        Self {
            current_generation: Arc::new(AtomicU64::new(0)),
            version_history: Arc::new(RwLock::new(Vec::new())),
            max_history,
        }
    }

    /// Record a new model version being loaded.
    ///
    /// Atomically increments the generation counter, appends a [`ModelVersion`]
    /// to the history (evicting the oldest if the history is full), and returns
    /// the new generation number.
    pub fn record_reload(
        &self,
        description: impl Into<String>,
        path: Option<String>,
    ) -> Generation {
        let new_gen = self.current_generation.fetch_add(1, Ordering::SeqCst) + 1;

        let version = ModelVersion {
            generation: new_gen,
            path,
            loaded_at: Instant::now(),
            description: description.into(),
        };

        let mut history = self
            .version_history
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());

        // Evict the oldest entry when the history is full.
        if self.max_history > 0 && history.len() >= self.max_history {
            history.remove(0);
        }
        history.push(version);

        new_gen
    }

    /// Return the current model generation (atomic, relaxed read).
    pub fn current_generation(&self) -> Generation {
        self.current_generation.load(Ordering::Relaxed)
    }

    /// Return the full version history, most-recent first.
    pub fn version_history(&self) -> Vec<ModelVersion> {
        let history = self
            .version_history
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut v: Vec<ModelVersion> = history.clone();
        v.reverse();
        v
    }

    /// Return the most recently recorded version, or `None` if no reload
    /// has been performed yet.
    pub fn current_version(&self) -> Option<ModelVersion> {
        let history = self
            .version_history
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        history.last().cloned()
    }

    /// Number of reloads performed (== length of the history buffer, capped
    /// at `max_history`).
    ///
    /// Note: this reflects the number of history records retained, not the
    /// total number of reloads ever performed.  Use [`Self::current_generation`]
    /// for a monotonically increasing reload count.
    pub fn reload_count(&self) -> usize {
        let history = self
            .version_history
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        history.len()
    }
}

impl Default for HotReloadCoordinator {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ReloadEvent
// ─────────────────────────────────────────────────────────────────────────────

/// A single reload notification recorded in a [`ReloadLog`].
#[derive(Debug, Clone)]
pub struct ReloadEvent {
    /// The generation that was replaced.
    pub old_generation: Generation,
    /// The generation that replaced it.
    pub new_generation: Generation,
    /// Human-readable description of the reload.
    pub description: String,
    /// Wall-clock time the event was recorded.
    pub timestamp: Instant,
}

// ─────────────────────────────────────────────────────────────────────────────
// ReloadLog
// ─────────────────────────────────────────────────────────────────────────────

/// A bounded, append-only log of [`ReloadEvent`]s.
///
/// When the log reaches its capacity, the oldest events are dropped (FIFO).
pub struct ReloadLog {
    events: Vec<ReloadEvent>,
    capacity: usize,
}

impl ReloadLog {
    /// Create a new log with the given maximum event capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            events: Vec::new(),
            capacity,
        }
    }

    /// Record a reload transition from `old` generation to `new` generation.
    ///
    /// If the log is at capacity the oldest event is removed first.
    pub fn record(&mut self, old: Generation, new: Generation, description: impl Into<String>) {
        if self.capacity > 0 && self.events.len() >= self.capacity {
            self.events.remove(0);
        }
        self.events.push(ReloadEvent {
            old_generation: old,
            new_generation: new,
            description: description.into(),
            timestamp: Instant::now(),
        });
    }

    /// Return references to the `n` most recent events (or all events if
    /// fewer than `n` are available).
    pub fn recent_events(&self, n: usize) -> Vec<&ReloadEvent> {
        let start = self.events.len().saturating_sub(n);
        self.events[start..].iter().collect()
    }

    /// Total number of events currently stored in the log.
    pub fn total_events(&self) -> usize {
        self.events.len()
    }

    /// Human-readable summary of the log.
    pub fn summary(&self) -> String {
        format!(
            "ReloadLog: {} events (capacity {})",
            self.events.len(),
            self.capacity,
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests (unit, inline)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinator_starts_at_zero() {
        let c = HotReloadCoordinator::new();
        assert_eq!(c.current_generation(), 0);
    }

    #[test]
    fn coordinator_record_increments() {
        let c = HotReloadCoordinator::new();
        let g1 = c.record_reload("first", None);
        let g2 = c.record_reload("second", None);
        assert_eq!(g1, 1);
        assert_eq!(g2, 2);
        assert_eq!(c.current_generation(), 2);
    }

    #[test]
    fn reload_log_basic() {
        let mut log = ReloadLog::new(10);
        assert_eq!(log.total_events(), 0);
        log.record(0, 1, "initial load");
        assert_eq!(log.total_events(), 1);
        assert!(!log.summary().is_empty());
    }
}

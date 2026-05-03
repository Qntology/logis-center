//! Bounded request queue with backpressure for the inference pipeline.
//!
//! [`BoundedQueue`] is a generic FIFO queue with a fixed capacity.  When the
//! queue is full, [`BoundedQueue::try_push`] returns `false` immediately,
//! allowing the caller to issue an HTTP 503 response rather than blocking
//! indefinitely.  [`BoundedQueue::push_timeout`] blocks for up to a given
//! [`Duration`] waiting for a slot to become available.
//!
//! [`InferenceQueue`] builds on top of [`BoundedQueue`] and wraps every
//! submitted work item with a one-shot [`std::sync::mpsc`] channel so that
//! callers can await the inference result asynchronously.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::sampling::SamplingParams;

// ─────────────────────────────────────────────────────────────────────────────
// QueueStats
// ─────────────────────────────────────────────────────────────────────────────

/// A serialisable snapshot of queue utilisation.
#[derive(Debug, Clone, serde::Serialize)]
pub struct QueueStats {
    /// Number of items currently waiting in the queue.
    pub len: usize,
    /// Maximum number of items the queue can hold.
    pub capacity: usize,
    /// `len / capacity` as a fraction in `[0.0, 1.0]`.
    pub utilization: f32,
    /// Total items ever successfully enqueued.
    pub total_enqueued: u64,
    /// Total items ever successfully dequeued.
    pub total_dequeued: u64,
    /// Total items dropped due to backpressure.
    pub total_dropped: u64,
    /// `total_dropped / (total_enqueued + total_dropped)`.
    pub drop_rate: f32,
}

// ─────────────────────────────────────────────────────────────────────────────
// BoundedQueue
// ─────────────────────────────────────────────────────────────────────────────

/// Thread-safe bounded FIFO queue with condvar-based blocking and backpressure.
///
/// The internal state is stored behind a `Mutex<VecDeque<(T, Instant)>>`.
/// Two [`Condvar`]s (`not_empty` and `not_full`) allow producers and consumers
/// to park efficiently instead of busy-waiting.
pub struct BoundedQueue<T> {
    /// Inner queue guarded by a mutex.
    queue: Mutex<VecDeque<(T, Instant)>>,
    /// Signalled whenever an item is pushed.
    not_empty: Condvar,
    /// Signalled whenever an item is popped.
    not_full: Condvar,
    /// Hard capacity limit.
    capacity: usize,
    /// Cumulative items enqueued without being dropped.
    pub total_enqueued: AtomicU64,
    /// Cumulative items dequeued.
    pub total_dequeued: AtomicU64,
    /// Cumulative items rejected due to a full queue.
    pub total_dropped: AtomicU64,
}

impl<T: Send> BoundedQueue<T> {
    /// Create a new bounded queue with the given maximum capacity.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "queue capacity must be at least 1");
        Self {
            queue: Mutex::new(VecDeque::with_capacity(capacity)),
            not_empty: Condvar::new(),
            not_full: Condvar::new(),
            capacity,
            total_enqueued: AtomicU64::new(0),
            total_dequeued: AtomicU64::new(0),
            total_dropped: AtomicU64::new(0),
        }
    }

    /// Attempt to push `item` without blocking.
    ///
    /// Returns `true` on success, `false` if the queue is already at capacity
    /// (backpressure: the caller should propagate a 503 to the client).
    pub fn try_push(&self, item: T) -> bool {
        let mut guard = self
            .queue
            .lock()
            .expect("queue mutex should not be poisoned");

        if guard.len() >= self.capacity {
            self.total_dropped.fetch_add(1, Ordering::Relaxed);
            return false;
        }

        guard.push_back((item, Instant::now()));
        self.total_enqueued.fetch_add(1, Ordering::Relaxed);
        self.not_empty.notify_one();
        true
    }

    /// Push `item`, blocking up to `timeout` for a free slot.
    ///
    /// Returns `true` if the item was enqueued before the timeout elapsed,
    /// `false` otherwise.
    pub fn push_timeout(&self, item: T, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;

        let mut guard = self
            .queue
            .lock()
            .expect("queue mutex should not be poisoned");

        loop {
            if guard.len() < self.capacity {
                guard.push_back((item, Instant::now()));
                self.total_enqueued.fetch_add(1, Ordering::Relaxed);
                self.not_empty.notify_one();
                return true;
            }

            let remaining = match deadline.checked_duration_since(Instant::now()) {
                Some(d) => d,
                None => {
                    self.total_dropped.fetch_add(1, Ordering::Relaxed);
                    return false;
                }
            };

            let (new_guard, timed_out) = self
                .not_full
                .wait_timeout(guard, remaining)
                .expect("queue condvar should not be poisoned");
            guard = new_guard;

            if timed_out.timed_out() {
                self.total_dropped.fetch_add(1, Ordering::Relaxed);
                return false;
            }
        }
    }

    /// Remove and return the oldest item without blocking.
    ///
    /// Returns `None` if the queue is empty.
    pub fn pop(&self) -> Option<T> {
        let mut guard = self
            .queue
            .lock()
            .expect("queue mutex should not be poisoned");

        guard.pop_front().map(|(item, _enqueued_at)| {
            self.total_dequeued.fetch_add(1, Ordering::Relaxed);
            self.not_full.notify_one();
            item
        })
    }

    /// Remove and return the oldest item, blocking up to `timeout` for one to
    /// become available.
    ///
    /// Returns `None` on timeout.
    pub fn pop_timeout(&self, timeout: Duration) -> Option<T> {
        let deadline = Instant::now() + timeout;

        let mut guard = self
            .queue
            .lock()
            .expect("queue mutex should not be poisoned");

        loop {
            if let Some((item, _)) = guard.pop_front() {
                self.total_dequeued.fetch_add(1, Ordering::Relaxed);
                self.not_full.notify_one();
                return Some(item);
            }

            let remaining = deadline.checked_duration_since(Instant::now())?;

            let (new_guard, timed_out) = self
                .not_empty
                .wait_timeout(guard, remaining)
                .expect("queue condvar should not be poisoned");
            guard = new_guard;

            if timed_out.timed_out() && guard.is_empty() {
                return None;
            }
        }
    }

    /// Current number of items in the queue.
    pub fn len(&self) -> usize {
        self.queue
            .lock()
            .expect("queue mutex should not be poisoned")
            .len()
    }

    /// `true` if the queue contains no items.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `true` if the queue is at capacity.
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Maximum number of items the queue can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current fill level as a fraction in `[0.0, 1.0]`.
    pub fn utilization(&self) -> f32 {
        self.len() as f32 / self.capacity as f32
    }

    /// Take a snapshot of queue statistics.
    pub fn stats(&self) -> QueueStats {
        let len = self.len();
        let enqueued = self.total_enqueued.load(Ordering::Relaxed);
        let dropped = self.total_dropped.load(Ordering::Relaxed);
        let attempted = enqueued + dropped;
        let drop_rate = if attempted == 0 {
            0.0
        } else {
            dropped as f32 / attempted as f32
        };

        QueueStats {
            len,
            capacity: self.capacity,
            utilization: len as f32 / self.capacity as f32,
            total_enqueued: enqueued,
            total_dequeued: self.total_dequeued.load(Ordering::Relaxed),
            total_dropped: dropped,
            drop_rate,
        }
    }

    /// Drain all items from the queue and return them in FIFO order.
    pub fn drain(&self) -> Vec<T> {
        let mut guard = self
            .queue
            .lock()
            .expect("queue mutex should not be poisoned");

        let count = guard.len();
        let items: Vec<T> = guard.drain(..).map(|(item, _)| item).collect();
        self.total_dequeued
            .fetch_add(count as u64, Ordering::Relaxed);
        self.not_full.notify_all();
        items
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InferenceWorkItem
// ─────────────────────────────────────────────────────────────────────────────

/// A single unit of work to be processed by the inference engine.
pub struct InferenceWorkItem {
    /// Unique monotonically-increasing request identifier.
    pub id: u64,
    /// Pre-tokenised prompt.
    pub prompt_tokens: Vec<u32>,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Sampling hyper-parameters for this request.
    pub params: SamplingParams,
    /// Wall-clock time at which this item was submitted to the queue.
    pub created_at: Instant,
    /// Channel through which the inference result is delivered.
    pub result_tx: std::sync::mpsc::SyncSender<Vec<u32>>,
}

impl InferenceWorkItem {
    /// How long this item has been waiting in the queue.
    pub fn wait_time(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Whether this item has been waiting longer than `ttl`.
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.wait_time() > ttl
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// InferenceQueue
// ─────────────────────────────────────────────────────────────────────────────

/// High-level inference request queue wrapping [`BoundedQueue<InferenceWorkItem>`].
///
/// Each call to [`InferenceQueue::submit`] returns a [`std::sync::mpsc::Receiver`]
/// through which the caller can retrieve the generated token IDs once inference
/// completes.  Returns `None` immediately if the queue is full (backpressure).
pub struct InferenceQueue {
    queue: Arc<BoundedQueue<InferenceWorkItem>>,
    next_id: AtomicU64,
}

impl InferenceQueue {
    /// Create a new inference queue with the given maximum pending-request capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: Arc::new(BoundedQueue::new(capacity)),
            next_id: AtomicU64::new(1),
        }
    }

    /// Submit an inference request.
    ///
    /// Returns a `Receiver` that will yield the generated token IDs once the
    /// request is processed, or `None` if the queue is full.
    pub fn submit(
        &self,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        params: SamplingParams,
    ) -> Option<std::sync::mpsc::Receiver<Vec<u32>>> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);

        let item = InferenceWorkItem {
            id,
            prompt_tokens,
            max_tokens,
            params,
            created_at: Instant::now(),
            result_tx: tx,
        };

        if self.queue.try_push(item) {
            Some(rx)
        } else {
            None
        }
    }

    /// Number of pending requests currently in the queue.
    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    /// `true` if the queue is at capacity.
    pub fn is_full(&self) -> bool {
        self.queue.is_full()
    }

    /// Take a statistics snapshot.
    pub fn stats(&self) -> QueueStats {
        self.queue.stats()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    // ── BoundedQueue ──────────────────────────────────────────────────────

    #[test]
    fn test_bounded_queue_try_push() {
        let q: BoundedQueue<u32> = BoundedQueue::new(4);
        assert!(q.try_push(1));
        assert!(q.try_push(2));
        assert_eq!(q.len(), 2);
        assert_eq!(q.total_enqueued.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_bounded_queue_try_push_full_returns_false() {
        let q: BoundedQueue<u32> = BoundedQueue::new(2);
        assert!(q.try_push(10));
        assert!(q.try_push(20));
        // Queue is now full — further push must fail.
        assert!(!q.try_push(30));
        assert_eq!(q.total_dropped.load(Ordering::Relaxed), 1);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_bounded_queue_pop_empty_returns_none() {
        let q: BoundedQueue<u32> = BoundedQueue::new(4);
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn test_bounded_queue_fifo_order() {
        let q: BoundedQueue<u32> = BoundedQueue::new(8);
        for i in 0..5u32 {
            assert!(q.try_push(i));
        }
        for expected in 0..5u32 {
            assert_eq!(q.pop(), Some(expected));
        }
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn test_bounded_queue_stats() {
        let q: BoundedQueue<u32> = BoundedQueue::new(4);
        q.try_push(1);
        q.try_push(2);
        q.pop();

        let stats = q.stats();
        assert_eq!(stats.capacity, 4);
        assert_eq!(stats.len, 1);
        assert_eq!(stats.total_enqueued, 2);
        assert_eq!(stats.total_dequeued, 1);
        assert_eq!(stats.total_dropped, 0);
        assert!((stats.utilization - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn test_bounded_queue_drain() {
        let q: BoundedQueue<u32> = BoundedQueue::new(8);
        for i in 0..4u32 {
            q.try_push(i);
        }
        let items = q.drain();
        assert_eq!(items, vec![0, 1, 2, 3]);
        assert_eq!(q.len(), 0);
        assert_eq!(q.total_dequeued.load(Ordering::Relaxed), 4);
    }

    // ── InferenceQueue ────────────────────────────────────────────────────

    #[test]
    fn test_inference_queue_submit_and_depth() {
        let iq = InferenceQueue::new(8);
        let _rx1 = iq
            .submit(vec![1, 2, 3], 16, SamplingParams::default())
            .expect("submit should succeed on an empty queue");
        let _rx2 = iq
            .submit(vec![4, 5, 6], 16, SamplingParams::default())
            .expect("second submit should succeed");

        assert_eq!(iq.queue_depth(), 2);
        assert!(!iq.is_full());
    }

    #[test]
    fn test_inference_queue_full_returns_none() {
        let iq = InferenceQueue::new(2);

        let _rx1 = iq
            .submit(vec![1], 8, SamplingParams::default())
            .expect("first submit");
        let _rx2 = iq
            .submit(vec![2], 8, SamplingParams::default())
            .expect("second submit");

        // Queue is now full — next submit must return None.
        assert!(iq.is_full());
        let result = iq.submit(vec![3], 8, SamplingParams::default());
        assert!(result.is_none(), "submit to a full queue must return None");
    }
}

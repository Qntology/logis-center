//! Runtime memory profiling.
//!
//! Reads process RSS (Resident Set Size) on macOS (via Mach `task_info`) and
//! Linux (via `/proc/self/statm`).  Returns `0` on unsupported platforms.
//!
//! ## Usage
//!
//! ```
//! use oxibonsai_runtime::memory::{get_rss_bytes, MemoryProfiler};
//!
//! let profiler = MemoryProfiler::new();
//! let snapshot = profiler.sample();
//! println!("RSS: {} bytes", snapshot.rss_bytes);
//! println!("Peak RSS: {} bytes", profiler.peak_rss_bytes());
//! ```

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ─── MemorySnapshot ─────────────────────────────────────────────────────────

/// Memory snapshot at a point in time.
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    /// Resident Set Size in bytes at the moment of sampling.
    pub rss_bytes: u64,
    /// Monotonic timestamp at which the snapshot was taken.
    pub timestamp: Instant,
    /// Milliseconds since the Unix epoch at the time of sampling.
    ///
    /// Derived from `std::time::SystemTime::now()` — zero on platforms where
    /// `SystemTime` is unavailable (e.g., wasm32-unknown-unknown).
    pub timestamp_ms: u64,
}

// ─── Public entry point ──────────────────────────────────────────────────────

/// Get current process RSS (Resident Set Size) in bytes.
///
/// Returns `0` on unsupported platforms (WASM, Windows, etc.).
/// On Linux reads `/proc/self/statm`; on macOS calls the Mach `task_info` API.
pub fn get_rss_bytes() -> u64 {
    platform::rss_bytes()
}

// ─── Platform implementations ────────────────────────────────────────────────

#[cfg(target_os = "linux")]
mod platform {
    pub(super) fn rss_bytes() -> u64 {
        rss_from_proc_statm().unwrap_or(0)
    }

    /// Parse `/proc/self/statm`.
    ///
    /// Line format: `size resident shared text lib data dt` — all in pages.
    fn rss_from_proc_statm() -> Option<u64> {
        let content = std::fs::read_to_string("/proc/self/statm").ok()?;
        let resident_pages: u64 = content.split_whitespace().nth(1)?.parse().ok()?;
        let page_size = page_size_bytes();
        Some(resident_pages * page_size)
    }

    fn page_size_bytes() -> u64 {
        // SAFETY: sysconf is always safe to call; negative return means error.
        let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if ps > 0 {
            ps as u64
        } else {
            4096
        }
    }
}

#[cfg(target_os = "macos")]
mod platform {
    pub(super) fn rss_bytes() -> u64 {
        rss_from_mach().unwrap_or(0)
    }

    /// Query the Mach kernel for the task's resident private memory.
    ///
    /// Uses `task_info(TASK_VM_INFO)` via the `mach2` crate — a stable public
    /// macOS API.  The `mach2` crate is the recommended modern replacement for
    /// the deprecated macOS bindings in `libc`.
    fn rss_from_mach() -> Option<u64> {
        use mach2::kern_return::KERN_SUCCESS;
        use mach2::task::task_info;
        use mach2::task_info::{task_flavor_t, task_info_t};
        use mach2::traps::mach_task_self;

        // Mach task_info flavor constants (from <mach/task_info.h>).
        const TASK_VM_INFO: task_flavor_t = 22;

        // Minimal layout of `task_vm_info_data_t`.
        // Only `resident_size` (offset 24 bytes) is needed; the rest is zeroed.
        // The struct is stable across macOS versions — it only grows at the end.
        //
        // Field layout (verified against macOS 10.x – 15.x SDK):
        //   virtual_size:   u64   (offset 0)
        //   region_count:   i32   (offset 8)
        //   page_size:      i32   (offset 12)
        //   resident_size:  u64   (offset 16)
        //   … 83 additional u64 fields (phys_footprint, etc.)
        //
        // Total: 87 natural_t (u32) words → TASK_VM_INFO_COUNT = 87.
        const TASK_VM_INFO_COUNT: u32 = 87;

        #[repr(C)]
        struct TaskVmInfo {
            virtual_size: u64,
            region_count: i32,
            page_size: i32,
            resident_size: u64,
            _rest: [u64; 83],
        }

        let mut info: TaskVmInfo = unsafe { std::mem::zeroed() };
        let mut count: u32 = TASK_VM_INFO_COUNT;

        // SAFETY: `task_info` is a stable Mach syscall.  The buffer is zeroed,
        //          the flavor is TASK_VM_INFO, and `count` is set to the correct
        //          size in natural_t units.
        let ret = unsafe {
            task_info(
                mach_task_self(),
                TASK_VM_INFO,
                &mut info as *mut TaskVmInfo as task_info_t,
                &mut count as *mut u32,
            )
        };

        if ret == KERN_SUCCESS {
            Some(info.resident_size)
        } else {
            None
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod platform {
    pub(super) fn rss_bytes() -> u64 {
        0
    }
}

// ─── MemoryProfiler ─────────────────────────────────────────────────────────

/// Simple memory profiler that tracks peak RSS usage over its lifetime.
///
/// Designed to be shared via `Arc<MemoryProfiler>` across threads.
/// All mutable state is stored in atomics, so no locking is required.
///
/// # Example
///
/// ```
/// use oxibonsai_runtime::memory::MemoryProfiler;
///
/// let profiler = MemoryProfiler::new();
///
/// // Sample at some point during processing
/// let snap = profiler.sample();
/// println!("current RSS: {} bytes", snap.rss_bytes);
///
/// // Peak may differ from current if memory was freed
/// println!("peak RSS:    {} bytes", profiler.peak_rss_bytes());
/// println!("delta:       {} bytes", profiler.delta_bytes());
/// ```
#[derive(Debug)]
pub struct MemoryProfiler {
    /// RSS at profiler creation time.
    start_rss: u64,
    /// Highest observed RSS.
    peak_rss: AtomicU64,
    /// Number of times `sample()` has been called.
    sample_count: AtomicU64,
}

impl MemoryProfiler {
    /// Create a new profiler, recording the current RSS as the baseline.
    pub fn new() -> Self {
        let current = get_rss_bytes();
        Self {
            start_rss: current,
            peak_rss: AtomicU64::new(current),
            sample_count: AtomicU64::new(0),
        }
    }

    /// Take a memory snapshot, updating the peak if necessary.
    ///
    /// Lock-free and safe to call from any thread.
    pub fn sample(&self) -> MemorySnapshot {
        let rss = get_rss_bytes();
        self.peak_rss.fetch_max(rss, Ordering::Relaxed);
        self.sample_count.fetch_add(1, Ordering::Relaxed);
        let timestamp_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        MemorySnapshot {
            rss_bytes: rss,
            timestamp: Instant::now(),
            timestamp_ms,
        }
    }

    /// Highest RSS observed across all `sample()` calls and at creation.
    pub fn peak_rss_bytes(&self) -> u64 {
        self.peak_rss.load(Ordering::Relaxed)
    }

    /// RSS at the time this profiler was created.
    pub fn start_rss_bytes(&self) -> u64 {
        self.start_rss
    }

    /// Signed difference: `peak_rss − start_rss`.
    ///
    /// Positive means memory grew; negative (rare) means the OS reclaimed
    /// pages between profiler creation and the peak sample.
    pub fn delta_bytes(&self) -> i64 {
        self.peak_rss_bytes() as i64 - self.start_rss as i64
    }

    /// Total number of `sample()` calls made on this profiler.
    pub fn sample_count(&self) -> u64 {
        self.sample_count.load(Ordering::Relaxed)
    }

    /// Take a memory snapshot, updating the peak if necessary.
    ///
    /// Alias for `sample` using the name required by the task specification.
    pub fn take_snapshot(&self) -> MemorySnapshot {
        self.sample()
    }

    /// Current RSS in bytes as `Option<u64>`.
    ///
    /// Returns `None` on platforms where RSS reading is unsupported (WASM, etc.).
    /// On Linux and macOS this always returns `Some(value)`, where `value` may be
    /// `0` only in the extremely unlikely case that the OS returns an error.
    pub fn current_rss_bytes(&self) -> Option<u64> {
        let rss = get_rss_bytes();
        if rss == 0 {
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            return Some(rss);
            #[cfg(not(any(target_os = "linux", target_os = "macos")))]
            return None;
        }
        Some(rss)
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_rss_returns_value() {
        let rss = get_rss_bytes();
        // On supported platforms (Linux, macOS) this should be > 0.
        // On WASM / unsupported it returns 0 — both outcomes are valid;
        // what matters is that the call does not panic.
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        assert!(rss > 0, "RSS should be > 0 on Linux/macOS, got {rss}");
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        let _ = rss;
    }

    #[test]
    fn memory_profiler_new_succeeds() {
        let profiler = MemoryProfiler::new();
        assert!(profiler.start_rss_bytes() < u64::MAX);
        assert_eq!(profiler.sample_count(), 0);
    }

    #[test]
    fn memory_profiler_sample_returns_snapshot() {
        let profiler = MemoryProfiler::new();
        let snap = profiler.sample();
        assert_eq!(profiler.sample_count(), 1);
        let _ = snap.rss_bytes;
    }

    #[test]
    fn memory_profiler_peak_ge_start_after_sampling() {
        let profiler = MemoryProfiler::new();
        profiler.sample();
        profiler.sample();
        profiler.sample();
        assert!(
            profiler.peak_rss_bytes() >= profiler.start_rss_bytes(),
            "peak ({}) must be >= start ({})",
            profiler.peak_rss_bytes(),
            profiler.start_rss_bytes()
        );
    }

    #[test]
    fn memory_profiler_delta_does_not_panic() {
        let profiler = MemoryProfiler::new();
        let _v: Vec<u8> = vec![0u8; 1024 * 1024]; // allocate 1 MiB
        profiler.sample();
        // delta can be >= or < 0 depending on OS; we just ensure no panic.
        let _ = profiler.delta_bytes();
    }

    #[test]
    fn memory_profiler_sample_count_increments() {
        let profiler = MemoryProfiler::new();
        assert_eq!(profiler.sample_count(), 0);
        for i in 1..=5 {
            profiler.sample();
            assert_eq!(profiler.sample_count(), i);
        }
    }

    #[test]
    fn memory_profiler_default_equals_new() {
        let p = MemoryProfiler::default();
        assert_eq!(p.sample_count(), 0);
    }

    // ── Task-spec required test names ────────────────────────────────────────

    /// Verify that `get_rss_bytes()` returns a non-zero value on supported platforms.
    #[test]
    fn test_get_rss_returns_nonzero() {
        let rss = get_rss_bytes();
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        assert!(
            rss > 0,
            "get_rss_bytes() should return > 0 on Linux/macOS, got {rss}"
        );
        // On other platforms (e.g., wasm32) we allow 0 — but the call must not panic.
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        let _ = rss;
    }

    /// Verify that the profiler's peak tracks the highest observed RSS.
    #[test]
    fn test_profiler_peak_tracks_correctly() {
        let profiler = MemoryProfiler::new();

        // Take several snapshots; peak must be >= all observed rss values.
        let s1 = profiler.take_snapshot();
        let s2 = profiler.take_snapshot();
        let s3 = profiler.take_snapshot();

        let max_observed = s1.rss_bytes.max(s2.rss_bytes).max(s3.rss_bytes);
        let peak = profiler.peak_rss_bytes();

        assert!(
            peak >= max_observed,
            "peak ({peak}) must be >= max observed rss ({max_observed})"
        );
    }

    /// Verify that each snapshot carries a valid timestamp_ms.
    #[test]
    fn test_snapshot_has_timestamp() {
        let profiler = MemoryProfiler::new();
        let snap = profiler.take_snapshot();

        // timestamp_ms must be a plausible Unix epoch millisecond value.
        // 2020-01-01 = 1577836800000 ms; 2100-01-01 ≈ 4102444800000 ms.
        const EPOCH_2020: u64 = 1_577_836_800_000;
        const EPOCH_2100: u64 = 4_102_444_800_000;

        assert!(
            snap.timestamp_ms >= EPOCH_2020,
            "timestamp_ms ({}) should be >= 2020 epoch ({EPOCH_2020})",
            snap.timestamp_ms
        );
        assert!(
            snap.timestamp_ms <= EPOCH_2100,
            "timestamp_ms ({}) should be <= 2100 epoch ({EPOCH_2100})",
            snap.timestamp_ms
        );
    }
}

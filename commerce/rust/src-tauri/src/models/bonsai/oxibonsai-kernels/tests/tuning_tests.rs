//! Integration tests for platform tuning, aligned allocations, and prefetch hints.

use oxibonsai_kernels::aligned::{align_to_cache_line, AlignedBlocks, AlignedBuffer, ALIGNMENT};
use oxibonsai_kernels::prefetch::{
    prefetch_range_read, prefetch_read, prefetch_write, PrefetchConfig, PrefetchLocality,
    PrefetchStrategy,
};
use oxibonsai_kernels::tuning::{PlatformProfile, TuningSummary};

// ── PlatformProfile detection tests ─────────────────────────────────────

#[test]
fn platform_detect_has_positive_cores() {
    let p = PlatformProfile::detect();
    assert!(p.logical_cores > 0, "must detect at least 1 logical core");
    assert!(p.physical_cores > 0, "must detect at least 1 physical core");
}

#[test]
fn platform_detect_physical_leq_logical() {
    let p = PlatformProfile::detect();
    assert!(
        p.physical_cores <= p.logical_cores,
        "physical ({}) must not exceed logical ({})",
        p.physical_cores,
        p.logical_cores
    );
}

#[test]
fn platform_detect_cache_positive() {
    let p = PlatformProfile::detect();
    assert_eq!(p.cache_line_bytes, 64);
    assert!(p.l1_cache_bytes > 0, "L1 cache must be > 0");
    assert!(p.l2_cache_bytes > 0, "L2 cache must be > 0");
    assert!(
        p.l2_cache_bytes >= p.l1_cache_bytes,
        "L2 ({}) should be >= L1 ({})",
        p.l2_cache_bytes,
        p.l1_cache_bytes
    );
}

#[test]
fn platform_detect_simd_flags_consistent() {
    let p = PlatformProfile::detect();
    // On this platform (either x86 or ARM), at most one family should be active
    // (can't have both AVX2 and NEON on the same CPU)
    if p.has_neon {
        assert!(!p.has_avx2, "NEON and AVX2 are mutually exclusive");
        assert!(!p.has_avx512, "NEON and AVX-512 are mutually exclusive");
    }
}

// ── Threshold computation tests ─────────────────────────────────────────

#[test]
fn thresholds_for_various_core_counts() {
    let core_counts = [1, 2, 4, 8, 16, 32, 64];
    let mut prev_gemv = usize::MAX;
    let mut prev_gemm = usize::MAX;

    for &cores in &core_counts {
        let p = PlatformProfile::with_cores(cores, cores);
        let t = p.compute_thresholds();

        // Monotonically non-increasing as cores increase
        assert!(
            t.par_gemv_min_rows <= prev_gemv,
            "gemv threshold should not increase with more cores: {} cores => {}, prev was {}",
            cores,
            t.par_gemv_min_rows,
            prev_gemv
        );
        assert!(
            t.par_gemm_min_batch <= prev_gemm,
            "gemm threshold should not increase with more cores: {} cores => {}, prev was {}",
            cores,
            t.par_gemm_min_batch,
            prev_gemm
        );

        prev_gemv = t.par_gemv_min_rows;
        prev_gemm = t.par_gemm_min_batch;
    }
}

#[test]
fn threshold_block_sizes_multiple_of_8() {
    let p = PlatformProfile::detect();
    let t = p.compute_thresholds();
    assert_eq!(t.tiled_gemm_block_m % 8, 0, "block_m not multiple of 8");
    assert_eq!(t.tiled_gemm_block_n % 8, 0, "block_n not multiple of 8");
    assert_eq!(
        t.tiled_gemm_block_k % 128,
        0,
        "block_k not multiple of 128 (group size)"
    );
}

#[test]
fn threshold_block_k_at_least_128() {
    let p = PlatformProfile::with_cache(8 * 1024, 64 * 1024); // small cache
    let t = p.compute_thresholds();
    assert!(
        t.tiled_gemm_block_k >= 128,
        "block_k ({}) must be at least one group (128)",
        t.tiled_gemm_block_k
    );
}

#[test]
fn should_parallelize_gemv_below_threshold() {
    let p = PlatformProfile::with_cores(4, 4);
    let t = p.compute_thresholds();
    assert!(
        !t.should_parallelize_gemv(1),
        "should not parallelize 1 row"
    );
    assert!(
        !t.should_parallelize_gemv(t.par_gemv_min_rows - 1),
        "should not parallelize just below threshold"
    );
}

#[test]
fn should_parallelize_gemv_at_threshold() {
    let p = PlatformProfile::with_cores(4, 4);
    let t = p.compute_thresholds();
    assert!(
        t.should_parallelize_gemv(t.par_gemv_min_rows),
        "should parallelize at threshold"
    );
}

#[test]
fn should_parallelize_gemm_decisions() {
    let p = PlatformProfile::with_cores(8, 8);
    let t = p.compute_thresholds();
    assert!(!t.should_parallelize_gemm(1));
    assert!(t.should_parallelize_gemm(t.par_gemm_min_batch));
    assert!(t.should_parallelize_gemm(t.par_gemm_min_batch + 100));
}

// ── Global profile caching tests ────────────────────────────────────────

#[test]
fn global_profile_returns_same_values() {
    let p1 = PlatformProfile::global();
    let p2 = PlatformProfile::global();
    assert_eq!(p1.logical_cores, p2.logical_cores);
    assert_eq!(p1.physical_cores, p2.physical_cores);
    assert_eq!(p1.l1_cache_bytes, p2.l1_cache_bytes);
}

#[test]
fn global_thresholds_consistent() {
    let t1 = PlatformProfile::global_thresholds();
    let t2 = PlatformProfile::global_thresholds();
    assert_eq!(t1.par_gemv_min_rows, t2.par_gemv_min_rows);
    assert_eq!(t1.par_gemm_min_batch, t2.par_gemm_min_batch);
}

// ── TuningSummary tests ─────────────────────────────────────────────────

#[test]
fn tuning_summary_display_contains_key_info() {
    let summary = TuningSummary::current();
    let text = format!("{summary}");
    assert!(text.contains("Platform Tuning Summary"));
    assert!(text.contains("Cores:"));
    assert!(text.contains("Cache:"));
    assert!(text.contains("SIMD:"));
    assert!(text.contains("par_gemv_min_rows:"));
    assert!(text.contains("par_gemm_min_batch:"));
    assert!(text.contains("tiled_gemm_block:"));
}

// ── AlignedBuffer tests ─────────────────────────────────────────────────

#[test]
fn aligned_buffer_allocation_and_access() {
    let buf = AlignedBuffer::new(128);
    assert_eq!(buf.len(), 128);
    assert!(!buf.is_empty());
    // Verify zero-initialized
    for &v in buf.as_slice() {
        assert!((v - 0.0).abs() < f32::EPSILON);
    }
}

#[test]
fn aligned_buffer_alignment_verified() {
    let buf = AlignedBuffer::new(256);
    let ptr_val = buf.as_ptr() as usize;
    assert_eq!(
        ptr_val % ALIGNMENT,
        0,
        "pointer {ptr_val:#x} not 64-byte aligned"
    );
}

#[test]
fn aligned_buffer_zero_length_safe() {
    let buf = AlignedBuffer::new(0);
    assert_eq!(buf.len(), 0);
    assert!(buf.is_empty());
    assert_eq!(buf.as_slice().len(), 0);
    // Drop should not crash
}

#[test]
fn aligned_buffer_large_allocation() {
    let buf = AlignedBuffer::new(10_000);
    assert_eq!(buf.len(), 10_000);
    assert_eq!(buf.as_ptr() as usize % ALIGNMENT, 0);
    // Verify we can write and read back
    let slice = buf.as_slice();
    assert_eq!(slice.len(), 10_000);
}

#[test]
fn aligned_buffer_write_and_read() {
    let mut buf = AlignedBuffer::new(64);
    {
        let s = buf.as_mut_slice();
        // Simple LCG for deterministic test data (no rand crate)
        let mut state: u64 = 0xDEAD_BEEF;
        for elem in s.iter_mut() {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *elem = (state >> 33) as f32 / 1000.0;
        }
    }
    // Verify non-zero data was written
    let non_zero_count = buf
        .as_slice()
        .iter()
        .filter(|&&v| v.abs() > f32::EPSILON)
        .count();
    assert!(
        non_zero_count > 0,
        "should have written some non-zero values"
    );
}

// ── AlignedBlocks tests ─────────────────────────────────────────────────

#[test]
fn aligned_blocks_allocation() {
    let blocks = AlignedBlocks::new(16);
    assert_eq!(blocks.len(), 16);
    assert!(!blocks.is_empty());
    assert_eq!(blocks.as_ptr() as usize % ALIGNMENT, 0);
}

#[test]
fn aligned_blocks_zero_length_safe() {
    let blocks = AlignedBlocks::new(0);
    assert_eq!(blocks.len(), 0);
    assert!(blocks.is_empty());
    assert_eq!(blocks.as_slice().len(), 0);
}

// ── Cache-line split tests ──────────────────────────────────────────────

#[test]
fn cache_line_split_preserves_length() {
    let buf = AlignedBuffer::new(200);
    let data = buf.as_slice();
    let (prefix, aligned, suffix) = align_to_cache_line(data);
    assert_eq!(
        prefix.len() + aligned.len() + suffix.len(),
        data.len(),
        "split must preserve total element count"
    );
}

#[test]
fn cache_line_split_aligned_buffer() {
    let buf = AlignedBuffer::new(128);
    let (prefix, aligned, suffix) = align_to_cache_line(buf.as_slice());
    // Buffer is already aligned, so prefix should be empty
    assert!(
        prefix.is_empty(),
        "prefix should be empty for aligned allocation"
    );
    assert_eq!(aligned.len() + suffix.len(), 128);
}

#[test]
fn cache_line_split_empty() {
    let empty: &[f32] = &[];
    let (p, a, s) = align_to_cache_line(empty);
    assert!(p.is_empty());
    assert!(a.is_empty());
    assert!(s.is_empty());
}

// ── Prefetch smoke tests ────────────────────────────────────────────────

#[test]
fn prefetch_config_defaults() {
    let config = PrefetchConfig::default();
    assert_eq!(config.lookahead_blocks, 4);
    assert_eq!(config.strategy, PrefetchStrategy::Temporal);
}

#[test]
fn prefetch_read_does_not_crash() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    prefetch_read(data.as_ptr(), PrefetchLocality::High);
    prefetch_read(data.as_ptr(), PrefetchLocality::Medium);
    prefetch_read(data.as_ptr(), PrefetchLocality::Low);
}

#[test]
fn prefetch_write_does_not_crash() {
    let mut data = [0.0f32; 32];
    prefetch_write(data.as_mut_ptr(), PrefetchLocality::High);
    data[0] = 1.0;
    assert!((data[0] - 1.0).abs() < f32::EPSILON);
}

#[test]
fn prefetch_range_read_does_not_crash() {
    let data = vec![0.0f32; 4096];
    let byte_count = data.len() * std::mem::size_of::<f32>();
    prefetch_range_read(data.as_ptr(), byte_count, PrefetchLocality::High);
    prefetch_range_read(data.as_ptr(), byte_count, PrefetchLocality::Low);
}

#[test]
fn prefetch_strategy_variants() {
    assert_eq!(PrefetchStrategy::None, PrefetchStrategy::None);
    assert_ne!(PrefetchStrategy::Temporal, PrefetchStrategy::NonTemporal);
    assert_ne!(PrefetchStrategy::None, PrefetchStrategy::Temporal);
}

#[test]
fn prefetch_gemm_config_scales_with_batch() {
    let small = PrefetchConfig::for_gemm(2);
    let large = PrefetchConfig::for_gemm(64);
    assert_eq!(small.strategy, PrefetchStrategy::Temporal);
    assert_eq!(large.strategy, PrefetchStrategy::NonTemporal);
    assert!(large.lookahead_blocks >= small.lookahead_blocks);
}

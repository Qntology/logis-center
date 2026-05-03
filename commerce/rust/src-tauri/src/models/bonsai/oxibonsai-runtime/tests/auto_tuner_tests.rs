use oxibonsai_runtime::auto_tuner::{AutoTuner, CpuFeatures, KvCacheType, MemoryBudget, SimdTier};

// ---------------------------------------------------------------------------
// CpuFeatures
// ---------------------------------------------------------------------------

#[test]
fn cpu_features_detect() {
    let _cpu = CpuFeatures::detect();
}

#[test]
fn cpu_features_logical_cores_positive() {
    let cpu = CpuFeatures::detect();
    assert!(cpu.logical_cores > 0);
}

#[test]
fn cpu_features_best_simd_tier() {
    let cpu = CpuFeatures::detect();
    let tier = cpu.best_simd_tier();
    // Must be one of the valid tiers
    assert!(tier >= SimdTier::Scalar);
}

#[test]
fn cpu_features_recommended_threads() {
    let cpu = CpuFeatures::detect();
    assert!(cpu.recommended_threads() > 0);
}

#[test]
fn cpu_features_summary_nonempty() {
    let cpu = CpuFeatures::detect();
    let s = cpu.summary();
    assert!(!s.is_empty());
}

// ---------------------------------------------------------------------------
// SimdTier
// ---------------------------------------------------------------------------

#[test]
fn simd_tier_ordering() {
    assert!(SimdTier::Scalar < SimdTier::Sse42);
    assert!(SimdTier::Sse42 < SimdTier::Neon);
    assert!(SimdTier::Neon < SimdTier::Avx2);
    assert!(SimdTier::Avx2 < SimdTier::Avx512);
}

#[test]
fn simd_tier_vector_width() {
    assert_eq!(SimdTier::Scalar.vector_width_bits(), 64);
    assert_eq!(SimdTier::Sse42.vector_width_bits(), 128);
    assert_eq!(SimdTier::Neon.vector_width_bits(), 128);
    assert_eq!(SimdTier::Avx2.vector_width_bits(), 256);
    assert_eq!(SimdTier::Avx512.vector_width_bits(), 512);
}

#[test]
fn simd_tier_speedup() {
    assert!((SimdTier::Scalar.expected_speedup_over_scalar() - 1.0).abs() < f32::EPSILON);
    assert!(SimdTier::Sse42.expected_speedup_over_scalar() > 1.0);
    assert!(SimdTier::Neon.expected_speedup_over_scalar() > 1.0);
    assert!(SimdTier::Avx2.expected_speedup_over_scalar() > 1.0);
    assert!(SimdTier::Avx512.expected_speedup_over_scalar() > 1.0);
}

// ---------------------------------------------------------------------------
// MemoryBudget
// ---------------------------------------------------------------------------

#[test]
fn memory_budget_estimate() {
    let budget = MemoryBudget::estimate(8192, 1_000_000_000, 1.125);
    assert!(budget.total_system_bytes > 0);
    assert!(budget.model_weight_bytes > 0);
    assert!(budget.kv_cache_budget > 0);
    assert!(budget.runtime_overhead > 0);
}

#[test]
fn memory_budget_max_context_positive() {
    // 8 GB, 100M params, 4-bit => plenty of room
    let budget = MemoryBudget::estimate(8192, 100_000_000, 4.0);
    let max_ctx = budget.max_context_length(32, 32, 128);
    assert!(max_ctx > 0, "max_context should be > 0, got {max_ctx}");
}

#[test]
fn memory_budget_fits_small_context() {
    let budget = MemoryBudget::estimate(4096, 100_000_000, 1.125);
    assert!(budget.fits_context(128, 32, 32, 128));
}

#[test]
fn memory_budget_summary_nonempty() {
    let budget = MemoryBudget::estimate(4096, 100_000_000, 1.125);
    assert!(!budget.summary().is_empty());
}

// ---------------------------------------------------------------------------
// AutoTuner
// ---------------------------------------------------------------------------

#[test]
fn auto_tuner_new() {
    let _tuner = AutoTuner::new();
}

#[test]
fn auto_tuner_recommend() {
    let tuner = AutoTuner::with_memory_mb(8192);
    let rec = tuner.recommend(100_000_000, 1.125, 32, 32, 128);
    assert!(rec.thread_count > 0);
    assert!(rec.batch_size > 0);
    assert!(rec.max_context > 0);
    assert!(rec.estimated_tokens_per_second > 0.0);
}

#[test]
fn auto_tuner_recommend_uses_flash_decode() {
    // With enough memory and a large model, max_context >= 2048 => flash_decode
    let tuner = AutoTuner::with_memory_mb(16384);
    let rec = tuner.recommend(100_000_000, 1.125, 32, 32, 128);
    // max_context should be large enough for flash decode
    if rec.max_context >= 2048 {
        assert!(rec.use_flash_decode);
    }
}

#[test]
fn tuning_recommendation_summary() {
    let tuner = AutoTuner::with_memory_mb(4096);
    let rec = tuner.recommend(100_000_000, 1.125, 32, 32, 128);
    let s = rec.summary();
    assert!(!s.is_empty());
}

// ---------------------------------------------------------------------------
// KvCacheType
// ---------------------------------------------------------------------------

#[test]
fn kv_cache_type_bytes() {
    assert_eq!(KvCacheType::Fp32.bytes_per_element(), 4);
    assert_eq!(KvCacheType::Fp16.bytes_per_element(), 2);
    assert_eq!(KvCacheType::Int8.bytes_per_element(), 1);
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

#[test]
fn auto_tuner_report_nonempty() {
    let tuner = AutoTuner::new();
    let report = tuner.report();
    assert!(!report.is_empty());
    assert!(report.contains("AutoTuner"));
}

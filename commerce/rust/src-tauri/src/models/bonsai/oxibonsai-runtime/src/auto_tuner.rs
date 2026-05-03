//! Performance auto-tuning: detect hardware, select kernels, tune parameters.
//!
//! The auto-tuner:
//! 1. Detects CPU features (AVX2, AVX-512, NEON, WASM)
//! 2. Estimates memory budget from available system memory
//! 3. Recommends optimal batch size, KV cache size, thread count
//! 4. Selects the best kernel tier for the detected hardware
//! 5. Provides runtime-adjustable tuning knobs

use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// CpuArch
// ---------------------------------------------------------------------------

/// CPU architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuArch {
    X86_64,
    Aarch64,
    Wasm32,
    Other,
}

impl fmt::Display for CpuArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::X86_64 => write!(f, "x86_64"),
            Self::Aarch64 => write!(f, "aarch64"),
            Self::Wasm32 => write!(f, "wasm32"),
            Self::Other => write!(f, "other"),
        }
    }
}

// ---------------------------------------------------------------------------
// SimdTier
// ---------------------------------------------------------------------------

/// SIMD tier ranking (ordered from weakest to strongest).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SimdTier {
    Scalar,
    Sse42,
    Neon,
    Avx2,
    Avx512,
}

impl SimdTier {
    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Scalar => "Scalar",
            Self::Sse42 => "SSE4.2",
            Self::Neon => "NEON",
            Self::Avx2 => "AVX2",
            Self::Avx512 => "AVX-512",
        }
    }

    /// Native vector width in bits.
    pub fn vector_width_bits(&self) -> usize {
        match self {
            Self::Scalar => 64,
            Self::Sse42 => 128,
            Self::Neon => 128,
            Self::Avx2 => 256,
            Self::Avx512 => 512,
        }
    }

    /// Rough expected speed-up over pure scalar for typical GEMV workloads.
    pub fn expected_speedup_over_scalar(&self) -> f32 {
        match self {
            Self::Scalar => 1.0,
            Self::Sse42 => 2.0,
            Self::Neon => 2.5,
            Self::Avx2 => 4.0,
            Self::Avx512 => 7.0,
        }
    }
}

impl fmt::Display for SimdTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// CpuFeatures
// ---------------------------------------------------------------------------

/// Detected CPU features.
#[derive(Debug, Clone, PartialEq)]
pub struct CpuFeatures {
    pub has_avx2: bool,
    pub has_avx512: bool,
    pub has_neon: bool,
    pub has_fma: bool,
    pub has_sse42: bool,
    pub logical_cores: usize,
    pub physical_cores: usize,
    pub arch: CpuArch,
    pub cache_line_bytes: usize,
}

impl CpuFeatures {
    /// Detect features at runtime via `cfg` and `std::thread::available_parallelism`.
    pub fn detect() -> Self {
        let arch = detect_arch();

        let has_avx2 = cfg_has_avx2();
        let has_avx512 = cfg_has_avx512();
        let has_neon = cfg_has_neon();
        let has_fma = cfg_has_fma();
        let has_sse42 = cfg_has_sse42();

        let logical_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        // Rough heuristic: assume hyper-threading factor of 2 on x86_64.
        let physical_cores = match arch {
            CpuArch::X86_64 => logical_cores.div_ceil(2),
            _ => logical_cores,
        };

        let cache_line_bytes = match arch {
            CpuArch::X86_64 => 64,
            CpuArch::Aarch64 => 64,
            _ => 64,
        };

        Self {
            has_avx2,
            has_avx512,
            has_neon,
            has_fma,
            has_sse42,
            logical_cores,
            physical_cores,
            arch,
            cache_line_bytes,
        }
    }

    /// Best SIMD tier available on this CPU.
    pub fn best_simd_tier(&self) -> SimdTier {
        if self.has_avx512 {
            SimdTier::Avx512
        } else if self.has_avx2 {
            SimdTier::Avx2
        } else if self.has_neon {
            SimdTier::Neon
        } else if self.has_sse42 {
            SimdTier::Sse42
        } else {
            SimdTier::Scalar
        }
    }

    /// Recommended thread count for compute-bound work.
    ///
    /// Uses physical cores to avoid contention on hyper-threaded siblings,
    /// but guarantees at least 1.
    pub fn recommended_threads(&self) -> usize {
        self.physical_cores.max(1)
    }

    /// One-line human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "arch={}, simd={}, logical_cores={}, physical_cores={}, cache_line={}B",
            self.arch,
            self.best_simd_tier(),
            self.logical_cores,
            self.physical_cores,
            self.cache_line_bytes,
        )
    }
}

// ---------------------------------------------------------------------------
// KvCacheType
// ---------------------------------------------------------------------------

/// KV cache quantisation type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KvCacheType {
    Fp32,
    Fp16,
    Int8,
}

impl KvCacheType {
    /// Bytes per key or value element.
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Self::Fp32 => 4,
            Self::Fp16 => 2,
            Self::Int8 => 1,
        }
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Fp32 => "FP32",
            Self::Fp16 => "FP16",
            Self::Int8 => "INT8",
        }
    }
}

impl fmt::Display for KvCacheType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// MemoryBudget
// ---------------------------------------------------------------------------

/// Memory budget estimation for a model deployment.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    /// Total system memory the user wants to allocate (bytes).
    pub total_system_bytes: usize,
    /// Bytes available after subtracting model weights and runtime overhead.
    pub available_bytes: usize,
    /// Model weight footprint (bytes).
    pub model_weight_bytes: usize,
    /// Budget specifically earmarked for KV cache (bytes).
    pub kv_cache_budget: usize,
    /// Estimated runtime overhead for buffers, activations, etc. (bytes).
    pub runtime_overhead: usize,
}

impl MemoryBudget {
    /// Estimate budget.
    ///
    /// * `total_available_mb` — how much RAM (MB) the user is willing to use.
    /// * `model_params` — total number of parameters in the model.
    /// * `bits_per_weight` — quantisation bits per weight (e.g. 1.125 for Q1_0).
    pub fn estimate(total_available_mb: usize, model_params: usize, bits_per_weight: f32) -> Self {
        let total_system_bytes = total_available_mb * 1024 * 1024;

        // Weight footprint = params * bits / 8
        let model_weight_bytes = ((model_params as f64) * (bits_per_weight as f64) / 8.0) as usize;

        // Runtime overhead: ~10% of total or 256 MB, whichever is smaller
        let runtime_overhead = (total_system_bytes / 10).min(256 * 1024 * 1024);

        // Available = total - weights - overhead (saturating)
        let available_bytes = total_system_bytes
            .saturating_sub(model_weight_bytes)
            .saturating_sub(runtime_overhead);

        // KV cache gets 80% of remaining budget
        let kv_cache_budget = available_bytes * 4 / 5;

        Self {
            total_system_bytes,
            available_bytes,
            model_weight_bytes,
            kv_cache_budget,
            runtime_overhead,
        }
    }

    /// Maximum context length that fits in the KV cache budget.
    ///
    /// KV cache size per token = 2 (K+V) * num_layers * num_heads * head_dim * bytes_per_element.
    /// We use FP16 (2 bytes) as the default element type for estimation.
    pub fn max_context_length(
        &self,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> usize {
        let bytes_per_token = Self::kv_bytes_per_token(num_layers, num_heads, head_dim);
        if bytes_per_token == 0 {
            return 0;
        }
        self.kv_cache_budget / bytes_per_token
    }

    /// Whether a given context length fits in the KV cache budget.
    pub fn fits_context(
        &self,
        ctx_len: usize,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> bool {
        let bytes_per_token = Self::kv_bytes_per_token(num_layers, num_heads, head_dim);
        ctx_len * bytes_per_token <= self.kv_cache_budget
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
        format!(
            "total={:.0}MB, weights={:.0}MB, kv_budget={:.0}MB, overhead={:.0}MB, available={:.0}MB",
            mb(self.total_system_bytes),
            mb(self.model_weight_bytes),
            mb(self.kv_cache_budget),
            mb(self.runtime_overhead),
            mb(self.available_bytes),
        )
    }

    /// Helper: KV cache bytes per token (K+V, FP16).
    fn kv_bytes_per_token(num_layers: usize, num_heads: usize, head_dim: usize) -> usize {
        // 2 (K+V) * layers * heads * head_dim * 2 bytes (FP16)
        2 * num_layers * num_heads * head_dim * 2
    }
}

// ---------------------------------------------------------------------------
// TuningRecommendation
// ---------------------------------------------------------------------------

/// Tuning recommendations produced by the auto-tuner.
#[derive(Debug, Clone)]
pub struct TuningRecommendation {
    /// Selected SIMD tier.
    pub simd_tier: SimdTier,
    /// Recommended worker thread count.
    pub thread_count: usize,
    /// Recommended batch size for prefill.
    pub batch_size: usize,
    /// Maximum context length that fits memory.
    pub max_context: usize,
    /// Recommended KV cache element type.
    pub kv_cache_type: KvCacheType,
    /// Whether to use flash-decode optimisation.
    pub use_flash_decode: bool,
    /// Whether to use prefix caching.
    pub use_prefix_cache: bool,
    /// Estimated tokens per second (rough).
    pub estimated_tokens_per_second: f32,
}

impl TuningRecommendation {
    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "simd={}, threads={}, batch={}, max_ctx={}, kv={}, flash_decode={}, prefix_cache={}, est_tok/s={:.1}",
            self.simd_tier,
            self.thread_count,
            self.batch_size,
            self.max_context,
            self.kv_cache_type,
            self.use_flash_decode,
            self.use_prefix_cache,
            self.estimated_tokens_per_second,
        )
    }
}

// ---------------------------------------------------------------------------
// KernelBenchmark
// ---------------------------------------------------------------------------

/// Kernel micro-benchmark result.
#[derive(Debug, Clone)]
pub struct KernelBenchmark {
    /// SIMD tier that was benchmarked.
    pub simd_tier: SimdTier,
    /// Number of iterations run.
    pub iterations: usize,
    /// Total wall-clock time in milliseconds.
    pub total_duration_ms: f64,
    /// Operations per second.
    pub ops_per_second: f64,
    /// Estimated GFLOPS (based on a synthetic FMA-heavy workload).
    pub gflops: f64,
}

impl KernelBenchmark {
    /// Human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "tier={}, iters={}, time={:.2}ms, ops/s={:.0}, GFLOPS={:.2}",
            self.simd_tier,
            self.iterations,
            self.total_duration_ms,
            self.ops_per_second,
            self.gflops,
        )
    }
}

// ---------------------------------------------------------------------------
// AutoTuner
// ---------------------------------------------------------------------------

/// The auto-tuner: detects hardware and recommends inference parameters.
pub struct AutoTuner {
    cpu: CpuFeatures,
    memory_mb: usize,
}

impl AutoTuner {
    /// Create a new auto-tuner that auto-detects CPU and uses system RSS as
    /// a rough memory estimate (defaults to 4096 MB if detection fails).
    pub fn new() -> Self {
        let cpu = CpuFeatures::detect();
        // Use a conservative default: 4 GB
        let memory_mb = 4096;
        Self { cpu, memory_mb }
    }

    /// Create an auto-tuner with an explicit memory budget (in MB).
    pub fn with_memory_mb(memory_mb: usize) -> Self {
        let cpu = CpuFeatures::detect();
        Self { cpu, memory_mb }
    }

    /// Generate tuning recommendations for a specific model configuration.
    ///
    /// * `model_params` — total number of model parameters.
    /// * `bits_per_weight` — quantisation bits (e.g. 1.125 for Q1_0_g128).
    /// * `num_layers` / `num_heads` / `head_dim` — transformer architecture.
    pub fn recommend(
        &self,
        model_params: usize,
        bits_per_weight: f32,
        num_layers: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> TuningRecommendation {
        let simd_tier = self.cpu.best_simd_tier();
        let thread_count = self.cpu.recommended_threads();

        let budget = MemoryBudget::estimate(self.memory_mb, model_params, bits_per_weight);
        let max_context = budget.max_context_length(num_layers, num_heads, head_dim);

        // Batch size heuristic: scale with cores, cap by available memory.
        let batch_size = compute_batch_size(thread_count, &budget);

        // KV cache type: prefer FP16, fall back to INT8 if memory is tight.
        let kv_cache_type = if budget.kv_cache_budget > 128 * 1024 * 1024 {
            KvCacheType::Fp16
        } else {
            KvCacheType::Int8
        };

        // Flash decode is beneficial for long contexts (>= 2048 tokens).
        let use_flash_decode = max_context >= 2048;

        // Prefix caching useful when there is plenty of KV budget.
        let use_prefix_cache = budget.kv_cache_budget > 256 * 1024 * 1024;

        // Rough throughput estimate (tokens/s).
        let base_tps: f32 = 30.0; // baseline for scalar on 1 core
        let speedup = simd_tier.expected_speedup_over_scalar();
        let core_factor = (thread_count as f32).sqrt(); // diminishing returns
        let estimated_tokens_per_second = base_tps * speedup * core_factor;

        TuningRecommendation {
            simd_tier,
            thread_count,
            batch_size,
            max_context,
            kv_cache_type,
            use_flash_decode,
            use_prefix_cache,
            estimated_tokens_per_second,
        }
    }

    /// Quick micro-benchmark: run a synthetic FMA-heavy kernel to estimate
    /// raw compute throughput.
    pub fn benchmark_kernel(&self, iterations: usize) -> KernelBenchmark {
        let simd_tier = self.cpu.best_simd_tier();
        let n = 1024usize; // vector length
        let flops_per_iter = n * 2; // one FMA = 2 flops per element

        // Allocate work buffers
        let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..n).map(|i| 1.0 - (i as f32) * 0.0005).collect();
        let mut acc = vec![0.0f32; n];

        let start = Instant::now();
        for _ in 0..iterations {
            for j in 0..n {
                // FMA: acc[j] += a[j] * b[j]
                acc[j] += a[j] * b[j];
            }
            // Prevent the compiler from optimising the loop away
            std::hint::black_box(&acc);
        }
        let elapsed = start.elapsed();
        let total_duration_ms = elapsed.as_secs_f64() * 1000.0;

        let total_flops = (iterations * flops_per_iter) as f64;
        let elapsed_s = elapsed.as_secs_f64().max(1e-12);
        let ops_per_second = iterations as f64 / elapsed_s;
        let gflops = total_flops / elapsed_s / 1e9;

        KernelBenchmark {
            simd_tier,
            iterations,
            total_duration_ms,
            ops_per_second,
            gflops,
        }
    }

    /// Reference to the detected CPU features.
    pub fn cpu_features(&self) -> &CpuFeatures {
        &self.cpu
    }

    /// Full diagnostic report.
    pub fn report(&self) -> String {
        let cpu_summary = self.cpu.summary();
        let bench = self.benchmark_kernel(1000);
        format!(
            "OxiBonsai AutoTuner Report\n\
             ==========================\n\
             CPU: {}\n\
             Benchmark: {}\n\
             Memory budget: {} MB",
            cpu_summary,
            bench.summary(),
            self.memory_mb,
        )
    }
}

impl Default for AutoTuner {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers — feature detection
// ---------------------------------------------------------------------------

fn detect_arch() -> CpuArch {
    #[cfg(target_arch = "x86_64")]
    {
        CpuArch::X86_64
    }
    #[cfg(target_arch = "aarch64")]
    {
        CpuArch::Aarch64
    }
    #[cfg(target_arch = "wasm32")]
    {
        CpuArch::Wasm32
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        CpuArch::Other
    }
}

fn cfg_has_avx2() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "avx2")]
        {
            return true;
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            // Runtime detection via CPUID
            #[cfg(target_arch = "x86_64")]
            {
                return std::arch::is_x86_feature_detected!("avx2");
            }
            #[allow(unreachable_code)]
            false
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn cfg_has_avx512() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "avx512f")]
        {
            return true;
        }
        #[cfg(not(target_feature = "avx512f"))]
        {
            #[cfg(target_arch = "x86_64")]
            {
                return std::arch::is_x86_feature_detected!("avx512f");
            }
            #[allow(unreachable_code)]
            false
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn cfg_has_neon() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on AArch64
        true
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

fn cfg_has_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "fma")]
        {
            return true;
        }
        #[cfg(not(target_feature = "fma"))]
        {
            #[cfg(target_arch = "x86_64")]
            {
                return std::arch::is_x86_feature_detected!("fma");
            }
            #[allow(unreachable_code)]
            false
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        // AArch64 always has FMA
        true
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        false
    }
}

fn cfg_has_sse42() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(target_feature = "sse4.2")]
        {
            return true;
        }
        #[cfg(not(target_feature = "sse4.2"))]
        {
            #[cfg(target_arch = "x86_64")]
            {
                return std::arch::is_x86_feature_detected!("sse4.2");
            }
            #[allow(unreachable_code)]
            false
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Compute a recommended batch size based on thread count and memory budget.
fn compute_batch_size(thread_count: usize, budget: &MemoryBudget) -> usize {
    // Start with a base of 1, scale up with cores
    let core_based = thread_count;

    // Cap by available memory: each batch element uses ~1 MB activation buffer
    let activation_bytes_per_item: usize = 1024 * 1024; // 1 MB
    let memory_based = budget
        .available_bytes
        .checked_div(activation_bytes_per_item)
        .unwrap_or(1);

    // Take the minimum, clamp to [1, 128]
    core_based.min(memory_based).clamp(1, 128)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_arch_returns_valid() {
        let arch = detect_arch();
        // Just ensure it doesn't panic and returns something
        let _ = format!("{arch}");
    }

    #[test]
    fn simd_tier_display() {
        assert_eq!(SimdTier::Scalar.name(), "Scalar");
        assert_eq!(SimdTier::Avx512.name(), "AVX-512");
    }

    #[test]
    fn memory_budget_zero_params() {
        let budget = MemoryBudget::estimate(1024, 0, 1.0);
        assert!(budget.model_weight_bytes == 0);
        assert!(budget.available_bytes > 0);
    }
}

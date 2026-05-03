//! High-level convenience functions for common OxiBonsai operations.

use crate::error::{RuntimeError, RuntimeResult};

/// Generation result with metadata.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    /// The generated text.
    pub text: String,
    /// Token IDs of the generated tokens.
    pub token_ids: Vec<u32>,
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,
    /// Number of generated tokens.
    pub generated_tokens: usize,
    /// Generation speed in tokens per second.
    pub tokens_per_second: f64,
    /// Reason generation stopped (e.g. "stop", "length", "error").
    pub finish_reason: String,
}

/// Simple token generation statistics.
#[derive(Debug, Clone, Default)]
pub struct TokenStats {
    /// Total tokens (prompt + completion).
    pub total_tokens: usize,
    /// Number of tokens in the prompt.
    pub prompt_tokens: usize,
    /// Number of generated tokens.
    pub completion_tokens: usize,
    /// Time to first token in milliseconds.
    pub time_to_first_token_ms: f64,
    /// Average generation speed in tokens per second.
    pub tokens_per_second: f64,
}

/// Information about a model file.
#[derive(Debug, Clone)]
pub struct ModelFileInfo {
    /// File path.
    pub path: String,
    /// File size in bytes.
    pub size_bytes: u64,
    /// Detected format description.
    pub format: String,
    /// Whether the file appears to be a valid GGUF file.
    pub is_valid_gguf: bool,
}

/// Validate that a model file exists and has the correct format.
///
/// Checks for file existence, reads the magic number, and verifies
/// it matches the GGUF format (magic = 0x46554747).
pub fn validate_model_file(path: &str) -> RuntimeResult<ModelFileInfo> {
    let metadata = std::fs::metadata(path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            RuntimeError::FileNotFound {
                path: path.to_string(),
            }
        } else {
            RuntimeError::Io(e)
        }
    })?;

    if !metadata.is_file() {
        return Err(RuntimeError::Config(format!(
            "path '{}' is not a regular file",
            path
        )));
    }

    let size_bytes = metadata.len();

    // Check GGUF magic number (first 4 bytes)
    let mut is_valid_gguf = false;
    let mut format = "unknown".to_string();

    if size_bytes >= 4 {
        let file = std::fs::File::open(path).map_err(RuntimeError::Io)?;
        let mut reader = std::io::BufReader::new(file);
        let mut magic_bytes = [0u8; 4];
        use std::io::Read;
        if reader.read_exact(&mut magic_bytes).is_ok() {
            let magic = u32::from_le_bytes(magic_bytes);
            if magic == 0x46554747 {
                is_valid_gguf = true;
                format = "GGUF".to_string();
            } else {
                format = format!("unknown (magic: 0x{:08X})", magic);
            }
        }
    }

    Ok(ModelFileInfo {
        path: path.to_string(),
        size_bytes,
        format,
        is_valid_gguf,
    })
}

/// Memory usage estimate for model inference.
///
/// # Example
///
/// ```
/// use oxibonsai_runtime::convenience::estimate_memory_requirements;
///
/// let est = estimate_memory_requirements(
///     1_000_000_000, // 1 GB model
///     4096,          // max sequence length
///     8,             // KV heads
///     128,           // head dim
///     36,            // layers
/// );
/// assert!(est.total_bytes > est.model_weights_bytes);
/// assert!(est.fits_in_memory);
/// ```
#[derive(Debug, Clone)]
pub struct MemoryEstimate {
    /// Memory required for model weights.
    pub model_weights_bytes: u64,
    /// Memory required for KV cache.
    pub kv_cache_bytes: u64,
    /// Estimated runtime overhead (buffers, activations, etc.).
    pub runtime_overhead_bytes: u64,
    /// Total estimated memory requirement.
    pub total_bytes: u64,
    /// Whether the model fits in available memory (heuristic check).
    pub fits_in_memory: bool,
}

/// Estimate memory requirements for inference.
///
/// This provides a rough estimate based on model dimensions.
/// For 1-bit models, weight memory is significantly reduced compared
/// to FP16/FP32 models.
///
/// # Parameters
/// - `model_size_bytes`: Size of the model file on disk.
/// - `max_seq_len`: Maximum sequence length for KV cache.
/// - `num_kv_heads`: Number of KV attention heads.
/// - `head_dim`: Dimension of each attention head.
/// - `num_layers`: Number of transformer layers.
pub fn estimate_memory_requirements(
    model_size_bytes: u64,
    max_seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_layers: usize,
) -> MemoryEstimate {
    let model_weights_bytes = model_size_bytes;

    // KV cache: 2 (K+V) * num_layers * num_kv_heads * head_dim * max_seq_len * 4 bytes (f32)
    let kv_cache_bytes =
        2u64 * num_layers as u64 * num_kv_heads as u64 * head_dim as u64 * max_seq_len as u64 * 4;

    // Runtime overhead: ~10% of model weights + some fixed overhead for activations
    let runtime_overhead_bytes = model_weights_bytes / 10 + 256 * 1024 * 1024; // +256MB base

    let total_bytes = model_weights_bytes + kv_cache_bytes + runtime_overhead_bytes;

    // Heuristic: check against a reasonable memory budget (e.g. 90% of typical systems)
    // For now, we just check if total is under 64GB which covers most systems
    let fits_in_memory = total_bytes < 64 * 1024 * 1024 * 1024;

    MemoryEstimate {
        model_weights_bytes,
        kv_cache_bytes,
        runtime_overhead_bytes,
        total_bytes,
        fits_in_memory,
    }
}

/// Format a token count for human-readable display.
///
/// # Example
///
/// ```
/// use oxibonsai_runtime::convenience::format_token_count;
///
/// assert_eq!(format_token_count(42), "42 tokens");
/// assert_eq!(format_token_count(1_500), "1.5K tokens");
/// assert_eq!(format_token_count(3_500_000), "3.5M tokens");
/// ```
pub fn format_token_count(count: usize) -> String {
    if count < 1_000 {
        format!("{} tokens", count)
    } else if count < 1_000_000 {
        format!("{:.1}K tokens", count as f64 / 1_000.0)
    } else if count < 1_000_000_000 {
        format!("{:.1}M tokens", count as f64 / 1_000_000.0)
    } else {
        format!("{:.1}B tokens", count as f64 / 1_000_000_000.0)
    }
}

/// Format a byte count for human-readable display.
///
/// # Example
///
/// ```
/// use oxibonsai_runtime::convenience::format_bytes;
///
/// assert_eq!(format_bytes(512), "512 B");
/// assert_eq!(format_bytes(1024), "1.00 KB");
/// assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
/// assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
/// ```
pub fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = 1024 * KB;
    const GB: u64 = 1024 * MB;
    const TB: u64 = 1024 * GB;

    if bytes < KB {
        format!("{} B", bytes)
    } else if bytes < MB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else if bytes < GB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes < TB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    }
}

/// Format a duration for human-readable display.
///
/// Examples: "123ms", "1.23s", "5m 30s", "1h 15m"
pub fn format_duration(duration: std::time::Duration) -> String {
    let total_ms = duration.as_millis();

    if total_ms < 1_000 {
        format!("{}ms", total_ms)
    } else if total_ms < 60_000 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if total_ms < 3_600_000 {
        let minutes = duration.as_secs() / 60;
        let seconds = duration.as_secs() % 60;
        format!("{}m {}s", minutes, seconds)
    } else {
        let hours = duration.as_secs() / 3600;
        let minutes = (duration.as_secs() % 3600) / 60;
        format!("{}h {}m", hours, minutes)
    }
}

/// Format tokens per second for display.
///
/// Examples: "23.4 t/s", "0.5 t/s", "150.0 t/s"
pub fn format_tokens_per_second(tps: f64) -> String {
    if tps < 0.0 {
        "0.0 t/s".to_string()
    } else if tps < 10.0 {
        format!("{:.2} t/s", tps)
    } else if tps < 1000.0 {
        format!("{:.1} t/s", tps)
    } else {
        format!("{:.0} t/s", tps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── format_token_count ──

    #[test]
    fn format_token_count_small() {
        assert_eq!(format_token_count(0), "0 tokens");
        assert_eq!(format_token_count(42), "42 tokens");
        assert_eq!(format_token_count(999), "999 tokens");
    }

    #[test]
    fn format_token_count_thousands() {
        assert_eq!(format_token_count(1_000), "1.0K tokens");
        assert_eq!(format_token_count(1_234), "1.2K tokens");
        assert_eq!(format_token_count(999_999), "1000.0K tokens");
    }

    #[test]
    fn format_token_count_millions() {
        assert_eq!(format_token_count(1_000_000), "1.0M tokens");
        assert_eq!(format_token_count(3_500_000), "3.5M tokens");
    }

    #[test]
    fn format_token_count_billions() {
        assert_eq!(format_token_count(1_000_000_000), "1.0B tokens");
    }

    // ── format_bytes ──

    #[test]
    fn format_bytes_small() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1023), "1023 B");
    }

    #[test]
    fn format_bytes_kb() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1536), "1.50 KB");
    }

    #[test]
    fn format_bytes_mb() {
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(512 * 1024 * 1024), "512.00 MB");
    }

    #[test]
    fn format_bytes_gb() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(
            format_bytes(2 * 1024 * 1024 * 1024 + 300 * 1024 * 1024),
            "2.29 GB"
        );
    }

    #[test]
    fn format_bytes_tb() {
        assert_eq!(format_bytes(1024u64 * 1024 * 1024 * 1024), "1.00 TB");
    }

    // ── format_duration ──

    #[test]
    fn format_duration_ms() {
        assert_eq!(format_duration(std::time::Duration::from_millis(0)), "0ms");
        assert_eq!(
            format_duration(std::time::Duration::from_millis(123)),
            "123ms"
        );
        assert_eq!(
            format_duration(std::time::Duration::from_millis(999)),
            "999ms"
        );
    }

    #[test]
    fn format_duration_seconds() {
        assert_eq!(
            format_duration(std::time::Duration::from_millis(1_000)),
            "1.00s"
        );
        assert_eq!(
            format_duration(std::time::Duration::from_millis(1_230)),
            "1.23s"
        );
    }

    #[test]
    fn format_duration_minutes() {
        assert_eq!(
            format_duration(std::time::Duration::from_secs(90)),
            "1m 30s"
        );
        assert_eq!(
            format_duration(std::time::Duration::from_secs(330)),
            "5m 30s"
        );
    }

    #[test]
    fn format_duration_hours() {
        assert_eq!(
            format_duration(std::time::Duration::from_secs(4500)),
            "1h 15m"
        );
    }

    // ── format_tokens_per_second ──

    #[test]
    fn format_tps() {
        assert_eq!(format_tokens_per_second(-1.0), "0.0 t/s");
        assert_eq!(format_tokens_per_second(0.0), "0.00 t/s");
        assert_eq!(format_tokens_per_second(0.5), "0.50 t/s");
        assert_eq!(format_tokens_per_second(23.4), "23.4 t/s");
        assert_eq!(format_tokens_per_second(150.0), "150.0 t/s");
        assert_eq!(format_tokens_per_second(1500.0), "1500 t/s");
    }

    // ── memory estimation ──

    #[test]
    fn estimate_memory_basic() {
        let est = estimate_memory_requirements(
            1_000_000_000, // ~1GB model
            4096,          // max_seq_len
            8,             // num_kv_heads
            128,           // head_dim
            36,            // num_layers
        );

        assert_eq!(est.model_weights_bytes, 1_000_000_000);
        // KV cache: 2 * 36 * 8 * 128 * 4096 * 4 = 1,207,959,552
        assert_eq!(est.kv_cache_bytes, 2 * 36 * 8 * 128 * 4096 * 4);
        assert!(est.total_bytes > est.model_weights_bytes + est.kv_cache_bytes);
        assert!(est.fits_in_memory);
    }

    #[test]
    fn estimate_memory_large_model() {
        let est = estimate_memory_requirements(
            100_000_000_000, // 100GB
            32768,
            64,
            128,
            80,
        );
        // This should not fit in 64GB
        assert!(!est.fits_in_memory);
    }

    // ── validate_model_file ──

    #[test]
    fn validate_model_file_nonexistent() {
        let path = std::env::temp_dir().join("nonexistent_oxibonsai_model_12345.gguf");
        let result = validate_model_file(path.to_str().expect("path is valid UTF-8"));
        assert!(result.is_err());
    }

    #[test]
    fn validate_model_file_not_gguf() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxibonsai_test_not_gguf.bin");
        std::fs::write(&path, b"this is not a gguf file").expect("write temp file");

        let result = validate_model_file(&path.to_string_lossy());
        assert!(result.is_ok());
        let info = result.expect("should return info");
        assert!(!info.is_valid_gguf);
        assert!(info.format.contains("unknown"));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn validate_model_file_valid_gguf_magic() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxibonsai_test_gguf_magic.bin");
        // GGUF magic = 0x46554747 = little-endian bytes [0x47, 0x47, 0x55, 0x46]
        let mut data = vec![0x47u8, 0x47, 0x55, 0x46];
        data.extend_from_slice(&[0u8; 100]); // pad with zeros
        std::fs::write(&path, &data).expect("write temp file");

        let result = validate_model_file(&path.to_string_lossy());
        assert!(result.is_ok());
        let info = result.expect("should return info");
        assert!(info.is_valid_gguf);
        assert_eq!(info.format, "GGUF");
        assert!(info.size_bytes > 0);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn validate_model_file_empty() {
        let dir = std::env::temp_dir();
        let path = dir.join("oxibonsai_test_empty.bin");
        std::fs::write(&path, b"").expect("write temp file");

        let result = validate_model_file(&path.to_string_lossy());
        assert!(result.is_ok());
        let info = result.expect("should return info");
        assert!(!info.is_valid_gguf);

        let _ = std::fs::remove_file(&path);
    }

    // ── GenerationResult / TokenStats ──

    #[test]
    fn generation_result_clone() {
        let result = GenerationResult {
            text: "hello".to_string(),
            token_ids: vec![1, 2, 3],
            prompt_tokens: 5,
            generated_tokens: 3,
            tokens_per_second: 10.0,
            finish_reason: "stop".to_string(),
        };
        let cloned = result.clone();
        assert_eq!(cloned.text, "hello");
        assert_eq!(cloned.generated_tokens, 3);
    }

    #[test]
    fn token_stats_default() {
        let stats = TokenStats::default();
        assert_eq!(stats.total_tokens, 0);
        assert_eq!(stats.prompt_tokens, 0);
        assert_eq!(stats.completion_tokens, 0);
        assert!((stats.time_to_first_token_ms - 0.0).abs() < f64::EPSILON);
        assert!((stats.tokens_per_second - 0.0).abs() < f64::EPSILON);
    }
}

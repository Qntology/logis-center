//! Production-quality GGUF model loader with validation, streaming, and memory budgeting.
//!
//! This module provides high-level utilities for loading GGUF model files with:
//! - Configurable memory budgets and validation strictness
//! - Lazy tensor metadata loading (no weight data read upfront)
//! - Streaming chunk iterators for progressive loading
//! - Memory footprint estimation before committing to a full load

use std::io::Read;
use std::path::Path;
use std::time::Instant;

// ─────────────────────────────────────────────────────────────────────────────
// LoadError
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during GGUF model loading.
#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    /// An underlying I/O error (e.g., file not found, permission denied).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// The GGUF file could not be parsed (malformed binary).
    #[error("GGUF parse error: {0}")]
    Parse(String),

    /// Loading this file would exceed the configured memory budget.
    #[error("memory budget exceeded: need {need} bytes, budget {budget} bytes")]
    MemoryBudgetExceeded { need: u64, budget: u64 },

    /// The GGUF version in the file header is not supported.
    #[error("unsupported GGUF version: {0}")]
    UnsupportedVersion(u32),

    /// A required structural invariant was violated.
    #[error("validation failed: {0}")]
    ValidationFailed(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration governing how a GGUF model is loaded.
#[derive(Debug, Clone)]
pub struct LoadConfig {
    /// Maximum memory (in bytes) the loader is allowed to consume; `None` = unlimited.
    pub max_memory_bytes: Option<usize>,
    /// Whether to validate file-level checksums (currently advisory).
    pub validate_checksums: bool,
    /// If `true`, tensors with unrecognised quantisation types are silently skipped
    /// rather than returning an error.
    pub allow_unknown_quant_types: bool,
    /// Size of each streaming chunk in bytes when using [`TensorChunkIter`].
    pub streaming_chunk_size: usize,
    /// If `true`, reject GGUF files that declare an unsupported version.
    pub strict_version: bool,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: None,
            validate_checksums: false,
            allow_unknown_quant_types: true,
            streaming_chunk_size: 4 * 1024 * 1024, // 4 MiB
            strict_version: false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LoadStats
// ─────────────────────────────────────────────────────────────────────────────

/// Statistics gathered during a model loading operation.
#[derive(Debug, Clone, Default)]
pub struct LoadStats {
    /// Number of tensors successfully loaded.
    pub tensors_loaded: usize,
    /// Total bytes of tensor weight data loaded.
    pub bytes_loaded: u64,
    /// Tensors skipped because their quantisation type was unrecognised.
    pub skipped_tensors: usize,
    /// Wall-clock time for the entire load operation in milliseconds.
    pub load_time_ms: u64,
    /// Approximate peak memory usage (bytes) during loading.
    pub peak_memory_bytes: usize,
    /// Non-fatal issues found during validation (empty = clean).
    pub validation_warnings: Vec<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// TensorEntry
// ─────────────────────────────────────────────────────────────────────────────

/// Known quantisation type IDs and their human-readable names.
const KNOWN_QUANT_TYPES: &[(u32, &str)] = &[
    (0, "F32"),
    (1, "F16"),
    (2, "Q4_0"),
    (3, "Q4_1"),
    (6, "Q5_0"),
    (7, "Q5_1"),
    (8, "Q8_0"),
    (9, "Q8_1"),
    (10, "Q2_K"),
    (11, "Q3_K"),
    (12, "Q4_K"),
    (13, "Q5_K"),
    (14, "Q6_K"),
    (15, "Q8_K"),
    (30, "BF16"),
    (35, "TQ2_0"),
    (41, "Q1_0_g128"),
    (42, "TQ2_0_g128"),
];

/// A loaded tensor entry — contains metadata only; no weight bytes are held here.
/// Use [`load_tensor_metadata`] to obtain a collection of these, then open the
/// file and seek to `offset` to read the actual data.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    /// Tensor name as stored in the GGUF file (e.g. `"blk.0.attn_q.weight"`).
    pub name: String,
    /// Shape dimensions (e.g. `[4096, 4096]`).
    pub shape: Vec<u64>,
    /// Raw GGUF quantisation type ID.
    pub quant_type_id: u32,
    /// Byte offset of this tensor's data from the start of the tensor data section.
    pub offset: u64,
    /// Number of bytes occupied by this tensor in the data section.
    pub size_bytes: u64,
}

impl TensorEntry {
    /// Total number of elements across all dimensions.
    pub fn element_count(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Human-readable name for the quantisation type, or `"UNKNOWN"`.
    pub fn quant_name(&self) -> &'static str {
        KNOWN_QUANT_TYPES
            .iter()
            .find(|(id, _)| *id == self.quant_type_id)
            .map(|(_, name)| *name)
            .unwrap_or("UNKNOWN")
    }

    /// Returns `true` when the quantisation type ID is one OxiBonsai recognises.
    pub fn is_known_quant(&self) -> bool {
        KNOWN_QUANT_TYPES
            .iter()
            .any(|(id, _)| *id == self.quant_type_id)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GGUF raw-parsing helpers (pure Rust, no external deps beyond std)
// ─────────────────────────────────────────────────────────────────────────────

/// GGUF magic bytes in little-endian order: ASCII "GGUF" = bytes [0x47,0x47,0x55,0x46] → LE u32 0x46554747.
const GGUF_MAGIC: u32 = 0x4655_4747;

/// Supported GGUF versions.
const SUPPORTED_VERSIONS: &[u32] = &[2, 3];

/// Read a little-endian u32 from a cursor.
fn read_u32_le(buf: &[u8], pos: &mut usize) -> Result<u32, LoadError> {
    if *pos + 4 > buf.len() {
        return Err(LoadError::Parse(format!(
            "unexpected EOF at offset {} reading u32",
            pos
        )));
    }
    let v = u32::from_le_bytes(
        buf[*pos..*pos + 4]
            .try_into()
            .map_err(|_| LoadError::Parse("slice conversion failed for u32".to_string()))?,
    );
    *pos += 4;
    Ok(v)
}

/// Read a little-endian u64 from a cursor.
fn read_u64_le(buf: &[u8], pos: &mut usize) -> Result<u64, LoadError> {
    if *pos + 8 > buf.len() {
        return Err(LoadError::Parse(format!(
            "unexpected EOF at offset {} reading u64",
            pos
        )));
    }
    let v = u64::from_le_bytes(
        buf[*pos..*pos + 8]
            .try_into()
            .map_err(|_| LoadError::Parse("slice conversion failed for u64".to_string()))?,
    );
    *pos += 8;
    Ok(v)
}

/// Read a GGUF string: [u64 length][bytes].
fn read_gguf_string(buf: &[u8], pos: &mut usize) -> Result<String, LoadError> {
    let len = read_u64_le(buf, pos)? as usize;
    if *pos + len > buf.len() {
        return Err(LoadError::Parse(format!(
            "string of length {len} extends beyond buffer"
        )));
    }
    let s = std::str::from_utf8(&buf[*pos..*pos + len])
        .map_err(|e| LoadError::Parse(format!("invalid UTF-8 in string: {e}")))?
        .to_string();
    *pos += len;
    Ok(s)
}

/// Skip over a GGUF metadata value (we don't need the values for metadata-only loading).
/// Returns the number of bytes consumed.
fn skip_metadata_value(buf: &[u8], pos: &mut usize, value_type: u32) -> Result<(), LoadError> {
    match value_type {
        0 | 1 => {
            // uint8, int8
            if *pos + 1 > buf.len() {
                return Err(LoadError::Parse("EOF in u8/i8 value".to_string()));
            }
            *pos += 1;
        }
        2 | 3 => {
            // uint16, int16
            if *pos + 2 > buf.len() {
                return Err(LoadError::Parse("EOF in u16/i16 value".to_string()));
            }
            *pos += 2;
        }
        4..=7 => {
            // uint32, int32, float32, bool
            if *pos + 4 > buf.len() {
                return Err(LoadError::Parse(
                    "EOF in u32/i32/f32/bool value".to_string(),
                ));
            }
            *pos += 4;
        }
        8 => {
            // string
            read_gguf_string(buf, pos)?;
        }
        9 => {
            // array: [value_type: u32][count: u64][elements...]
            let elem_type = read_u32_le(buf, pos)?;
            let count = read_u64_le(buf, pos)?;
            for _ in 0..count {
                skip_metadata_value(buf, pos, elem_type)?;
            }
        }
        10..=12 => {
            // uint64, int64, float64
            if *pos + 8 > buf.len() {
                return Err(LoadError::Parse("EOF in u64/i64/f64 value".to_string()));
            }
            *pos += 8;
        }
        other => {
            return Err(LoadError::Parse(format!(
                "unknown metadata value type id: {other}"
            )));
        }
    }
    Ok(())
}

/// Low-level result of parsing a GGUF file's header + metadata + tensor info.
struct ParsedGgufMeta {
    version: u32,
    tensor_entries: Vec<TensorEntry>,
}

/// Parse GGUF header, skip metadata KV, parse tensor info entries.
/// Does NOT read any tensor weight bytes.
fn parse_gguf_meta(buf: &[u8]) -> Result<ParsedGgufMeta, LoadError> {
    let mut pos = 0usize;

    // --- Header ---
    let magic = read_u32_le(buf, &mut pos)?;
    if magic != GGUF_MAGIC {
        return Err(LoadError::Parse(format!(
            "invalid GGUF magic: 0x{:08X} (expected 0x{:08X})",
            magic, GGUF_MAGIC
        )));
    }

    let version = read_u32_le(buf, &mut pos)?;

    let tensor_count = read_u64_le(buf, &mut pos)?;
    let metadata_kv_count = read_u64_le(buf, &mut pos)?;

    // --- Metadata KV pairs (skip values, we only need structure) ---
    for _ in 0..metadata_kv_count {
        // key
        read_gguf_string(buf, &mut pos)?;
        // value type
        let value_type = read_u32_le(buf, &mut pos)?;
        // value
        skip_metadata_value(buf, &mut pos, value_type)?;
    }

    // --- Tensor info entries ---
    let mut tensor_entries = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_gguf_string(buf, &mut pos)?;
        let n_dims = read_u32_le(buf, &mut pos)?;
        let mut shape = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            shape.push(read_u64_le(buf, &mut pos)?);
        }
        let quant_type_id = read_u32_le(buf, &mut pos)?;
        let offset = read_u64_le(buf, &mut pos)?;

        // Compute size_bytes using quantisation block math.
        let size_bytes = compute_tensor_size_bytes(&shape, quant_type_id);

        tensor_entries.push(TensorEntry {
            name,
            shape,
            quant_type_id,
            offset,
            size_bytes,
        });
    }

    Ok(ParsedGgufMeta {
        version,
        tensor_entries,
    })
}

/// Compute the byte size of a tensor given its shape and quant type ID.
fn compute_tensor_size_bytes(shape: &[u64], quant_type_id: u32) -> u64 {
    let element_count: u64 = shape.iter().product();
    let (block_size, block_bytes): (u64, u64) = match quant_type_id {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 40),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        30 => (1, 2),     // BF16
        35 => (256, 66),  // TQ2_0 (llama.cpp ternary, 256-element groups)
        41 => (128, 18),  // Q1_0_g128
        42 => (128, 34),  // TQ2_0_g128 (PrismML ternary, 128-element groups)
        // Unknown type: assume 1 byte per element as a conservative fallback
        _ => (1, 1),
    };
    let num_blocks = element_count.div_ceil(block_size);
    num_blocks * block_bytes
}

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Validates a GGUF file at `path`, checking:
/// - File exists and is readable
/// - Magic bytes are correct
/// - Version is in the supported set
/// - Tensor count and metadata are self-consistent
///
/// Returns a (possibly empty) list of advisory warning strings.
/// An empty list means the file passed all checks.
pub fn validate_gguf_file(path: &Path) -> Result<Vec<String>, LoadError> {
    let mut file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    let mut warnings = Vec::new();
    let start = Instant::now();

    let meta = parse_gguf_meta(&buf)?;

    if !SUPPORTED_VERSIONS.contains(&meta.version) {
        warnings.push(format!(
            "GGUF version {} is not in the officially supported set {:?}",
            meta.version, SUPPORTED_VERSIONS
        ));
    }

    if meta.tensor_entries.is_empty() {
        warnings.push("file contains zero tensors".to_string());
    }

    for entry in &meta.tensor_entries {
        if !entry.is_known_quant() {
            warnings.push(format!(
                "tensor '{}' has unknown quantisation type id {}",
                entry.name, entry.quant_type_id
            ));
        }
        if entry.shape.is_empty() {
            warnings.push(format!(
                "tensor '{}' has zero-dimensional shape",
                entry.name
            ));
        }
    }

    let _elapsed = start.elapsed();
    Ok(warnings)
}

/// Loads tensor metadata (names, shapes, types, offsets) from a GGUF file.
///
/// This is intentionally fast — no weight bytes are read.  Call this to build
/// a directory of available tensors before deciding which to materialise.
pub fn load_tensor_metadata(path: &Path) -> Result<Vec<TensorEntry>, LoadError> {
    let _t0 = Instant::now();

    let mut file = std::fs::File::open(path)?;
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)?;

    let meta = parse_gguf_meta(&buf)?;
    Ok(meta.tensor_entries)
}

/// Computes the expected memory footprint (in bytes) for fully loading all
/// tensor weight data from the given GGUF file.
///
/// This reads only the file header and tensor metadata — no weight bytes.
pub fn estimate_memory_bytes(path: &Path) -> Result<u64, LoadError> {
    let entries = load_tensor_metadata(path)?;
    let total: u64 = entries.iter().map(|e| e.size_bytes).sum();
    Ok(total)
}

/// Returns `true` when the GGUF file at `path` fits within `budget_bytes`.
///
/// Identical to calling [`estimate_memory_bytes`] and comparing.
pub fn fits_in_budget(path: &Path, budget_bytes: u64) -> Result<bool, LoadError> {
    let need = estimate_memory_bytes(path)?;
    Ok(need <= budget_bytes)
}

// ─────────────────────────────────────────────────────────────────────────────
// Streaming iterator
// ─────────────────────────────────────────────────────────────────────────────

/// An iterator that yields successive fixed-size byte chunks from a tensor's
/// raw data buffer.  Use this for progressive / streaming loading.
///
/// ```
/// # use oxibonsai_model::gguf_loader::TensorChunkIter;
/// let data = vec![0u8; 100];
/// let mut iter = TensorChunkIter::new(data, 32);
/// assert_eq!(iter.total_chunks(), 4); // ceil(100/32)
/// ```
pub struct TensorChunkIter {
    data: Vec<u8>,
    chunk_size: usize,
    pos: usize,
}

impl TensorChunkIter {
    /// Create a new chunk iterator over `data` with the given `chunk_size`.
    pub fn new(data: Vec<u8>, chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be > 0");
        Self {
            data,
            chunk_size,
            pos: 0,
        }
    }

    /// Total number of chunks (rounded up for any partial final chunk).
    pub fn total_chunks(&self) -> usize {
        if self.data.is_empty() {
            return 0;
        }
        self.data.len().div_ceil(self.chunk_size)
    }

    /// Remaining bytes not yet yielded by the iterator.
    pub fn bytes_remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }
}

impl Iterator for TensorChunkIter {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.data.len() {
            return None;
        }
        let end = (self.pos + self.chunk_size).min(self.data.len());
        let chunk = self.data[self.pos..end].to_vec();
        self.pos = end;
        Some(chunk)
    }
}

//! On-disk model cache for fast model reloading.
//!
//! Caches quantized model weights + metadata in a binary format (`.oxcache`)
//! for faster cold-start loading vs. re-parsing GGUF files.
//!
//! Format:
//!   Header: `"OXCA"` (4 bytes) + version u32 + num\_entries u64 + metadata\_len u32
//!   Metadata: JSON string (hand-serialised, no serde)
//!   Per entry: name\_len u32 + name (UTF-8) + quant\_type\_len u32 + quant\_type + data\_len u64 + data bytes

use std::collections::HashMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::time::SystemTime;

/// Magic bytes identifying an OxiBonsai disk cache file.
pub const CACHE_MAGIC: &[u8; 4] = b"OXCA";
/// Current cache format version.
pub const CACHE_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Error
// ---------------------------------------------------------------------------

/// Errors produced by disk-cache operations.
#[derive(Debug, thiserror::Error)]
pub enum DiskCacheError {
    /// Underlying I/O failure.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// File does not start with `OXCA`.
    #[error("invalid cache magic")]
    InvalidMagic,
    /// Cache file was written by a newer/older incompatible version.
    #[error("unsupported cache version: {0}")]
    UnsupportedVersion(u32),
    /// Hand-rolled JSON metadata could not be parsed.
    #[error("metadata parse error: {0}")]
    MetadataParse(String),
    /// Cache is older than its source file.
    #[error("cache is stale")]
    StaleCache,
}

// ---------------------------------------------------------------------------
// CacheEntry
// ---------------------------------------------------------------------------

/// An entry in the disk cache, representing one named tensor blob.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Tensor / weight name (e.g. `"layers.0.attn.q_proj"`).
    pub name: String,
    /// Raw bytes of the (possibly quantized) tensor data.
    pub data: Vec<u8>,
    /// Quantization format identifier (e.g. `"f32"`, `"int8"`, `"q1_0_g128"`).
    pub quant_type: String,
}

impl CacheEntry {
    /// Create a new cache entry.
    pub fn new(name: impl Into<String>, data: Vec<u8>, quant_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data,
            quant_type: quant_type.into(),
        }
    }

    /// Total size of the raw data in bytes.
    pub fn size_bytes(&self) -> usize {
        self.data.len()
    }
}

// ---------------------------------------------------------------------------
// DiskCache
// ---------------------------------------------------------------------------

/// In-memory representation of a `.oxcache` file.
#[derive(Debug)]
pub struct DiskCache {
    entries: Vec<CacheEntry>,
    metadata: HashMap<String, String>,
}

impl Default for DiskCache {
    fn default() -> Self {
        Self::new()
    }
}

impl DiskCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Append an entry.
    pub fn add_entry(&mut self, entry: CacheEntry) {
        self.entries.push(entry);
    }

    /// Set a metadata key-value pair.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Look up a metadata value.
    pub fn get_metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Find an entry by name.
    pub fn get_entry(&self, name: &str) -> Option<&CacheEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Number of entries.
    pub fn num_entries(&self) -> usize {
        self.entries.len()
    }

    /// Sum of all entry data sizes.
    pub fn total_data_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.data.len()).sum()
    }

    // ----- persistence -----

    /// Save to a file path.
    pub fn save(&self, path: &Path) -> Result<(), DiskCacheError> {
        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);
        self.write_to(&mut writer)
    }

    /// Load from a file path.
    pub fn load(path: &Path) -> Result<Self, DiskCacheError> {
        let file = std::fs::File::open(path)?;
        let mut reader = BufReader::new(file);
        Self::read_from(&mut reader)
    }

    /// Serialize to an arbitrary writer.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<(), DiskCacheError> {
        // Magic
        writer.write_all(CACHE_MAGIC)?;

        // Version (u32 LE)
        writer.write_all(&CACHE_VERSION.to_le_bytes())?;

        // Number of entries (u64 LE)
        writer.write_all(&(self.entries.len() as u64).to_le_bytes())?;

        // Metadata as JSON string
        let meta_json = metadata_to_json(&self.metadata);
        let meta_bytes = meta_json.as_bytes();
        writer.write_all(&(meta_bytes.len() as u32).to_le_bytes())?;
        writer.write_all(meta_bytes)?;

        // Entries
        for entry in &self.entries {
            // name
            let name_bytes = entry.name.as_bytes();
            writer.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(name_bytes)?;

            // quant_type
            let qt_bytes = entry.quant_type.as_bytes();
            writer.write_all(&(qt_bytes.len() as u32).to_le_bytes())?;
            writer.write_all(qt_bytes)?;

            // data
            writer.write_all(&(entry.data.len() as u64).to_le_bytes())?;
            writer.write_all(&entry.data)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Deserialize from an arbitrary reader.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self, DiskCacheError> {
        // Magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != CACHE_MAGIC {
            return Err(DiskCacheError::InvalidMagic);
        }

        // Version
        let mut buf4 = [0u8; 4];
        reader.read_exact(&mut buf4)?;
        let version = u32::from_le_bytes(buf4);
        if version != CACHE_VERSION {
            return Err(DiskCacheError::UnsupportedVersion(version));
        }

        // Num entries
        let mut buf8 = [0u8; 8];
        reader.read_exact(&mut buf8)?;
        let num_entries = u64::from_le_bytes(buf8) as usize;

        // Metadata
        reader.read_exact(&mut buf4)?;
        let meta_len = u32::from_le_bytes(buf4) as usize;
        let mut meta_buf = vec![0u8; meta_len];
        reader.read_exact(&mut meta_buf)?;
        let meta_str = String::from_utf8(meta_buf)
            .map_err(|e| DiskCacheError::MetadataParse(e.to_string()))?;
        let metadata = metadata_from_json(&meta_str)?;

        // Entries
        let mut entries = Vec::with_capacity(num_entries);
        for _ in 0..num_entries {
            // name
            reader.read_exact(&mut buf4)?;
            let name_len = u32::from_le_bytes(buf4) as usize;
            let mut name_buf = vec![0u8; name_len];
            reader.read_exact(&mut name_buf)?;
            let name = String::from_utf8(name_buf)
                .map_err(|e| DiskCacheError::MetadataParse(e.to_string()))?;

            // quant_type
            reader.read_exact(&mut buf4)?;
            let qt_len = u32::from_le_bytes(buf4) as usize;
            let mut qt_buf = vec![0u8; qt_len];
            reader.read_exact(&mut qt_buf)?;
            let quant_type = String::from_utf8(qt_buf)
                .map_err(|e| DiskCacheError::MetadataParse(e.to_string()))?;

            // data
            reader.read_exact(&mut buf8)?;
            let data_len = u64::from_le_bytes(buf8) as usize;
            let mut data = vec![0u8; data_len];
            reader.read_exact(&mut data)?;

            entries.push(CacheEntry {
                name,
                data,
                quant_type,
            });
        }

        Ok(Self { entries, metadata })
    }

    /// Check if a cache file exists and has valid magic + version.
    pub fn is_valid_cache(path: &Path) -> bool {
        let file = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let mut reader = BufReader::new(file);

        let mut magic = [0u8; 4];
        if reader.read_exact(&mut magic).is_err() {
            return false;
        }
        if &magic != CACHE_MAGIC {
            return false;
        }

        let mut buf4 = [0u8; 4];
        if reader.read_exact(&mut buf4).is_err() {
            return false;
        }
        let version = u32::from_le_bytes(buf4);
        version == CACHE_VERSION
    }

    /// Returns `Ok(true)` if the cache file is newer than the source file.
    pub fn is_fresh(cache_path: &Path, source_path: &Path) -> Result<bool, DiskCacheError> {
        let cache_meta = std::fs::metadata(cache_path)?;
        let source_meta = std::fs::metadata(source_path)?;

        let cache_time = cache_meta.modified().map_err(DiskCacheError::Io)?;
        let source_time = source_meta.modified().map_err(DiskCacheError::Io)?;

        Ok(cache_time >= source_time)
    }
}

// ---------------------------------------------------------------------------
// CacheManager
// ---------------------------------------------------------------------------

/// Manages multiple cached model files with LRU eviction.
#[derive(Debug)]
pub struct CacheManager {
    cache_dir: String,
    max_cache_size_bytes: usize,
    entries: Vec<CacheFileInfo>,
}

/// Information about one cached model file on disk.
#[derive(Debug, Clone)]
pub struct CacheFileInfo {
    /// Absolute path to the `.oxcache` file.
    pub path: String,
    /// Size on disk in bytes.
    pub size_bytes: usize,
    /// Last time this cache was accessed / loaded.
    pub last_accessed: SystemTime,
    /// Human-readable model name.
    pub model_name: String,
}

impl CacheManager {
    /// Create a new manager for the given directory with a byte budget.
    pub fn new(cache_dir: impl Into<String>, max_size_bytes: usize) -> Self {
        Self {
            cache_dir: cache_dir.into(),
            max_cache_size_bytes: max_size_bytes,
            entries: Vec::new(),
        }
    }

    /// Register a cached file.
    pub fn register(&mut self, info: CacheFileInfo) {
        self.entries.push(info);
    }

    /// Total bytes used by all registered cache files.
    pub fn total_used_bytes(&self) -> usize {
        self.entries.iter().map(|e| e.size_bytes).sum()
    }

    /// Whether total usage exceeds the budget.
    pub fn should_evict(&self) -> bool {
        self.total_used_bytes() > self.max_cache_size_bytes
    }

    /// Candidates for eviction, sorted oldest-first (LRU).
    pub fn eviction_candidates(&self) -> Vec<&CacheFileInfo> {
        let mut sorted: Vec<&CacheFileInfo> = self.entries.iter().collect();
        sorted.sort_by_key(|e| e.last_accessed);
        sorted
    }

    /// Fraction of budget used (0.0 – 1.0+).
    pub fn utilization(&self) -> f32 {
        if self.max_cache_size_bytes == 0 {
            return 0.0;
        }
        self.total_used_bytes() as f32 / self.max_cache_size_bytes as f32
    }

    /// Human-readable summary.
    pub fn summary(&self) -> String {
        let used_mb = self.total_used_bytes() as f64 / (1024.0 * 1024.0);
        let max_mb = self.max_cache_size_bytes as f64 / (1024.0 * 1024.0);
        let pct = self.utilization() * 100.0;
        format!(
            "Cache dir: {dir}, {n} models, {used:.1}/{max:.1} MB ({pct:.1}%)",
            dir = self.cache_dir,
            n = self.entries.len(),
            used = used_mb,
            max = max_mb,
        )
    }
}

// ---------------------------------------------------------------------------
// Manual JSON helpers (no serde)
// ---------------------------------------------------------------------------

/// Serialize a `HashMap<String, String>` to a JSON object string.
fn metadata_to_json(map: &HashMap<String, String>) -> String {
    let mut out = String::from("{");
    let mut first = true;
    // Sort keys for deterministic output.
    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    for key in keys {
        let value = &map[key];
        if !first {
            out.push(',');
        }
        first = false;
        out.push('"');
        json_escape_into(&mut out, key);
        out.push_str("\":\"");
        json_escape_into(&mut out, value);
        out.push('"');
    }
    out.push('}');
    out
}

/// Deserialize a JSON object string to `HashMap<String, String>`.
fn metadata_from_json(s: &str) -> Result<HashMap<String, String>, DiskCacheError> {
    let s = s.trim();
    if s == "{}" || s.is_empty() {
        return Ok(HashMap::new());
    }
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'{') || bytes.last() != Some(&b'}') {
        return Err(DiskCacheError::MetadataParse(format!(
            "expected JSON object, got: {s}"
        )));
    }
    let inner = &s[1..s.len() - 1];
    let mut map = HashMap::new();
    if inner.trim().is_empty() {
        return Ok(map);
    }

    let chars: Vec<char> = inner.chars().collect();
    let mut pos = 0usize;

    loop {
        // Skip whitespace / commas.
        while pos < chars.len() && (chars[pos] == ',' || chars[pos].is_whitespace()) {
            pos += 1;
        }
        if pos >= chars.len() {
            break;
        }
        if chars[pos] != '"' {
            return Err(DiskCacheError::MetadataParse(format!(
                "expected '\"' at position {pos}, got '{}'",
                chars[pos]
            )));
        }
        pos += 1;
        let (key, new_pos) = parse_json_string(&chars, pos)?;
        pos = new_pos;

        // Skip ws, expect ':'
        skip_ws(&chars, &mut pos);
        if pos >= chars.len() || chars[pos] != ':' {
            return Err(DiskCacheError::MetadataParse(format!(
                "expected ':' after key '{key}'"
            )));
        }
        pos += 1;
        skip_ws(&chars, &mut pos);

        if pos >= chars.len() || chars[pos] != '"' {
            return Err(DiskCacheError::MetadataParse(format!(
                "expected '\"' for value of key '{key}'"
            )));
        }
        pos += 1;
        let (value, new_pos) = parse_json_string(&chars, pos)?;
        pos = new_pos;

        map.insert(key, value);
    }

    Ok(map)
}

fn parse_json_string(chars: &[char], mut pos: usize) -> Result<(String, usize), DiskCacheError> {
    let mut s = String::new();
    while pos < chars.len() {
        match chars[pos] {
            '"' => {
                pos += 1;
                return Ok((s, pos));
            }
            '\\' => {
                pos += 1;
                if pos >= chars.len() {
                    return Err(DiskCacheError::MetadataParse(
                        "unexpected end after backslash".into(),
                    ));
                }
                match chars[pos] {
                    '"' => s.push('"'),
                    '\\' => s.push('\\'),
                    'n' => s.push('\n'),
                    'r' => s.push('\r'),
                    't' => s.push('\t'),
                    other => {
                        return Err(DiskCacheError::MetadataParse(format!(
                            "unknown escape '\\{other}'"
                        )));
                    }
                }
                pos += 1;
            }
            ch => {
                s.push(ch);
                pos += 1;
            }
        }
    }
    Err(DiskCacheError::MetadataParse("unterminated string".into()))
}

fn skip_ws(chars: &[char], pos: &mut usize) {
    while *pos < chars.len() && chars[*pos].is_whitespace() {
        *pos += 1;
    }
}

fn json_escape_into(out: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
}

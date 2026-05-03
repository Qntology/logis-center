//! Streaming GGUF reader for progressive parsing.
//!
//! This module provides a state-machine-based parser that can consume GGUF data
//! incrementally as it arrives (e.g., from a network download), without requiring
//! the full file to be present in memory.
//!
//! # Usage
//!
//! ```rust,no_run
//! use oxibonsai_core::gguf::streaming::GgufStreamParser;
//!
//! let mut parser = GgufStreamParser::new();
//! // Feed data as it arrives:
//! // let consumed = parser.feed(&chunk)?;
//! // Check completion:
//! // if parser.is_complete() { let result = parser.finish()?; }
//! ```

use crate::error::BonsaiError;
use crate::gguf::types::{GgufTensorType, GgufValueType};

/// GGUF magic number: "GGUF" in little-endian = 0x46554747.
const GGUF_MAGIC: u32 = 0x4655_4747;

/// GGUF header size: magic(4) + version(4) + tensor_count(8) + metadata_kv_count(8) = 24 bytes.
const HEADER_SIZE: usize = 24;

/// Maximum string length accepted (256 MB).
const MAX_STRING_LEN: u64 = 256 * 1024 * 1024;

/// Maximum array element count accepted (16M entries).
const MAX_ARRAY_COUNT: u64 = 16 * 1024 * 1024;

/// Maximum tensor dimensions.
const MAX_TENSOR_DIMS: u32 = 1024;

/// Default alignment for tensor data in GGUF files (32 bytes).
const DEFAULT_ALIGNMENT: usize = 32;

/// State machine for progressive GGUF parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamState {
    /// Waiting for the 24-byte header.
    ReadingHeader,
    /// Parsing metadata key-value pairs; `remaining` entries left.
    ReadingMetadata { remaining: u64 },
    /// Parsing tensor info entries; `remaining` entries left.
    ReadingTensorInfo { remaining: u64 },
    /// All metadata and tensor info parsed; tensor data follows.
    ReadingTensorData,
    /// Parsing is fully complete.
    Complete,
}

/// A metadata value from the streaming parser.
///
/// This mirrors `MetadataValue` but is self-contained so the streaming module
/// does not depend on the cursor-based metadata parser.
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

/// Accumulated parse result from streaming.
#[derive(Debug, Clone)]
pub struct StreamedGguf {
    /// GGUF format version.
    pub version: u32,
    /// Parsed metadata key-value pairs (in order).
    pub metadata: Vec<(String, GgufValue)>,
    /// Parsed tensor info entries (in order).
    pub tensor_infos: Vec<StreamedTensorInfo>,
    /// Byte offset where tensor data begins (aligned).
    pub data_offset: u64,
}

/// Tensor information from the streaming parser.
#[derive(Debug, Clone)]
pub struct StreamedTensorInfo {
    /// Tensor name.
    pub name: String,
    /// Number of dimensions.
    pub n_dims: u32,
    /// Dimensions (up to 4; unused dims are 0).
    pub dims: [u64; 4],
    /// Quantization / data type.
    pub tensor_type: GgufTensorType,
    /// Byte offset within the tensor data section.
    pub offset: u64,
}

/// Streaming GGUF parser.
///
/// Feed bytes progressively via [`feed`](Self::feed). The parser buffers
/// incomplete data internally and advances through [`StreamState`] stages
/// as enough bytes accumulate.
#[derive(Debug)]
pub struct GgufStreamParser {
    state: StreamState,
    buffer: Vec<u8>,
    result: StreamedGguf,
    bytes_consumed: u64,
    // Cached header counts for progress estimation
    total_metadata: u64,
    total_tensors: u64,
}

impl GgufStreamParser {
    /// Create a new streaming parser in the initial state.
    pub fn new() -> Self {
        Self {
            state: StreamState::ReadingHeader,
            buffer: Vec::with_capacity(4096),
            result: StreamedGguf {
                version: 0,
                metadata: Vec::new(),
                tensor_infos: Vec::new(),
                data_offset: 0,
            },
            bytes_consumed: 0,
            total_metadata: 0,
            total_tensors: 0,
        }
    }

    /// Feed bytes into the parser. Returns the number of bytes consumed from `data`.
    ///
    /// If the parser needs more data to make progress, it returns `Ok(0)` after
    /// buffering the input. Call again with more data when available.
    pub fn feed(&mut self, data: &[u8]) -> Result<usize, BonsaiError> {
        if data.is_empty() {
            return Ok(0);
        }

        // Append new data to internal buffer
        self.buffer.extend_from_slice(data);
        let input_len = data.len();

        // Process as much as possible from the buffer
        loop {
            match &self.state {
                StreamState::ReadingHeader => {
                    if !self.try_parse_header()? {
                        break;
                    }
                }
                StreamState::ReadingMetadata { remaining } => {
                    if *remaining == 0 {
                        self.transition_to_tensor_info();
                        continue;
                    }
                    if !self.try_parse_one_metadata()? {
                        break;
                    }
                }
                StreamState::ReadingTensorInfo { remaining } => {
                    if *remaining == 0 {
                        self.finalize();
                        break;
                    }
                    if !self.try_parse_one_tensor_info()? {
                        break;
                    }
                }
                StreamState::ReadingTensorData | StreamState::Complete => {
                    break;
                }
            }
        }

        Ok(input_len)
    }

    /// Check if parsing is complete (all metadata + tensor info parsed).
    pub fn is_complete(&self) -> bool {
        matches!(
            self.state,
            StreamState::ReadingTensorData | StreamState::Complete
        )
    }

    /// Get current parse state.
    pub fn state(&self) -> &StreamState {
        &self.state
    }

    /// Get total bytes consumed so far.
    pub fn bytes_consumed(&self) -> u64 {
        self.bytes_consumed
    }

    /// Take the final result. Only valid after [`is_complete`](Self::is_complete) returns true.
    pub fn finish(self) -> Result<StreamedGguf, BonsaiError> {
        if !self.is_complete() {
            return Err(BonsaiError::UnexpectedEof {
                offset: self.bytes_consumed,
            });
        }
        Ok(self.result)
    }

    /// Estimated progress as a fraction in `[0.0, 1.0]`.
    ///
    /// Before the header is parsed, progress is based on bytes towards the 24-byte header.
    /// After the header, progress is based on how many metadata + tensor info entries
    /// have been parsed out of the total expected.
    pub fn progress(&self) -> f32 {
        match &self.state {
            StreamState::ReadingHeader => {
                // Progress towards header completion
                let have = self.buffer.len().min(HEADER_SIZE) as f32;
                (have / HEADER_SIZE as f32) * 0.1 // header is ~10% of progress
            }
            StreamState::ReadingMetadata { remaining } => {
                let total = self.total_metadata + self.total_tensors;
                if total == 0 {
                    return 0.5;
                }
                let done = self.total_metadata - remaining;
                0.1 + (done as f32 / total as f32) * 0.9
            }
            StreamState::ReadingTensorInfo { remaining } => {
                let total = self.total_metadata + self.total_tensors;
                if total == 0 {
                    return 0.9;
                }
                let done = self.total_metadata + (self.total_tensors - remaining);
                0.1 + (done as f32 / total as f32) * 0.9
            }
            StreamState::ReadingTensorData | StreamState::Complete => 1.0,
        }
    }

    // ---- Internal parsing methods ----

    /// Try to parse the 24-byte header from the buffer.
    /// Returns true if successful (state advanced), false if not enough data.
    fn try_parse_header(&mut self) -> Result<bool, BonsaiError> {
        if self.buffer.len() < HEADER_SIZE {
            return Ok(false);
        }

        let magic = read_u32_le(&self.buffer, 0);
        if magic != GGUF_MAGIC {
            return Err(BonsaiError::InvalidMagic { magic });
        }

        let version = read_u32_le(&self.buffer, 4);
        if version != 2 && version != 3 {
            return Err(BonsaiError::UnsupportedVersion { version });
        }

        let tensor_count = read_u64_le(&self.buffer, 8);
        let metadata_kv_count = read_u64_le(&self.buffer, 16);

        self.result.version = version;
        self.total_metadata = metadata_kv_count;
        self.total_tensors = tensor_count;
        self.bytes_consumed += HEADER_SIZE as u64;

        // Remove consumed header bytes from buffer
        self.buffer.drain(..HEADER_SIZE);

        self.state = StreamState::ReadingMetadata {
            remaining: metadata_kv_count,
        };
        Ok(true)
    }

    /// Try to parse one metadata KV entry from the buffer.
    /// Returns true if successful, false if not enough data.
    fn try_parse_one_metadata(&mut self) -> Result<bool, BonsaiError> {
        let mut pos = 0;

        // Parse key string
        let key = match try_read_gguf_string(&self.buffer, pos)? {
            Some((s, new_pos)) => {
                pos = new_pos;
                s
            }
            None => return Ok(false),
        };

        // Parse value type
        if pos + 4 > self.buffer.len() {
            return Ok(false);
        }
        let value_type_id = read_u32_le(&self.buffer, pos);
        let value_type = GgufValueType::from_id(value_type_id)?;
        pos += 4;

        // Parse value
        let (value, new_pos) = match try_read_value(&self.buffer, pos, value_type)? {
            Some(v) => v,
            None => return Ok(false),
        };
        pos = new_pos;

        self.bytes_consumed += pos as u64;
        self.buffer.drain(..pos);
        self.result.metadata.push((key, value));

        // Decrement remaining
        if let StreamState::ReadingMetadata { remaining } = &mut self.state {
            *remaining -= 1;
        }

        Ok(true)
    }

    /// Transition from metadata to tensor info reading.
    fn transition_to_tensor_info(&mut self) {
        self.state = StreamState::ReadingTensorInfo {
            remaining: self.total_tensors,
        };
    }

    /// Try to parse one tensor info entry from the buffer.
    /// Returns true if successful, false if not enough data.
    fn try_parse_one_tensor_info(&mut self) -> Result<bool, BonsaiError> {
        let mut pos = 0;

        // Parse name
        let name = match try_read_gguf_string(&self.buffer, pos)? {
            Some((s, new_pos)) => {
                pos = new_pos;
                s
            }
            None => return Ok(false),
        };

        // Parse n_dims (u32)
        if pos + 4 > self.buffer.len() {
            return Ok(false);
        }
        let n_dims = read_u32_le(&self.buffer, pos);
        pos += 4;

        if n_dims > MAX_TENSOR_DIMS {
            return Err(BonsaiError::InvalidMetadata {
                key: name,
                reason: format!("tensor has too many dimensions: {n_dims}"),
            });
        }

        // Parse dims (n_dims * u64)
        let dims_bytes = n_dims as usize * 8;
        if pos + dims_bytes > self.buffer.len() {
            return Ok(false);
        }
        let mut dims = [0u64; 4];
        for (i, dim) in dims.iter_mut().enumerate().take(n_dims.min(4) as usize) {
            *dim = read_u64_le(&self.buffer, pos + i * 8);
        }
        pos += dims_bytes;

        // Parse tensor type (u32)
        if pos + 4 > self.buffer.len() {
            return Ok(false);
        }
        let type_id = read_u32_le(&self.buffer, pos);
        let tensor_type = GgufTensorType::from_id(type_id)?;
        pos += 4;

        // Parse offset (u64)
        if pos + 8 > self.buffer.len() {
            return Ok(false);
        }
        let offset = read_u64_le(&self.buffer, pos);
        pos += 8;

        self.bytes_consumed += pos as u64;
        self.buffer.drain(..pos);

        self.result.tensor_infos.push(StreamedTensorInfo {
            name,
            n_dims,
            dims,
            tensor_type,
            offset,
        });

        // Decrement remaining
        if let StreamState::ReadingTensorInfo { remaining } = &mut self.state {
            *remaining -= 1;
        }

        Ok(true)
    }

    /// Finalize parsing: compute data offset with alignment and transition to complete state.
    fn finalize(&mut self) {
        // Check for alignment override in metadata
        let alignment = self
            .result
            .metadata
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| match v {
                GgufValue::Uint32(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(DEFAULT_ALIGNMENT);

        let offset = self.bytes_consumed as usize;
        let aligned = (offset + alignment - 1) & !(alignment - 1);
        self.result.data_offset = aligned as u64;

        self.state = StreamState::ReadingTensorData;
    }
}

impl Default for GgufStreamParser {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Low-level buffer readers (no std::io dependency) ----

/// Read a little-endian u32 from a byte slice at the given offset.
fn read_u32_le(buf: &[u8], offset: usize) -> u32 {
    let b = &buf[offset..offset + 4];
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Read a little-endian u64 from a byte slice at the given offset.
fn read_u64_le(buf: &[u8], offset: usize) -> u64 {
    let b = &buf[offset..offset + 8];
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

/// Read a little-endian i8 from a byte slice at the given offset.
fn read_i8_le(buf: &[u8], offset: usize) -> i8 {
    buf[offset] as i8
}

/// Read a little-endian i16 from a byte slice at the given offset.
fn read_i16_le(buf: &[u8], offset: usize) -> i16 {
    let b = &buf[offset..offset + 2];
    i16::from_le_bytes([b[0], b[1]])
}

/// Read a little-endian u16 from a byte slice at the given offset.
fn read_u16_le(buf: &[u8], offset: usize) -> u16 {
    let b = &buf[offset..offset + 2];
    u16::from_le_bytes([b[0], b[1]])
}

/// Read a little-endian i32 from a byte slice at the given offset.
fn read_i32_le(buf: &[u8], offset: usize) -> i32 {
    let b = &buf[offset..offset + 4];
    i32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Read a little-endian i64 from a byte slice at the given offset.
fn read_i64_le(buf: &[u8], offset: usize) -> i64 {
    let b = &buf[offset..offset + 8];
    i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

/// Read a little-endian f32 from a byte slice at the given offset.
fn read_f32_le(buf: &[u8], offset: usize) -> f32 {
    let b = &buf[offset..offset + 4];
    f32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// Read a little-endian f64 from a byte slice at the given offset.
fn read_f64_le(buf: &[u8], offset: usize) -> f64 {
    let b = &buf[offset..offset + 8];
    f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

/// Try to read a GGUF string from the buffer at `offset`.
/// Returns `Some((string, new_offset))` if enough data, `None` otherwise.
fn try_read_gguf_string(buf: &[u8], offset: usize) -> Result<Option<(String, usize)>, BonsaiError> {
    if offset + 8 > buf.len() {
        return Ok(None);
    }
    let len = read_u64_le(buf, offset);
    if len > MAX_STRING_LEN {
        return Err(BonsaiError::InvalidString {
            offset: offset as u64,
        });
    }
    let str_end = offset + 8 + len as usize;
    if str_end > buf.len() {
        return Ok(None);
    }
    let s =
        std::str::from_utf8(&buf[offset + 8..str_end]).map_err(|_| BonsaiError::InvalidString {
            offset: offset as u64,
        })?;
    Ok(Some((s.to_string(), str_end)))
}

/// Try to read a typed GGUF value from the buffer at `offset`.
/// Returns `Some((value, new_offset))` if enough data, `None` otherwise.
fn try_read_value(
    buf: &[u8],
    offset: usize,
    value_type: GgufValueType,
) -> Result<Option<(GgufValue, usize)>, BonsaiError> {
    match value_type {
        GgufValueType::Uint8 => {
            if offset + 1 > buf.len() {
                return Ok(None);
            }
            Ok(Some((GgufValue::Uint8(buf[offset]), offset + 1)))
        }
        GgufValueType::Int8 => {
            if offset + 1 > buf.len() {
                return Ok(None);
            }
            Ok(Some((GgufValue::Int8(read_i8_le(buf, offset)), offset + 1)))
        }
        GgufValueType::Uint16 => {
            if offset + 2 > buf.len() {
                return Ok(None);
            }
            Ok(Some((
                GgufValue::Uint16(read_u16_le(buf, offset)),
                offset + 2,
            )))
        }
        GgufValueType::Int16 => {
            if offset + 2 > buf.len() {
                return Ok(None);
            }
            Ok(Some((
                GgufValue::Int16(read_i16_le(buf, offset)),
                offset + 2,
            )))
        }
        GgufValueType::Uint32 => {
            if offset + 4 > buf.len() {
                return Ok(None);
            }
            Ok(Some((
                GgufValue::Uint32(read_u32_le(buf, offset)),
                offset + 4,
            )))
        }
        GgufValueType::Int32 => {
            if offset + 4 > buf.len() {
                return Ok(None);
            }
            Ok(Some((
                GgufValue::Int32(read_i32_le(buf, offset)),
                offset + 4,
            )))
        }
        GgufValueType::Float32 => {
            if offset + 4 > buf.len() {
                return Ok(None);
            }
            Ok(Some((
                GgufValue::Float32(read_f32_le(buf, offset)),
                offset + 4,
            )))
        }
        GgufValueType::Bool => {
            if offset + 1 > buf.len() {
                return Ok(None);
            }
            Ok(Some((GgufValue::Bool(buf[offset] != 0), offset + 1)))
        }
        GgufValueType::String => match try_read_gguf_string(buf, offset)? {
            Some((s, new_pos)) => Ok(Some((GgufValue::String(s), new_pos))),
            None => Ok(None),
        },
        GgufValueType::Array => {
            // Need element type (u32) + count (u64) = 12 bytes minimum
            if offset + 12 > buf.len() {
                return Ok(None);
            }
            let elem_type_id = read_u32_le(buf, offset);
            let elem_type = GgufValueType::from_id(elem_type_id)?;
            let count = read_u64_le(buf, offset + 4);
            if count > MAX_ARRAY_COUNT {
                return Err(BonsaiError::InvalidMetadata {
                    key: String::new(),
                    reason: format!("array count too large: {count}"),
                });
            }

            let mut pos = offset + 12;
            let mut values = Vec::with_capacity(count as usize);
            for _ in 0..count {
                match try_read_value(buf, pos, elem_type)? {
                    Some((v, new_pos)) => {
                        values.push(v);
                        pos = new_pos;
                    }
                    None => return Ok(None),
                }
            }
            Ok(Some((GgufValue::Array(values), pos)))
        }
        GgufValueType::Uint64 => {
            if offset + 8 > buf.len() {
                return Ok(None);
            }
            Ok(Some((
                GgufValue::Uint64(read_u64_le(buf, offset)),
                offset + 8,
            )))
        }
        GgufValueType::Int64 => {
            if offset + 8 > buf.len() {
                return Ok(None);
            }
            Ok(Some((
                GgufValue::Int64(read_i64_le(buf, offset)),
                offset + 8,
            )))
        }
        GgufValueType::Float64 => {
            if offset + 8 > buf.len() {
                return Ok(None);
            }
            Ok(Some((
                GgufValue::Float64(read_f64_le(buf, offset)),
                offset + 8,
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_creates_new_parser() {
        let parser = GgufStreamParser::default();
        assert_eq!(*parser.state(), StreamState::ReadingHeader);
        assert_eq!(parser.bytes_consumed(), 0);
        assert!(!parser.is_complete());
    }
}

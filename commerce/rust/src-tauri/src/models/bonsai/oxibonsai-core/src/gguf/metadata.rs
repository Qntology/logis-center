//! GGUF metadata key-value store.
//!
//! The GGUF metadata section stores typed key-value pairs that describe
//! the model architecture, tokenizer configuration, and other properties.

use std::collections::HashMap;

use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Read;

use crate::error::{BonsaiError, BonsaiResult};
use crate::gguf::types::GgufValueType;

/// A typed metadata value from the GGUF key-value store.
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl MetadataValue {
    /// Try to extract a u32 value.
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint32(v) => Some(*v),
            Self::Uint64(v) => u32::try_from(*v).ok(),
            Self::Int32(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Try to extract a u64 value.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint64(v) => Some(*v),
            Self::Uint32(v) => Some(u64::from(*v)),
            Self::Int64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Try to extract a f32 value.
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    /// Try to extract a string value.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }

    /// Try to extract a bool value.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }
}

/// Key-value metadata store from GGUF file.
#[derive(Debug, Clone)]
pub struct MetadataStore {
    entries: HashMap<String, MetadataValue>,
}

impl MetadataStore {
    /// Create an empty metadata store.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Read metadata entries from a byte-slice cursor.
    pub fn parse(data: &[u8], offset: usize, count: u64) -> BonsaiResult<(Self, usize)> {
        let mut cursor = std::io::Cursor::new(data);
        cursor.set_position(offset as u64);

        let mut store = Self::new();
        for _ in 0..count {
            let (key, value) = read_kv_pair(&mut cursor)?;
            store.entries.insert(key, value);
        }

        Ok((store, cursor.position() as usize))
    }

    /// Get a metadata value by key.
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.entries.get(key)
    }

    /// Get a required string value, returning an error if missing.
    pub fn get_string(&self, key: &str) -> BonsaiResult<&str> {
        self.get(key)
            .and_then(|v| v.as_str())
            .ok_or_else(|| BonsaiError::MissingConfigKey {
                key: key.to_string(),
            })
    }

    /// Get a required u32 value, returning an error if missing.
    pub fn get_u32(&self, key: &str) -> BonsaiResult<u32> {
        self.get(key)
            .and_then(|v| v.as_u32())
            .ok_or_else(|| BonsaiError::MissingConfigKey {
                key: key.to_string(),
            })
    }

    /// Get a required u64 value, returning an error if missing.
    pub fn get_u64(&self, key: &str) -> BonsaiResult<u64> {
        self.get(key)
            .and_then(|v| v.as_u64())
            .ok_or_else(|| BonsaiError::MissingConfigKey {
                key: key.to_string(),
            })
    }

    /// Get a required f32 value.
    pub fn get_f32(&self, key: &str) -> BonsaiResult<f32> {
        self.get(key)
            .and_then(|v| v.as_f32())
            .ok_or_else(|| BonsaiError::MissingConfigKey {
                key: key.to_string(),
            })
    }

    /// Get an optional u32 value with a default.
    pub fn get_u32_or(&self, key: &str, default: u32) -> u32 {
        self.get(key).and_then(|v| v.as_u32()).unwrap_or(default)
    }

    /// Get an optional f32 value with a default.
    pub fn get_f32_or(&self, key: &str, default: f32) -> f32 {
        self.get(key).and_then(|v| v.as_f32()).unwrap_or(default)
    }

    /// Number of entries in the store.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &MetadataValue)> {
        self.entries.iter()
    }
}

impl Default for MetadataStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Maximum string length we accept from GGUF metadata (256 MB).
const MAX_STRING_LEN: u64 = 256 * 1024 * 1024;

/// Maximum array element count we accept from GGUF metadata (16 M entries).
const MAX_ARRAY_COUNT: u64 = 16 * 1024 * 1024;

/// Read a GGUF string: [u64 length] [utf8 bytes].
fn read_gguf_string<R: Read>(reader: &mut R) -> BonsaiResult<String> {
    let len = reader
        .read_u64::<LittleEndian>()
        .map_err(BonsaiError::MmapError)?;
    if len > MAX_STRING_LEN {
        return Err(BonsaiError::InvalidString { offset: 0 });
    }
    let mut buf = vec![0u8; len as usize];
    reader
        .read_exact(&mut buf)
        .map_err(BonsaiError::MmapError)?;
    String::from_utf8(buf).map_err(|_| BonsaiError::InvalidString { offset: 0 })
}

/// Read a single typed value from the reader.
fn read_value<R: Read>(reader: &mut R, value_type: GgufValueType) -> BonsaiResult<MetadataValue> {
    match value_type {
        GgufValueType::Uint8 => {
            let v = reader.read_u8().map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Uint8(v))
        }
        GgufValueType::Int8 => {
            let v = reader.read_i8().map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Int8(v))
        }
        GgufValueType::Uint16 => {
            let v = reader
                .read_u16::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Uint16(v))
        }
        GgufValueType::Int16 => {
            let v = reader
                .read_i16::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Int16(v))
        }
        GgufValueType::Uint32 => {
            let v = reader
                .read_u32::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Uint32(v))
        }
        GgufValueType::Int32 => {
            let v = reader
                .read_i32::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Int32(v))
        }
        GgufValueType::Float32 => {
            let v = reader
                .read_f32::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Float32(v))
        }
        GgufValueType::Bool => {
            let v = reader.read_u8().map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Bool(v != 0))
        }
        GgufValueType::String => {
            let s = read_gguf_string(reader)?;
            Ok(MetadataValue::String(s))
        }
        GgufValueType::Array => {
            let elem_type_id = reader
                .read_u32::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            let elem_type = GgufValueType::from_id(elem_type_id)?;
            let count = reader
                .read_u64::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            if count > MAX_ARRAY_COUNT {
                return Err(BonsaiError::InvalidString { offset: 0 });
            }
            let mut values = Vec::with_capacity(count as usize);
            for _ in 0..count {
                values.push(read_value(reader, elem_type)?);
            }
            Ok(MetadataValue::Array(values))
        }
        GgufValueType::Uint64 => {
            let v = reader
                .read_u64::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Uint64(v))
        }
        GgufValueType::Int64 => {
            let v = reader
                .read_i64::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Int64(v))
        }
        GgufValueType::Float64 => {
            let v = reader
                .read_f64::<LittleEndian>()
                .map_err(BonsaiError::MmapError)?;
            Ok(MetadataValue::Float64(v))
        }
    }
}

/// Read a key-value pair from the reader.
fn read_kv_pair<R: Read>(reader: &mut R) -> BonsaiResult<(String, MetadataValue)> {
    let key = read_gguf_string(reader)?;
    let value_type_id = reader
        .read_u32::<LittleEndian>()
        .map_err(BonsaiError::MmapError)?;
    let value_type = GgufValueType::from_id(value_type_id)?;
    let value = read_value(reader, value_type)?;
    Ok((key, value))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_string_bytes(s: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(s.len() as u64).to_le_bytes());
        bytes.extend_from_slice(s.as_bytes());
        bytes
    }

    fn make_kv_u32(key: &str, value: u32) -> Vec<u8> {
        let mut bytes = make_string_bytes(key);
        bytes.extend_from_slice(&(GgufValueType::Uint32 as u32).to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
        bytes
    }

    #[test]
    fn parse_single_u32_metadata() {
        let data = make_kv_u32("test.key", 42);
        let (store, _) = MetadataStore::parse(&data, 0, 1).expect("metadata parse should succeed");
        assert_eq!(store.len(), 1);
        assert_eq!(
            store.get_u32("test.key").expect("test.key should exist"),
            42
        );
    }

    #[test]
    fn missing_key_returns_error() {
        let store = MetadataStore::new();
        assert!(store.get_u32("nonexistent").is_err());
    }
}

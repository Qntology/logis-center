//! GGUF file header parsing.
//!
//! The GGUF header is the first structure in the file:
//! ```text
//! [magic: u32]           — 0x46554747 ("GGUF" in little-endian)
//! [version: u32]         — Format version (2 or 3)
//! [tensor_count: u64]    — Number of tensors in the file
//! [metadata_kv_count: u64] — Number of key-value metadata pairs
//! ```

use byteorder::{LittleEndian, ReadBytesExt};
use std::io::Read;

use crate::error::{BonsaiError, BonsaiResult};

/// GGUF magic number: "GGUF" in little-endian = 0x46554747.
const GGUF_MAGIC: u32 = 0x4655_4747;

/// Parsed GGUF file header.
#[derive(Debug, Clone)]
pub struct GgufHeader {
    /// GGUF format version (2 or 3).
    pub version: u32,
    /// Number of tensors stored in this file.
    pub tensor_count: u64,
    /// Number of key-value metadata pairs.
    pub metadata_kv_count: u64,
}

impl GgufHeader {
    /// Parse a GGUF header from a byte slice starting at the given offset.
    ///
    /// Returns the parsed header and the byte offset immediately after it.
    pub fn parse(data: &[u8], offset: usize) -> BonsaiResult<(Self, usize)> {
        let mut cursor = std::io::Cursor::new(data);
        cursor.set_position(offset as u64);

        let magic = cursor
            .read_u32::<LittleEndian>()
            .map_err(|_| BonsaiError::UnexpectedEof {
                offset: offset as u64,
            })?;

        if magic != GGUF_MAGIC {
            return Err(BonsaiError::InvalidMagic { magic });
        }

        let version =
            cursor
                .read_u32::<LittleEndian>()
                .map_err(|_| BonsaiError::UnexpectedEof {
                    offset: cursor.position(),
                })?;

        if version != 2 && version != 3 {
            return Err(BonsaiError::UnsupportedVersion { version });
        }

        let tensor_count =
            cursor
                .read_u64::<LittleEndian>()
                .map_err(|_| BonsaiError::UnexpectedEof {
                    offset: cursor.position(),
                })?;

        let metadata_kv_count =
            cursor
                .read_u64::<LittleEndian>()
                .map_err(|_| BonsaiError::UnexpectedEof {
                    offset: cursor.position(),
                })?;

        let header = GgufHeader {
            version,
            tensor_count,
            metadata_kv_count,
        };

        Ok((header, cursor.position() as usize))
    }

    /// Read a GGUF header from a reader (file, memory-mapped data, etc.).
    pub fn read_from<R: Read>(reader: &mut R) -> BonsaiResult<Self> {
        let magic = reader
            .read_u32::<LittleEndian>()
            .map_err(|_| BonsaiError::UnexpectedEof { offset: 0 })?;

        if magic != GGUF_MAGIC {
            return Err(BonsaiError::InvalidMagic { magic });
        }

        let version = reader
            .read_u32::<LittleEndian>()
            .map_err(|_| BonsaiError::UnexpectedEof { offset: 4 })?;

        if version != 2 && version != 3 {
            return Err(BonsaiError::UnsupportedVersion { version });
        }

        let tensor_count = reader
            .read_u64::<LittleEndian>()
            .map_err(|_| BonsaiError::UnexpectedEof { offset: 8 })?;

        let metadata_kv_count = reader
            .read_u64::<LittleEndian>()
            .map_err(|_| BonsaiError::UnexpectedEof { offset: 16 })?;

        Ok(GgufHeader {
            version,
            tensor_count,
            metadata_kv_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_header() {
        let mut data = Vec::new();
        // Magic: "GGUF" = 0x46554747
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version: 3
        data.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count: 291
        data.extend_from_slice(&291u64.to_le_bytes());
        // Metadata KV count: 25
        data.extend_from_slice(&25u64.to_le_bytes());

        let (header, offset) = GgufHeader::parse(&data, 0).expect("header parse should succeed");
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 291);
        assert_eq!(header.metadata_kv_count, 25);
        assert_eq!(offset, 24); // 4 + 4 + 8 + 8
    }

    #[test]
    fn reject_invalid_magic() {
        let mut data = Vec::new();
        data.extend_from_slice(&0xDEADBEEFu32.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let result = GgufHeader::parse(&data, 0);
        assert!(result.is_err());
        match result.unwrap_err() {
            BonsaiError::InvalidMagic { magic } => assert_eq!(magic, 0xDEADBEEF),
            other => panic!("expected InvalidMagic, got: {other}"),
        }
    }

    #[test]
    fn reject_unsupported_version() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        data.extend_from_slice(&99u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let result = GgufHeader::parse(&data, 0);
        assert!(result.is_err());
        match result.unwrap_err() {
            BonsaiError::UnsupportedVersion { version } => assert_eq!(version, 99),
            other => panic!("expected UnsupportedVersion, got: {other}"),
        }
    }
}

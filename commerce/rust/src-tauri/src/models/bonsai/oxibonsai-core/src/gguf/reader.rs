//! GGUF file reader — orchestrates header, metadata, and tensor parsing.
//!
//! Provides a high-level [`GgufFile`] that parses a complete GGUF file
//! from either a byte slice or a memory-mapped file.

use crate::error::{BonsaiError, BonsaiResult};
use crate::gguf::header::GgufHeader;
use crate::gguf::metadata::MetadataStore;
use crate::gguf::tensor_info::TensorStore;

/// Default alignment for tensor data in GGUF files (32 bytes).
const DEFAULT_ALIGNMENT: usize = 32;

/// A parsed GGUF file, containing header, metadata, tensor info, and a
/// reference to the raw tensor data region.
#[derive(Debug)]
pub struct GgufFile<'a> {
    /// Parsed header.
    pub header: GgufHeader,
    /// Key-value metadata store.
    pub metadata: MetadataStore,
    /// Tensor metadata (names, shapes, types, offsets).
    pub tensors: TensorStore,
    /// Byte offset where tensor data begins.
    pub data_offset: usize,
    /// Raw file data (for tensor loading).
    pub data: &'a [u8],
}

impl<'a> GgufFile<'a> {
    /// Parse a GGUF file from a byte slice.
    pub fn parse(data: &'a [u8]) -> BonsaiResult<Self> {
        // 1. Parse header
        let (header, offset) = GgufHeader::parse(data, 0)?;

        tracing::debug!(
            version = header.version,
            tensors = header.tensor_count,
            metadata = header.metadata_kv_count,
            "parsed GGUF header"
        );

        // 2. Parse metadata
        let (metadata, offset) = MetadataStore::parse(data, offset, header.metadata_kv_count)?;

        tracing::debug!(entries = metadata.len(), "parsed metadata");

        // 3. Parse tensor info
        let (tensors, offset) = TensorStore::parse(data, offset, header.tensor_count)?;

        tracing::debug!(count = tensors.len(), "parsed tensor info");

        // 4. Compute data offset (aligned to DEFAULT_ALIGNMENT)
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(DEFAULT_ALIGNMENT as u32) as usize;

        let data_offset = align_offset(offset, alignment);

        Ok(GgufFile {
            header,
            metadata,
            tensors,
            data_offset,
            data,
        })
    }

    /// Get raw tensor data bytes for a named tensor.
    pub fn tensor_data(&self, name: &str) -> BonsaiResult<&'a [u8]> {
        let info = self.tensors.require(name)?;
        let start = self.data_offset + info.offset as usize;
        let size = info.data_size() as usize;
        let end = start + size;

        if end > self.data.len() {
            return Err(BonsaiError::UnexpectedEof { offset: end as u64 });
        }

        Ok(&self.data[start..end])
    }
}

/// Align an offset to the given alignment boundary.
fn align_offset(offset: usize, alignment: usize) -> usize {
    (offset + alignment - 1) & !(alignment - 1)
}

/// Load a GGUF file from disk using memory-mapping (if the `mmap` feature is enabled).
#[cfg(feature = "mmap")]
pub fn mmap_gguf_file(path: &std::path::Path) -> BonsaiResult<memmap2::Mmap> {
    let file = std::fs::File::open(path)?;
    // SAFETY: We treat the mapped memory as read-only and the file should not be
    // modified while we hold the mapping. This is the standard usage pattern
    // for memory-mapped model files.
    let mmap = unsafe { memmap2::Mmap::map(&file)? };
    Ok(mmap)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_offset_works() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(31, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
    }
}

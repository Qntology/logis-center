//! GGUF v3 binary format parser.
//!
//! Sub-modules handle different parts of the GGUF file:
//! - [`header`] — Magic number, version, tensor/KV counts
//! - [`metadata`] — Typed key-value store
//! - [`tensor_info`] — Tensor names, shapes, types, and offsets
//! - [`types`] — GGUF data type enumerations

pub mod compat;
pub mod header;
pub mod metadata;
pub mod model_card;
pub mod reader;
pub mod streaming;
pub mod tensor_info;
pub mod types;
pub mod writer;

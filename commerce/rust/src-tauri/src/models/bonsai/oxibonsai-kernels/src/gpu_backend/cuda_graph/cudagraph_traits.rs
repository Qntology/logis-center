//! # CudaGraph - Trait Implementations
//!
//! This module contains trait implementations for `CudaGraph`.
//!
//! ## Implemented Traits
//!
//! - `Send`
//! - `Sync`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::cudagraph_type::CudaGraph;

unsafe impl Send for CudaGraph {}

unsafe impl Sync for CudaGraph {}

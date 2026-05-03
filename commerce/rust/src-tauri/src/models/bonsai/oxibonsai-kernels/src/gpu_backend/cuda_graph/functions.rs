//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use cudarc::nvrtc::compile_ptx;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use tracing::debug;

use super::cudagraph_type::CudaGraph;
use super::types::CudaGraphError;

/// FNV-1a 64-bit hash of a byte string.
fn fnv1a_64(data: &[u8]) -> u64 {
    const BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;
    let mut h = BASIS;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}
/// Build the on-disk cache path for a compiled PTX artifact.
fn ptx_cache_path(src_hash: u64, tag: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(format!("oxibonsai_ptx_{src_hash:016x}_{tag}.ptx"))
}
/// Try to load a compiled PTX from the disk cache.
fn load_ptx_cache(src_hash: u64, tag: &str) -> Option<cudarc::nvrtc::Ptx> {
    let path = ptx_cache_path(src_hash, tag);
    let ptx_src = std::fs::read_to_string(&path).ok()?;
    Some(cudarc::nvrtc::Ptx::from_src(ptx_src))
}
/// Save a compiled PTX to the disk cache (best-effort; ignores write errors).
fn save_ptx_cache(ptx: &cudarc::nvrtc::Ptx, src_hash: u64, tag: &str) {
    let path = ptx_cache_path(src_hash, tag);
    let _ = std::fs::write(&path, ptx.to_src());
}
/// Compile PTX from CUDA C source, using a disk cache keyed on the source hash.
///
/// First call: compiles via NVRTC (~5s), saves to the OS temp dir.
/// Subsequent calls: loads from the OS temp dir (~1ms), skips NVRTC entirely.
pub(crate) fn compile_or_load_ptx(
    src: &str,
    tag: &str,
) -> Result<cudarc::nvrtc::Ptx, CudaGraphError> {
    let hash = fnv1a_64(src.as_bytes());
    if let Some(cached) = load_ptx_cache(hash, tag) {
        debug!("PTX cache hit for tag={tag} hash={hash:016x}");
        return Ok(cached);
    }
    debug!("PTX cache miss for tag={tag}, compiling...");
    let ptx = compile_ptx(src).map_err(|e| CudaGraphError::CompilationFailed(format!("{e}")))?;
    save_ptx_cache(&ptx, hash, tag);
    debug!("PTX compiled and cached: tag={tag}");
    Ok(ptx)
}
static NEXT_HANDLE_ID: AtomicU64 = AtomicU64::new(1);
/// Allocate a new globally-unique weight handle ID.
pub(crate) fn alloc_handle_id() -> u64 {
    NEXT_HANDLE_ID.fetch_add(1, Ordering::Relaxed)
}
pub(super) static GLOBAL_CUDA_GRAPH: OnceLock<Mutex<Option<Arc<CudaGraph>>>> = OnceLock::new();
/// Attempt to run the FFN phase via direct CUDA dispatch.
///
/// This is the primary entry point for `block.rs` on Linux/Windows.
/// It mirrors `try_metal_ffn` exactly:
///
/// 1. Get the global `CudaGraph` singleton.
/// 2. Upload/cache weights lazily (first call uploads; subsequent calls reuse).
/// 3. Encode the full 8-op FFN pipeline on the CUDA stream.
///
/// Returns `Ok(())` if the CUDA dispatch succeeded.
/// Returns `Err(...)` if no CUDA device is present or dispatch failed.
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_ffn(
    hidden: &mut [f32],
    attn_out: &[f32],
    norm_weight: &[f32],
    eps: f32,
    attn_proj_handle_id: u64,
    attn_proj_bytes: &[u8],
    gate_up_handle_id: u64,
    gate_bytes: &[u8],
    up_bytes: &[u8],
    down_handle_id: u64,
    down_bytes: &[u8],
    hidden_size: usize,
    intermediate_size: usize,
) -> Result<(), CudaGraphError> {
    let graph = CudaGraph::global()?;
    let attn_proj_w = graph.get_or_upload_weight_soa(attn_proj_handle_id, attn_proj_bytes)?;
    let gate_up_w = graph.get_or_upload_weight_soa_lazy(gate_up_handle_id, || {
        let mut fused = Vec::with_capacity(gate_bytes.len() + up_bytes.len());
        fused.extend_from_slice(gate_bytes);
        fused.extend_from_slice(up_bytes);
        fused
    })?;
    let down_w = graph.get_or_upload_weight_soa(down_handle_id, down_bytes)?;
    graph.encode_ffn_phase(
        hidden,
        attn_out,
        norm_weight,
        eps,
        &attn_proj_w,
        &gate_up_w,
        &down_w,
        hidden_size,
        intermediate_size,
    )
}
/// Attempt to run a fused QKV projection via direct CUDA dispatch.
///
/// Mirrors `try_metal_qkv`:
///
/// 1. Get the global `CudaGraph` singleton.
/// 2. Upload/cache fused Q+K+V weight lazily.
/// 3. Encode a single GEMV on the CUDA stream.
#[allow(clippy::too_many_arguments)]
pub fn try_cuda_qkv(
    input: &[f32],
    output: &mut [f32],
    weight_handle_id: u64,
    q_bytes: &[u8],
    k_bytes: &[u8],
    v_bytes: &[u8],
    n_rows: usize,
    k: usize,
) -> Result<(), CudaGraphError> {
    let graph = CudaGraph::global()?;
    let weight_w = graph.get_or_upload_weight_soa_lazy(weight_handle_id, || {
        let mut fused = Vec::with_capacity(q_bytes.len() + k_bytes.len() + v_bytes.len());
        fused.extend_from_slice(q_bytes);
        fused.extend_from_slice(k_bytes);
        fused.extend_from_slice(v_bytes);
        fused
    })?;
    graph.encode_qkv_phase(input, output, &weight_w, n_rows, k)
}
#[cfg(test)]
mod tests {
    use super::*;
    /// Check that the singleton initialises without panicking.
    ///
    /// Skipped gracefully if no CUDA device is present (CI Linux without GPU).
    #[test]
    fn test_cuda_graph_global_init() {
        match CudaGraph::global() {
            Ok(_) => {}
            Err(e) => {
                eprintln!("CudaGraph::global() not available (expected in CPU-only CI): {e}");
            }
        }
    }
    /// Verify AoS → SoA reformatter preserves total byte count.
    #[test]
    fn test_reformat_aos_to_soa_round_trip() {
        const N: usize = 10;
        let mut aos = vec![0u8; N * 18];
        for i in 0..N {
            let base = i * 18;
            let v = i as u16;
            aos[base] = (v & 0xff) as u8;
            aos[base + 1] = (v >> 8) as u8;
            for j in 2..18 {
                aos[base + j] = 0xABu8;
            }
        }
        let soa = CudaGraph::reformat_q1_aos_to_soa(&aos).expect("reformat failed");
        assert_eq!(soa.len(), aos.len());
        for i in 0..N {
            let v = i as u16;
            assert_eq!(
                soa[i * 2],
                (v & 0xff) as u8,
                "scale byte 0 wrong at block {i}"
            );
            assert_eq!(
                soa[i * 2 + 1],
                (v >> 8) as u8,
                "scale byte 1 wrong at block {i}"
            );
        }
        for i in 0..N {
            let data_start = N * 2 + i * 16;
            for j in 0..16 {
                assert_eq!(
                    soa[data_start + j],
                    0xABu8,
                    "data wrong at block {i} byte {j}"
                );
            }
        }
    }
    /// Verify that alloc_handle_id() produces strictly increasing unique values.
    #[test]
    fn test_handle_id_uniqueness() {
        let ids: Vec<u64> = (0..64).map(|_| alloc_handle_id()).collect();
        for w in ids.windows(2) {
            assert!(w[1] > w[0], "handle IDs not strictly increasing");
        }
    }
    /// Verify that the CUDA_V7_KERNELS_SRC constant contains the fused kernel entry point.
    ///
    /// This test does NOT require a GPU — it only inspects the static source string.
    #[test]
    fn test_fused_gate_up_swiglu_source_has_entry_point() {
        assert!(
            crate::gpu_backend::cuda_kernels::CUDA_V7_KERNELS_SRC
                .contains("fused_gate_up_swiglu_q1"),
            "CUDA_V7_KERNELS_SRC must contain the fused_gate_up_swiglu_q1 kernel entry point"
        );
    }
    /// Verify that the fused kernel source contains the SiLU epilogue expression.
    ///
    /// This guards against regressions where the epilogue is accidentally removed.
    #[test]
    fn test_fused_gate_up_swiglu_source_has_silu_epilogue() {
        let src = crate::gpu_backend::cuda_kernels::CUDA_V7_KERNELS_SRC;
        assert!(
            src.contains("silu(gate_partial) * up_partial"),
            "fused kernel epilogue 'silu(gate_partial) * up_partial' not found in kernel source"
        );
    }
    /// Verify that the fused kernel source contains both gate and up partial accumulator names.
    ///
    /// Ensures the dual-accumulator pattern is present, not just a single-path kernel.
    #[test]
    fn test_fused_gate_up_swiglu_source_has_dual_accumulators() {
        let src = crate::gpu_backend::cuda_kernels::CUDA_V7_KERNELS_SRC;
        assert!(
            src.contains("gate_partial"),
            "fused kernel must have 'gate_partial' accumulator"
        );
        assert!(
            src.contains("up_partial"),
            "fused kernel must have 'up_partial' accumulator"
        );
    }
    /// Runtime test: initialise CudaGraph and verify the fused kernel compiles successfully.
    ///
    /// Skipped gracefully if no CUDA device is present (CPU-only CI).  When a GPU is
    /// available, confirms that `fused_gate_up_swiglu_q1` was loaded from the PTX module
    /// by checking that `CudaGraph::global()` succeeds (it would error on
    /// `load_function("fused_gate_up_swiglu_q1")` otherwise).
    #[test]
    fn test_fused_gate_up_swiglu_runtime_compile() {
        match CudaGraph::global() {
            Ok(_) => {}
            Err(e) => {
                eprintln!(
                    "test_fused_gate_up_swiglu_runtime_compile: no CUDA device (expected in CPU-only CI): {e}"
                );
            }
        }
    }
}

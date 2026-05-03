//! End-to-end ONNX → GGUF conversion for Ternary-Bonsai-1.7B-ONNX.
//!
//! Reads a MatMulNBits-quantized `.onnx` file (together with its `.onnx_data`
//! sidecar, when present) and emits a single OxiBonsai GGUF file whose
//! weight tensors are re-quantized to TQ2_0_g128.
//!
//! Usage:
//!
//! ```text
//! cargo run -p oxibonsai-model --example onnx_convert --release -- \
//!     <path/to/model.onnx> <output.gguf>
//! ```
//!
//! Optional third argument is the target quantisation format; it defaults
//! to `tq2_0_g128` (the only one supported today).

use std::path::Path;
use std::time::Instant;

use oxibonsai_model::convert_onnx_to_gguf;

fn main() {
    if let Err(e) = run() {
        eprintln!("onnx_convert error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let onnx_path = args.get(1).ok_or_else(|| {
        "usage: onnx_convert <path/to/model.onnx> <output.gguf> [quant]".to_string()
    })?;
    let out_path = args.get(2).ok_or_else(|| {
        "usage: onnx_convert <path/to/model.onnx> <output.gguf> [quant]".to_string()
    })?;
    let quant = args.get(3).map(String::as_str).unwrap_or("tq2_0_g128");

    let onnx_path = Path::new(onnx_path);
    let out_path = Path::new(out_path);

    println!("── OxiBonsai ONNX → GGUF converter ──");
    println!("  source : {}", onnx_path.display());
    println!("  target : {}", out_path.display());
    println!("  quant  : {quant}");

    let t0 = Instant::now();
    let stats = convert_onnx_to_gguf(onnx_path, out_path, quant).map_err(|e| format!("{e}"))?;
    let elapsed = t0.elapsed();

    println!();
    println!("── Conversion complete ──");
    println!("  total tensors       : {}", stats.n_tensors);
    println!("  TQ2_0_g128 weights  : {}", stats.n_ternary);
    println!("  FP32 norms          : {}", stats.n_fp32);
    println!(
        "  output file size    : {} bytes ({:.2} MiB)",
        stats.output_bytes,
        (stats.output_bytes as f64) / (1024.0 * 1024.0)
    );
    println!("  elapsed             : {:.2?}", elapsed);

    Ok(())
}

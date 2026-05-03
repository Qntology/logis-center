//! Standalone GEMV V7 Metal microbenchmark.
//!
//! Measures actual GPU kernel throughput for Q1_g128 GEMV at the matrix
//! sizes used in Bonsai-8B inference (single-token decode path).
//!
//! ```text
//! cargo run --release --features metal --example gemv_benchmark -p oxibonsai-kernels
//! ```

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!("This benchmark requires the `metal` feature.");
    eprintln!(
        "Run: cargo run --release --features metal \
         --example gemv_benchmark -p oxibonsai-kernels"
    );
    std::process::exit(1);
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    metal_bench::run();
}

// ═══════════════════════════════════════════════════════════════════════════
// Metal benchmark implementation
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(all(feature = "metal", target_os = "macos"))]
mod metal_bench {
    use metal::{
        Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLResourceOptions,
        MTLSize,
    };
    use oxibonsai_kernels::gpu_backend::kernel_sources;
    use std::ffi::c_void;
    use std::time::Instant;

    // ── Benchmark configuration ──────────────────────────────────────────

    /// A single GEMV size to benchmark.
    struct BenchCase {
        name: &'static str,
        n_rows: u32,
        k: u32,
    }

    /// Bonsai-8B layer shapes (single-token decode = GEMV).
    const CASES: &[BenchCase] = &[
        BenchCase {
            name: "QKV  (6144 x 4096)",
            n_rows: 6144,
            k: 4096,
        },
        BenchCase {
            name: "Gate+Up (28672 x 4096)",
            n_rows: 28672,
            k: 4096,
        },
        BenchCase {
            name: "Down (4096 x 14336)",
            n_rows: 4096,
            k: 14336,
        },
        BenchCase {
            name: "LM head (151669 x 4096)",
            n_rows: 151669,
            k: 4096,
        },
    ];

    const WARMUP_ITERS: u32 = 10;
    const BENCH_ITERS: u32 = 100;

    // ── Entry point ──────────────────────────────────────────────────────

    pub fn run() {
        let device = Device::system_default().expect("no Metal-capable GPU found");
        let queue = device.new_command_queue();

        eprintln!("Device : {}", device.name());
        eprintln!("Warmup : {} iters", WARMUP_ITERS);
        eprintln!("Measure: {} iters", BENCH_ITERS);
        eprintln!();

        // Compile only the V7 kernel (the hot-path production kernel).
        let pipeline = compile_v7_pipeline(&device);

        // Header
        println!(
            "{:<30} {:>10} {:>10} {:>12} {:>10}",
            "Case", "Wt MB", "µs/call", "Eff GB/s", "GFLOP/s"
        );
        println!("{}", "─".repeat(76));

        for case in CASES {
            bench_one(&device, &queue, &pipeline, case);
        }

        eprintln!();
        eprintln!("Eff GB/s = (weight + input + output bytes) / kernel time");
        eprintln!("GFLOP/s  = 2 * n_rows * k / kernel time  (1 FMA = 2 FLOP)");
    }

    // ── Pipeline compilation ─────────────────────────────────────────────

    fn compile_v7_pipeline(device: &Device) -> ComputePipelineState {
        let source = kernel_sources::MSL_GEMV_Q1_G128_V7;
        let options = CompileOptions::new();
        let library = device
            .new_library_with_source(source, &options)
            .expect("MSL V7 compilation failed");
        let func = library
            .get_function("gemv_q1_g128_v7", None)
            .expect("gemv_q1_g128_v7 function not found in compiled library");
        device
            .new_compute_pipeline_state_with_function(&func)
            .expect("failed to create V7 compute pipeline state")
    }

    // ── Per-case benchmark ───────────────────────────────────────────────

    fn bench_one(
        device: &Device,
        queue: &CommandQueue,
        pipeline: &ComputePipelineState,
        case: &BenchCase,
    ) {
        let n_rows = case.n_rows;
        let k = case.k;

        // Q1_g128 block geometry: 18 bytes per block, 128 elements per block
        let blocks_per_row = k as usize / 128;
        let total_blocks = n_rows as usize * blocks_per_row;
        let weight_bytes = total_blocks * 18;
        let input_bytes = k as usize * 4; // f32
        let output_bytes = n_rows as usize * 4; // f32

        // ── Allocate GPU buffers ────────────────────────────────────────

        let weight_data = synthetic_q1_weights(total_blocks);
        let weight_buf = new_buffer_from_slice(device, &weight_data);

        let input_data = vec![1.0_f32; k as usize];
        let input_buf = new_buffer_from_slice(device, &input_data);

        let output_buf =
            device.new_buffer(output_bytes as u64, MTLResourceOptions::StorageModeShared);

        // ── Warmup ──────────────────────────────────────────────────────

        for _ in 0..WARMUP_ITERS {
            dispatch_v7(
                queue,
                pipeline,
                &weight_buf,
                &input_buf,
                &output_buf,
                n_rows,
                k,
            );
        }

        // ── Timed iterations ────────────────────────────────────────────

        let start = Instant::now();
        for _ in 0..BENCH_ITERS {
            dispatch_v7(
                queue,
                pipeline,
                &weight_buf,
                &input_buf,
                &output_buf,
                n_rows,
                k,
            );
        }
        let elapsed = start.elapsed();
        let per_ns = elapsed.as_nanos() as f64 / f64::from(BENCH_ITERS);
        let per_us = per_ns / 1_000.0;

        // ── Metrics ─────────────────────────────────────────────────────

        let total_bytes_io = weight_bytes + input_bytes + output_bytes;
        let eff_gbps = total_bytes_io as f64 / per_ns; // bytes/ns = GB/s
        let gflops = (2.0 * n_rows as f64 * k as f64) / per_ns; // GFLOP/s
        let weight_mb = weight_bytes as f64 / (1024.0 * 1024.0);

        println!(
            "{:<30} {:>9.1} {:>10.1} {:>11.1} {:>10.1}",
            case.name, weight_mb, per_us, eff_gbps, gflops,
        );
    }

    // ── GPU dispatch ─────────────────────────────────────────────────────

    /// Encode + commit + wait a single V7 GEMV dispatch.
    fn dispatch_v7(
        queue: &CommandQueue,
        pipeline: &ComputePipelineState,
        weight_buf: &Buffer,
        input_buf: &Buffer,
        output_buf: &Buffer,
        n_rows: u32,
        k: u32,
    ) {
        let tg_count = u64::from(n_rows).div_ceil(8);
        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();

        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(weight_buf), 0);
        enc.set_buffer(1, Some(input_buf), 0);
        enc.set_buffer(2, Some(output_buf), 0);

        // Scalar uniforms via set_bytes (no buffer allocation)
        enc.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_rows as *const u32 as *const c_void,
        );
        enc.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &k as *const u32 as *const c_void,
        );

        enc.dispatch_thread_groups(
            MTLSize::new(tg_count, 1, 1),
            MTLSize::new(256, 1, 1), // 8 simdgroups × 32 lanes
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    /// Create a shared-mode Metal buffer from a byte/float slice.
    fn new_buffer_from_slice<T>(device: &Device, data: &[T]) -> Buffer {
        let byte_len = std::mem::size_of_val(data) as u64;
        device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            byte_len,
            MTLResourceOptions::StorageModeShared,
        )
    }

    /// Generate synthetic Q1_g128 weight data.
    ///
    /// Each block = 18 bytes: 2-byte fp16 scale + 16 bytes of sign bits.
    /// Uses alternating bit patterns (0xAA) so the kernel exercises real
    /// memory bandwidth rather than hitting zero-skip shortcuts.
    fn synthetic_q1_weights(total_blocks: usize) -> Vec<u8> {
        let total_bytes = total_blocks * 18;
        let mut data = vec![0u8; total_bytes];

        for b in 0..total_blocks {
            let off = b * 18;
            // fp16 0.5 = 0x3800 (little-endian: 0x00, 0x38)
            data[off] = 0x00;
            data[off + 1] = 0x38;
            // 128 sign bits: alternating 10101010...
            for i in 0..16 {
                data[off + 2 + i] = 0xAA;
            }
        }
        data
    }
}

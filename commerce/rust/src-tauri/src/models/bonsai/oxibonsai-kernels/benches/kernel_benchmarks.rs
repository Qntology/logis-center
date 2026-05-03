use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use half::f16;
use oxibonsai_core::{BlockTQ2_0_g128, QK_TQ2_0_G128};
use oxibonsai_kernels::dispatch::KernelDispatcher;
use oxibonsai_kernels::traits::TernaryKernel;
use std::hint::black_box;

fn make_ternary_blocks(n_rows: usize, k: usize) -> Vec<BlockTQ2_0_g128> {
    let blocks_per_row = k / QK_TQ2_0_G128;
    (0..n_rows * blocks_per_row)
        .map(|i| BlockTQ2_0_g128 {
            qs: std::array::from_fn(|j| ((i * 37 + j * 13) & 0xFF) as u8),
            d: f16::from_f32(0.5 + i as f32 * 0.001),
        })
        .collect()
}

fn bench_dequant_ternary(c: &mut Criterion) {
    let dispatcher = KernelDispatcher::auto_detect();
    let mut group = c.benchmark_group("dequant_ternary_g128");
    for (label, n_blocks) in [("1x128", 1usize), ("2048x128", 2048usize)] {
        let blocks = make_ternary_blocks(n_blocks, QK_TQ2_0_G128);
        let mut output = vec![0.0f32; n_blocks * QK_TQ2_0_G128];
        group.bench_with_input(BenchmarkId::new("kernel", label), &n_blocks, |b, _| {
            b.iter(|| {
                dispatcher
                    .dequant_ternary_g128(black_box(&blocks), black_box(&mut output))
                    .expect(
                        "dequant_ternary_g128 should succeed with valid blocks and output buffer",
                    );
            });
        });
    }
    group.finish();
}

fn bench_gemv_ternary(c: &mut Criterion) {
    let dispatcher = KernelDispatcher::auto_detect();
    let k = 4096usize;
    let mut group = c.benchmark_group("gemv_ternary_g128");
    for n_rows in [2048usize, 6144usize] {
        let blocks = make_ternary_blocks(n_rows, k);
        let input = vec![0.1f32; k];
        let mut output = vec![0.0f32; n_rows];
        group.bench_with_input(BenchmarkId::new("direct", n_rows), &n_rows, |b, _| {
            b.iter(|| {
                dispatcher
                    .gemv_ternary_g128(
                        black_box(&blocks),
                        black_box(&input),
                        black_box(&mut output),
                        n_rows,
                        k,
                    )
                    .expect("gemv_ternary_g128 should succeed with valid ternary blocks and matching dimensions");
            });
        });
    }
    group.finish();
}

fn bench_gemv_ternary_par(c: &mut Criterion) {
    let dispatcher = KernelDispatcher::auto_detect();
    let k = 4096usize;
    let mut group = c.benchmark_group("gemv_ternary_g128_par");
    for n_rows in [2048usize, 6144usize] {
        let blocks = make_ternary_blocks(n_rows, k);
        let input = vec![0.1f32; k];
        let mut output = vec![0.0f32; n_rows];
        group.bench_with_input(BenchmarkId::new("adaptive", n_rows), &n_rows, |b, _| {
            b.iter(|| {
                oxibonsai_kernels::gemv_adaptive_ternary(
                    black_box(&dispatcher),
                    black_box(&blocks),
                    black_box(&input),
                    black_box(&mut output),
                    n_rows,
                    k,
                )
                .expect("gemv_adaptive_ternary should succeed with valid ternary blocks and matching dimensions");
            });
        });
    }
    group.finish();
}

fn bench_gemm_ternary_par(c: &mut Criterion) {
    let dispatcher = KernelDispatcher::auto_detect();
    let k = 4096usize;
    let n_rows = 2048usize;
    let batch = 8usize;
    let mut group = c.benchmark_group("gemm_ternary_g128_par");
    let blocks = make_ternary_blocks(n_rows, k);
    let input = vec![0.1f32; batch * k];
    let mut output = vec![0.0f32; batch * n_rows];
    group.bench_function(
        BenchmarkId::new("adaptive", format!("batch{}_{}rows_{}k", batch, n_rows, k)),
        |b| {
            b.iter(|| {
                oxibonsai_kernels::gemm_adaptive_ternary(
                    black_box(&dispatcher),
                    black_box(&blocks),
                    black_box(&input),
                    black_box(&mut output),
                    batch,
                    n_rows,
                    k,
                )
                .expect("gemm_adaptive_ternary should succeed with valid ternary blocks and matching batch/matrix dimensions");
            });
        },
    );
    group.finish();
}

criterion_group!(
    benches,
    bench_dequant_ternary,
    bench_gemv_ternary,
    bench_gemv_ternary_par,
    bench_gemm_ternary_par
);
criterion_main!(benches);

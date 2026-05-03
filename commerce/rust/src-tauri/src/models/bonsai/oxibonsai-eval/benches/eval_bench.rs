//! Placeholder eval benchmark — real workload to be filled in by the eval
//! team.  The file exists so that the `[[bench]]` entry in `Cargo.toml`
//! resolves and the workspace can compile.

use criterion::{criterion_group, criterion_main, Criterion};

fn placeholder(c: &mut Criterion) {
    c.bench_function("oxibonsai_eval_placeholder", |b| {
        b.iter(|| {
            // Minimal no-op so the harness links.
            std::hint::black_box(1usize + 1usize);
        });
    });
}

criterion_group!(benches, placeholder);
criterion_main!(benches);

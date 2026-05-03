//! Placeholder tokenizer benchmark.  Exists so the `[[bench]]` entry in
//! `Cargo.toml` resolves and the workspace compiles.

use criterion::{criterion_group, criterion_main, Criterion};

fn placeholder(c: &mut Criterion) {
    c.bench_function("oxibonsai_tokenizer_placeholder", |b| {
        b.iter(|| {
            std::hint::black_box(1usize + 1usize);
        });
    });
}

criterion_group!(benches, placeholder);
criterion_main!(benches);

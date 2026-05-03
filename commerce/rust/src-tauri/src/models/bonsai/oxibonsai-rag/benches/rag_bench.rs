//! Micro-benchmarks for the OxiBonsai RAG pipeline.
//!
//! The suite exercises two hot paths:
//!
//! - **Indexing throughput** — embedding + vector-store insertion rate.
//! - **Query latency** — end-to-end cost of embedding a query and returning
//!   the top-k chunks.
//!
//! All benchmarks use [`IdentityEmbedder`] so that timings reflect the
//! OxiBonsai code paths rather than third-party embedding backends.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use oxibonsai_rag::chunker::ChunkConfig;
use oxibonsai_rag::embedding::IdentityEmbedder;
use oxibonsai_rag::retriever::{Retriever, RetrieverConfig};

const CORPUS: &[&str] = &[
    "Rust is a systems programming language focused on safety and speed.",
    "Python is a high-level interpreted language with dynamic typing.",
    "Go emphasises simplicity, concurrency primitives, and fast compilation.",
    "Haskell is a purely functional language with lazy evaluation semantics.",
    "C++ combines low-level hardware control with high-level abstractions.",
    "Zig aims to be a modern successor to C with explicit control over allocations.",
    "Elixir runs on the BEAM virtual machine and targets fault-tolerant systems.",
    "Kotlin compiles to JVM bytecode and interoperates smoothly with Java.",
];

fn bench_indexing(c: &mut Criterion) {
    let mut group = c.benchmark_group("indexing");
    group.throughput(Throughput::Elements(CORPUS.len() as u64));

    group.bench_function("identity_64_dim", |b| {
        b.iter(|| {
            let embedder = IdentityEmbedder::new(64).expect("valid dim");
            let mut retriever = Retriever::new(embedder, RetrieverConfig::default());
            let chunk_cfg = ChunkConfig::default();
            for doc in CORPUS {
                let _ = retriever.add_document(black_box(doc), &chunk_cfg);
            }
            black_box(retriever.chunk_count())
        });
    });

    group.finish();
}

fn bench_query_latency(c: &mut Criterion) {
    let embedder = IdentityEmbedder::new(64).expect("valid dim");
    let mut retriever = Retriever::new(
        embedder,
        RetrieverConfig::default()
            .with_top_k(3)
            .with_min_score(-1.0),
    );
    let chunk_cfg = ChunkConfig::default();
    for doc in CORPUS {
        retriever
            .add_document(doc, &chunk_cfg)
            .expect("seed document");
    }

    let mut group = c.benchmark_group("query_latency");
    group.bench_function("top3", |b| {
        b.iter(|| {
            let results = retriever
                .retrieve(black_box("systems programming language"))
                .expect("retrieve");
            black_box(results.len())
        });
    });
    group.finish();
}

criterion_group!(benches, bench_indexing, bench_query_latency);
criterion_main!(benches);

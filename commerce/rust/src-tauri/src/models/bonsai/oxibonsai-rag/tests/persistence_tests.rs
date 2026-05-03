//! JSON persistence round-trip tests.

use std::fs;
use std::path::PathBuf;

use oxibonsai_rag::chunker::{Chunk, ChunkConfig};
use oxibonsai_rag::embedding::IdentityEmbedder;
use oxibonsai_rag::metadata_filter::MetadataValue;
use oxibonsai_rag::persistence::{IndexSnapshot, SCHEMA_VERSION};
use oxibonsai_rag::retriever::{Retriever, RetrieverConfig};
use oxibonsai_rag::vector_store::VectorStore;
use oxibonsai_rag::{Distance, RagError};

/// Unique temp path for a test run.
fn tmp_path(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    std::env::temp_dir().join(format!("oxibonsai_rag_{tag}_{pid}_{nanos}.json"))
}

// ── 1. roundtrip_empty_store ─────────────────────────────────────────────────

#[test]
fn roundtrip_empty_store() {
    let store = VectorStore::new(4);
    let path = tmp_path("empty");
    store.save_json(&path).expect("save");
    let loaded = VectorStore::load_json(&path).expect("load");
    assert_eq!(loaded.len(), 0);
    assert_eq!(loaded.dim(), 4);
    fs::remove_file(&path).ok();
}

// ── 2. roundtrip_small_store ─────────────────────────────────────────────────

#[test]
fn roundtrip_small_store() {
    let mut store = VectorStore::new(3);
    for i in 0..5 {
        let v = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
        let chunk = Chunk::new(format!("chunk-{i}"), 0, i, 0);
        store.insert(v, chunk).expect("insert");
    }
    let path = tmp_path("small");
    store.save_json(&path).expect("save");
    let loaded = VectorStore::load_json(&path).expect("load");
    assert_eq!(loaded.len(), 5);
    fs::remove_file(&path).ok();
}

// ── 3. roundtrip_large_store ─────────────────────────────────────────────────

#[test]
fn roundtrip_large_store() {
    let mut store = VectorStore::new(8);
    for i in 0..256 {
        let v: Vec<f32> = (0..8).map(|j| ((i + j) as f32) * 0.01).collect();
        let chunk = Chunk::new(format!("c{i}"), 0, i, 0);
        store.insert(v, chunk).expect("insert");
    }
    let path = tmp_path("large");
    store.save_json(&path).expect("save");
    let loaded = VectorStore::load_json(&path).expect("load");
    assert_eq!(loaded.len(), 256);
    fs::remove_file(&path).ok();
}

// ── 4. roundtrip_with_metadata ───────────────────────────────────────────────

#[test]
fn roundtrip_with_metadata() {
    let mut store = VectorStore::new(2);
    let chunk = Chunk::new("rustacean".into(), 0, 0, 0)
        .with_metadata("lang", "rust")
        .with_metadata("year", 2026_i64);
    store.insert(vec![1.0, 0.0], chunk).expect("insert");
    let path = tmp_path("metadata");
    store.save_json(&path).expect("save");
    let loaded = VectorStore::load_json(&path).expect("load");
    let entry_chunks: Vec<_> = loaded
        .search(&[1.0, 0.0], 1)
        .into_iter()
        .map(|r| r.chunk)
        .collect();
    assert_eq!(entry_chunks.len(), 1);
    let md = &entry_chunks[0].metadata;
    assert_eq!(md.get("lang"), Some(&MetadataValue::from("rust")));
    fs::remove_file(&path).ok();
}

// ── 5. corrupted_json_rejected ───────────────────────────────────────────────

#[test]
fn corrupted_json_rejected() {
    let path = tmp_path("corrupt");
    fs::write(&path, "{not valid json").expect("write");
    let err = VectorStore::load_json(&path);
    assert!(matches!(err, Err(RagError::Persistence(_))));
    fs::remove_file(&path).ok();
}

// ── 6. version_mismatch_rejected ─────────────────────────────────────────────

#[test]
fn version_mismatch_rejected() {
    let snapshot = IndexSnapshot {
        schema_version: 9999,
        dim: 2,
        distance: Distance::Cosine,
        entries: Vec::new(),
        tfidf_state: None,
    };
    let path = tmp_path("version");
    fs::write(&path, serde_json::to_string(&snapshot).expect("serialize")).expect("write");
    let err = VectorStore::load_json(&path);
    assert!(matches!(err, Err(RagError::Persistence(_))));
    fs::remove_file(&path).ok();
}

// ── 7. retriever_roundtrip ───────────────────────────────────────────────────

#[test]
fn retriever_roundtrip_restores_state() {
    let embedder = IdentityEmbedder::new(16).expect("valid dim");
    let mut retriever = Retriever::new(embedder, RetrieverConfig::default());
    retriever
        .add_document(
            "Rust is a systems programming language",
            &ChunkConfig::default().with_min_chunk_size(1),
        )
        .expect("index");

    let path = tmp_path("retriever");
    retriever.save(&path).expect("save");

    let embedder2 = IdentityEmbedder::new(16).expect("valid dim");
    let restored = Retriever::load(embedder2, &path).expect("load");

    assert_eq!(restored.document_count(), retriever.document_count());
    assert_eq!(restored.chunk_count(), retriever.chunk_count());
    fs::remove_file(&path).ok();
}

// ── 8. retriever_dim_mismatch_rejected ───────────────────────────────────────

#[test]
fn retriever_dim_mismatch_rejected() {
    let embedder = IdentityEmbedder::new(8).expect("valid dim");
    let mut retriever = Retriever::new(embedder, RetrieverConfig::default());
    retriever
        .add_document(
            "hello world content",
            &ChunkConfig::default().with_min_chunk_size(1),
        )
        .expect("index");

    let path = tmp_path("retriever_dim");
    retriever.save(&path).expect("save");

    let mismatched = IdentityEmbedder::new(16).expect("valid dim");
    let err = Retriever::load(mismatched, &path);
    assert!(matches!(err, Err(RagError::DimensionMismatch { .. })));
    fs::remove_file(&path).ok();
}

// ── 9. schema_version_constant ───────────────────────────────────────────────

#[test]
fn schema_version_constant_is_one() {
    assert_eq!(SCHEMA_VERSION, 1);
}

// ── 10. missing_file_returns_io ──────────────────────────────────────────────

#[test]
fn missing_file_returns_io_error() {
    let path = tmp_path("nonexistent");
    let err = VectorStore::load_json(&path);
    assert!(matches!(err, Err(RagError::Io(_))));
}

// ── 11. distance_metric_preserved ────────────────────────────────────────────

#[test]
fn distance_metric_is_preserved() {
    let store = VectorStore::new_with_distance(3, Distance::Euclidean);
    let path = tmp_path("metric");
    store.save_json(&path).expect("save");
    let loaded = VectorStore::load_json(&path).expect("load");
    assert_eq!(loaded.distance(), Distance::Euclidean);
    fs::remove_file(&path).ok();
}

// ── 12. dim_mismatch_in_entries_rejected ─────────────────────────────────────

#[test]
fn dim_mismatch_in_stored_entries_rejected() {
    // Craft a snapshot whose entry vector length disagrees with `dim`.
    let body = serde_json::json!({
        "schema_version": 1,
        "dim": 4,
        "distance": "Cosine",
        "entries": [{
            "id": 0,
            "vector": [1.0, 0.0],
            "chunk": {
                "text": "x",
                "doc_id": 0,
                "chunk_idx": 0,
                "char_offset": 0
            }
        }]
    });
    let path = tmp_path("mismatch");
    fs::write(&path, body.to_string()).expect("write");
    let err = VectorStore::load_json(&path);
    assert!(matches!(err, Err(RagError::DimensionMismatch { .. })));
    fs::remove_file(&path).ok();
}

// ── 13. tempdir_cleanup ──────────────────────────────────────────────────────

#[test]
fn tempdir_write_then_remove() {
    let store = VectorStore::new(2);
    let path = tmp_path("cleanup");
    store.save_json(&path).expect("save");
    assert!(path.exists());
    fs::remove_file(&path).expect("remove");
    assert!(!path.exists());
}

// ── 14. snapshot_json_is_pretty ──────────────────────────────────────────────

#[test]
fn snapshot_json_is_human_readable() {
    let store = VectorStore::new(2);
    let path = tmp_path("pretty");
    store.save_json(&path).expect("save");
    let body = fs::read_to_string(&path).expect("read");
    assert!(body.contains('\n'), "expected multi-line pretty JSON");
    fs::remove_file(&path).ok();
}

// ── 15. load_matches_save ────────────────────────────────────────────────────

#[test]
fn saved_then_loaded_store_matches() {
    let mut store = VectorStore::new(3);
    store
        .insert(vec![1.0, 0.0, 0.0], Chunk::new("hello".into(), 0, 0, 0))
        .expect("insert");
    let path = tmp_path("match");
    store.save_json(&path).expect("save");
    let loaded = VectorStore::load_json(&path).expect("load");
    assert_eq!(store.len(), loaded.len());
    assert_eq!(store.dim(), loaded.dim());
    assert_eq!(store.distance(), loaded.distance());
    fs::remove_file(&path).ok();
}

// ── 16. save_overwrites_existing ─────────────────────────────────────────────

#[test]
fn save_overwrites_existing_file() {
    let store1 = VectorStore::new(1);
    let path = tmp_path("overwrite");
    store1.save_json(&path).expect("save1");

    let mut store2 = VectorStore::new(1);
    store2
        .insert(vec![1.0], Chunk::new("a".into(), 0, 0, 0))
        .expect("insert");
    store2.save_json(&path).expect("save2");

    let loaded = VectorStore::load_json(&path).expect("load");
    assert_eq!(loaded.len(), 1);
    fs::remove_file(&path).ok();
}

// ── 17. empty_metadata_round_trips ───────────────────────────────────────────

#[test]
fn empty_metadata_round_trips() {
    let mut store = VectorStore::new(2);
    store
        .insert(vec![0.0, 1.0], Chunk::new("x".into(), 0, 0, 0))
        .expect("insert");
    let path = tmp_path("empty_md");
    store.save_json(&path).expect("save");
    let loaded = VectorStore::load_json(&path).expect("load");
    let hits = loaded.search(&[0.0, 1.0], 1);
    assert!(hits[0].chunk.metadata.is_empty());
    fs::remove_file(&path).ok();
}

// ── 18. metric_default_is_cosine ─────────────────────────────────────────────

#[test]
fn older_snapshot_without_distance_defaults_to_cosine() {
    let body = serde_json::json!({
        "schema_version": 1,
        "dim": 2,
        "entries": []
    });
    let path = tmp_path("legacy");
    fs::write(&path, body.to_string()).expect("write");
    let loaded = VectorStore::load_json(&path).expect("load");
    assert_eq!(loaded.distance(), Distance::Cosine);
    fs::remove_file(&path).ok();
}

// ── 19. retriever_snapshot_schema_version ────────────────────────────────────

#[test]
fn retriever_snapshot_rejects_unknown_version() {
    // Write a RetrieverSnapshot-shaped JSON with a bogus schema_version.
    let body = serde_json::json!({
        "schema_version": 2000,
        "doc_count": 0,
        "store": {
            "schema_version": 1,
            "dim": 8,
            "distance": "Cosine",
            "entries": []
        }
    });
    let path = tmp_path("retriever_ver");
    fs::write(&path, body.to_string()).expect("write");
    let embedder = IdentityEmbedder::new(8).expect("dim");
    let err = Retriever::load(embedder, &path);
    assert!(matches!(err, Err(RagError::Persistence(_))));
    fs::remove_file(&path).ok();
}

// ── 20. chunk_text_preserved ─────────────────────────────────────────────────

#[test]
fn chunk_text_is_preserved() {
    let mut store = VectorStore::new(2);
    let text = "The quick brown fox jumps over the lazy dog";
    store
        .insert(vec![1.0, 0.0], Chunk::new(text.into(), 0, 0, 0))
        .expect("insert");
    let path = tmp_path("text");
    store.save_json(&path).expect("save");
    let loaded = VectorStore::load_json(&path).expect("load");
    let hits = loaded.search(&[1.0, 0.0], 1);
    assert_eq!(hits[0].chunk.text, text);
    fs::remove_file(&path).ok();
}

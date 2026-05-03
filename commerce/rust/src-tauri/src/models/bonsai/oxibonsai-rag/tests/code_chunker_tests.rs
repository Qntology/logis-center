//! Language-aware code chunker tests.

use oxibonsai_rag::{CodeChunker, Language};

// ── Language::from_extension ─────────────────────────────────────────────────

#[test]
fn language_from_extension_known() {
    assert_eq!(Language::from_extension("rs"), Language::Rust);
    assert_eq!(Language::from_extension("py"), Language::Python);
    assert_eq!(Language::from_extension("json"), Language::Json);
}

#[test]
fn language_from_extension_unknown_is_plain() {
    assert_eq!(Language::from_extension("txt"), Language::Plain);
    assert_eq!(Language::from_extension("md"), Language::Plain);
}

#[test]
fn language_default_is_plain() {
    assert_eq!(Language::default(), Language::Plain);
}

// ── Rust splitter ─────────────────────────────────────────────────────────────

#[test]
fn rust_splits_on_fn() {
    let src = "\nfn alpha() { 1 }\nfn beta() { 2 }\nfn gamma() { 3 }\n";
    let chunker = CodeChunker::new(Language::Rust).with_min_chunk_chars(1);
    let chunks = chunker.chunk(src, 0).expect("chunk");
    assert!(chunks.len() >= 3, "got {} chunks", chunks.len());
}

#[test]
fn rust_splits_on_impl() {
    let src = "\nstruct Foo;\nimpl Foo { fn bar() {} }\nimpl Clone for Foo { fn clone(&self) -> Self { Self } }\n";
    let chunker = CodeChunker::new(Language::Rust).with_min_chunk_chars(1);
    let chunks = chunker.chunk(src, 0).expect("chunk");
    assert!(chunks.len() >= 2, "got {} chunks", chunks.len());
}

#[test]
fn rust_splits_on_struct_and_enum() {
    let src = "\nstruct A;\nstruct B;\nenum E { X, Y }\n";
    let chunker = CodeChunker::new(Language::Rust).with_min_chunk_chars(1);
    let chunks = chunker.chunk(src, 0).expect("chunk");
    assert!(chunks.len() >= 3, "got {} chunks", chunks.len());
}

#[test]
fn rust_splits_on_mod() {
    let src = "\nmod inner {\n    pub fn hi() {}\n}\nmod outer {\n    pub fn hello() {}\n}\n";
    let chunker = CodeChunker::new(Language::Rust).with_min_chunk_chars(1);
    let chunks = chunker.chunk(src, 0).expect("chunk");
    assert!(chunks.len() >= 2, "got {} chunks", chunks.len());
}

#[test]
fn rust_splits_on_pub_fn() {
    let src = "\npub fn exported() {}\nfn private() {}\n";
    let chunker = CodeChunker::new(Language::Rust).with_min_chunk_chars(1);
    let chunks = chunker.chunk(src, 0).expect("chunk");
    assert!(chunks.len() >= 2, "got {} chunks", chunks.len());
}

// ── Python splitter ──────────────────────────────────────────────────────────

#[test]
fn python_splits_on_class_and_def() {
    let src = "\nclass Foo:\n    pass\n\ndef hello():\n    return 1\n\ndef bye():\n    return 2\n";
    let chunker = CodeChunker::new(Language::Python).with_min_chunk_chars(1);
    let chunks = chunker.chunk(src, 0).expect("chunk");
    assert!(chunks.len() >= 3, "got {} chunks", chunks.len());
}

#[test]
fn python_splits_multiple_defs() {
    let src = "\ndef a():\n    pass\n\ndef b():\n    pass\n\ndef c():\n    pass\n";
    let chunker = CodeChunker::new(Language::Python).with_min_chunk_chars(1);
    let chunks = chunker.chunk(src, 0).expect("chunk");
    assert!(chunks.len() >= 3);
}

#[test]
fn python_async_def_recognised() {
    let src = "\nasync def fetch():\n    return 1\n\nasync def fetch2():\n    return 2\n";
    let chunker = CodeChunker::new(Language::Python).with_min_chunk_chars(1);
    let chunks = chunker.chunk(src, 0).expect("chunk");
    assert!(chunks.len() >= 2);
}

// ── JSON splitter ────────────────────────────────────────────────────────────

#[test]
fn json_array_splits_by_element() {
    let src = r#"[1, 2, 3, 4, 5]"#;
    let chunker = CodeChunker::new(Language::Json).with_min_chunk_chars(1);
    let chunks = chunker.chunk(src, 0).expect("chunk");
    assert_eq!(chunks.len(), 5);
}

#[test]
fn json_object_splits_by_key() {
    let src = r#"{"a": 1, "b": 2, "c": 3}"#;
    let chunker = CodeChunker::new(Language::Json).with_min_chunk_chars(1);
    let chunks = chunker.chunk(src, 0).expect("chunk");
    assert_eq!(chunks.len(), 3);
}

#[test]
fn json_nested_array_of_objects() {
    let src = r#"[{"x": 1}, {"y": 2}, {"z": 3}]"#;
    let chunker = CodeChunker::new(Language::Json).with_min_chunk_chars(1);
    let chunks = chunker.chunk(src, 0).expect("chunk");
    assert_eq!(chunks.len(), 3);
}

#[test]
fn malformed_json_falls_back_to_splitter() {
    let src = r#"{not valid json at all"#;
    let chunker = CodeChunker::new(Language::Json).with_min_chunk_chars(1);
    let chunks = chunker.chunk(src, 0).expect("chunk");
    assert!(!chunks.is_empty());
}

// ── Plain (fallback) ─────────────────────────────────────────────────────────

#[test]
fn plain_single_chunk_for_short_text() {
    let src = "A short document with no structural markers.";
    let chunker = CodeChunker::new(Language::Plain)
        .with_fallback_window(1024)
        .with_min_chunk_chars(1);
    let chunks = chunker.chunk(src, 0).expect("chunk");
    assert_eq!(chunks.len(), 1);
}

#[test]
fn plain_multiple_chunks_for_long_text() {
    let src = "x".repeat(5_000);
    let chunker = CodeChunker::new(Language::Plain)
        .with_fallback_window(512)
        .with_min_chunk_chars(1);
    let chunks = chunker.chunk(&src, 0).expect("chunk");
    assert!(chunks.len() > 1);
}

// ── Edge cases ───────────────────────────────────────────────────────────────

#[test]
fn empty_text_yields_no_chunks() {
    let chunker = CodeChunker::new(Language::Rust);
    let chunks = chunker.chunk("", 0).expect("chunk");
    assert!(chunks.is_empty());
}

#[test]
fn whitespace_only_text_yields_no_chunks() {
    let chunker = CodeChunker::new(Language::Python);
    let chunks = chunker.chunk("   \n\t\n", 0).expect("chunk");
    assert!(chunks.is_empty());
}

#[test]
fn doc_id_is_propagated() {
    let src = "\nfn a() {}\nfn b() {}\n";
    let chunker = CodeChunker::new(Language::Rust).with_min_chunk_chars(1);
    let chunks = chunker.chunk(src, 42).expect("chunk");
    assert!(chunks.iter().all(|c| c.doc_id == 42));
}

#[test]
fn chunk_indices_are_monotonic() {
    let src = "\nfn a() {}\nfn b() {}\nfn c() {}\n";
    let chunker = CodeChunker::new(Language::Rust).with_min_chunk_chars(1);
    let chunks = chunker.chunk(src, 0).expect("chunk");
    for (expected_idx, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.chunk_idx, expected_idx);
    }
}

#[test]
fn min_chunk_chars_filters_short_bodies() {
    let src = "\nfn a() {}\nfn b() {}\n";
    let chunker = CodeChunker::new(Language::Rust).with_min_chunk_chars(100);
    let chunks = chunker.chunk(src, 0).expect("chunk");
    // With min=100, each split is too short — we fall through to the
    // single-chunk fallback path.
    assert_eq!(chunks.len(), 1);
}

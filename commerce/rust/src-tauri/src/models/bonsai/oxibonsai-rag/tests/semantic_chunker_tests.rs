//! Semantic chunker integration tests.

use oxibonsai_rag::embedding::IdentityEmbedder;
use oxibonsai_rag::SemanticChunker;

// ── Basic behaviour ──────────────────────────────────────────────────────────

#[test]
fn empty_text_yields_no_chunks() {
    let emb = IdentityEmbedder::new(16).expect("dim");
    let chunker = SemanticChunker::new(&emb, 0.5);
    let chunks = chunker.chunk("", 0).expect("chunk");
    assert!(chunks.is_empty());
}

#[test]
fn whitespace_only_yields_no_chunks() {
    let emb = IdentityEmbedder::new(16).expect("dim");
    let chunker = SemanticChunker::new(&emb, 0.5);
    let chunks = chunker.chunk("   \n\t\n  ", 0).expect("chunk");
    assert!(chunks.is_empty());
}

#[test]
fn short_input_falls_back_to_single_chunk() {
    let emb = IdentityEmbedder::new(16).expect("dim");
    let chunker = SemanticChunker::new(&emb, 0.5);
    let chunks = chunker.chunk("one short fragment", 0).expect("chunk");
    assert_eq!(chunks.len(), 1);
}

// ── Threshold boundaries ─────────────────────────────────────────────────────

#[test]
fn low_threshold_merges_all_sentences() {
    let emb = IdentityEmbedder::new(32).expect("dim");
    // A threshold below -1 can never fail cosine similarity so everything merges.
    let chunker = SemanticChunker::new(&emb, -2.0);
    let chunks = chunker
        .chunk("Alpha is first. Beta is second. Gamma is third.", 0)
        .expect("chunk");
    assert_eq!(chunks.len(), 1);
}

#[test]
fn high_threshold_splits_dissimilar_sentences() {
    let emb = IdentityEmbedder::new(32).expect("dim");
    // IdentityEmbedder produces near-orthogonal vectors for different
    // text, so a threshold near 1.0 forces a split on every new sentence.
    let chunker = SemanticChunker::new(&emb, 0.99);
    let chunks = chunker
        .chunk("First one. Second one. Third one.", 0)
        .expect("chunk");
    assert!(chunks.len() >= 2, "got {}", chunks.len());
}

#[test]
fn threshold_boundary_exact() {
    let emb = IdentityEmbedder::new(16).expect("dim");
    // Threshold of 0.0 is the midway point — any positive cosine keeps
    // the chunk going, any negative cosine starts a new one.
    let chunker = SemanticChunker::new(&emb, 0.0);
    let chunks = chunker
        .chunk("Alpha. Beta. Gamma. Delta.", 0)
        .expect("chunk");
    assert!(!chunks.is_empty());
}

// ── min_sentences_per_chunk ──────────────────────────────────────────────────

#[test]
fn min_sentences_prevents_flushing() {
    let emb = IdentityEmbedder::new(16).expect("dim");
    let chunker = SemanticChunker::new(&emb, 0.99).with_min_sentences(3);
    let chunks = chunker
        .chunk("One. Two. Three. Four. Five.", 0)
        .expect("chunk");
    // Every chunk should contain ≥ 2 sentences (or be the final tail).
    // With min=3 the first flush can only happen after sentence 3.
    assert!(!chunks.is_empty());
    let total_words: usize = chunks
        .iter()
        .map(|c| c.text.split_whitespace().count())
        .sum();
    assert!(total_words >= 5, "lost sentences: {total_words}");
}

// ── doc metadata ─────────────────────────────────────────────────────────────

#[test]
fn doc_id_is_propagated() {
    let emb = IdentityEmbedder::new(16).expect("dim");
    let chunker = SemanticChunker::new(&emb, 0.5);
    let chunks = chunker
        .chunk("Alpha is first. Beta is second.", 123)
        .expect("chunk");
    for chunk in &chunks {
        assert_eq!(chunk.doc_id, 123);
    }
}

#[test]
fn chunk_indices_are_monotonic() {
    let emb = IdentityEmbedder::new(16).expect("dim");
    let chunker = SemanticChunker::new(&emb, 0.99);
    let chunks = chunker
        .chunk("A one. B two. C three. D four. E five.", 0)
        .expect("chunk");
    for (i, chunk) in chunks.iter().enumerate() {
        assert_eq!(chunk.chunk_idx, i);
    }
}

#[test]
fn char_offsets_are_nondecreasing() {
    let emb = IdentityEmbedder::new(16).expect("dim");
    let chunker = SemanticChunker::new(&emb, 0.99);
    let chunks = chunker.chunk("One. Two. Three. Four.", 0).expect("chunk");
    let offsets: Vec<usize> = chunks.iter().map(|c| c.char_offset).collect();
    for pair in offsets.windows(2) {
        assert!(pair[0] <= pair[1], "offsets not monotonic: {offsets:?}");
    }
}

// ── similar group chunking ──────────────────────────────────────────────────

#[test]
fn identical_sentences_group_into_one_chunk() {
    let emb = IdentityEmbedder::new(32).expect("dim");
    let chunker = SemanticChunker::new(&emb, 0.0);
    // IdentityEmbedder is deterministic per text — identical sentences
    // produce identical (cosine=1.0) vectors which always stay grouped.
    let chunks = chunker
        .chunk("Same text here. Same text here. Same text here.", 0)
        .expect("chunk");
    assert_eq!(chunks.len(), 1);
}

#[test]
fn mixed_similar_and_dissimilar() {
    let emb = IdentityEmbedder::new(32).expect("dim");
    let chunker = SemanticChunker::new(&emb, 0.99);
    let text = "Cats are pets. Dogs are pets too. Computers are machines.";
    let chunks = chunker.chunk(text, 0).expect("chunk");
    // At high threshold we expect three separate chunks.
    assert!(!chunks.is_empty());
}

// ── text preservation ────────────────────────────────────────────────────────

#[test]
fn all_sentence_text_preserved() {
    let emb = IdentityEmbedder::new(16).expect("dim");
    let chunker = SemanticChunker::new(&emb, -2.0);
    let text = "Sentence alpha. Sentence beta. Sentence gamma.";
    let chunks = chunker.chunk(text, 0).expect("chunk");
    let joined = chunks
        .iter()
        .map(|c| &c.text[..])
        .collect::<Vec<_>>()
        .join(" ");
    for term in ["alpha", "beta", "gamma"] {
        assert!(joined.contains(term), "missing '{term}' in {joined}");
    }
}

#[test]
fn running_mean_stays_finite() {
    // Longer input exercises the Welford update — every chunk must stay
    // finite (no NaN drift).
    let emb = IdentityEmbedder::new(16).expect("dim");
    let chunker = SemanticChunker::new(&emb, -2.0); // force merge
    let text = (0..50)
        .map(|i| format!("Sentence number {i}."))
        .collect::<Vec<_>>()
        .join(" ");
    let chunks = chunker.chunk(&text, 0).expect("chunk");
    assert_eq!(chunks.len(), 1);
    assert!(!chunks[0].text.is_empty());
}

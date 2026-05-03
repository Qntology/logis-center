//! Tests for weight tying (tied embedding / LM head).

use oxibonsai_model::weight_tying::{TiedEmbedding, TyingError};

// ─── Construction tests ───────────────────────────────────────────────────────

#[test]
fn tied_embedding_new() {
    let te = TiedEmbedding::new(100, 64);
    assert_eq!(te.vocab_size, 100);
    assert_eq!(te.hidden_dim, 64);
    assert_eq!(te.weights.len(), 100 * 64);
}

#[test]
fn tied_embedding_embed_returns_correct_row() {
    let vocab_size = 4;
    let hidden_dim = 3;
    // weights = [0, 1, 2, | 3, 4, 5, | 6, 7, 8, | 9, 10, 11]
    let weights: Vec<f32> = (0..(vocab_size * hidden_dim)).map(|i| i as f32).collect();
    let te = TiedEmbedding::from_weights(weights, vocab_size, hidden_dim)
        .expect("from_weights should succeed");

    let row0 = te.embed(0).expect("embed(0) should succeed");
    assert_eq!(row0, vec![0.0, 1.0, 2.0]);

    let row2 = te.embed(2).expect("embed(2) should succeed");
    assert_eq!(row2, vec![6.0, 7.0, 8.0]);
}

#[test]
fn tied_embedding_embed_batch() {
    let vocab_size = 5;
    let hidden_dim = 2;
    let weights: Vec<f32> = (0..(vocab_size * hidden_dim)).map(|i| i as f32).collect();
    let te = TiedEmbedding::from_weights(weights, vocab_size, hidden_dim)
        .expect("from_weights should succeed");

    let batch = te
        .embed_batch(&[0, 2, 4])
        .expect("embed_batch should succeed");
    assert_eq!(batch.len(), 3);
    assert_eq!(batch[0], vec![0.0, 1.0]);
    assert_eq!(batch[1], vec![4.0, 5.0]);
    assert_eq!(batch[2], vec![8.0, 9.0]);
}

#[test]
fn tied_embedding_oob_error() {
    let te = TiedEmbedding::new(10, 8);
    let result = te.embed(10);
    assert!(
        matches!(
            result,
            Err(TyingError::TokenOutOfRange {
                id: 10,
                vocab_size: 10
            })
        ),
        "expected TokenOutOfRange error, got: {result:?}"
    );
}

// ─── Projection tests ─────────────────────────────────────────────────────────

#[test]
fn project_to_logits_shape() {
    let vocab_size = 32;
    let hidden_dim = 16;
    let te = TiedEmbedding::init_kaiming(vocab_size, hidden_dim, 7);
    let hidden = vec![1.0f32; hidden_dim];
    let logits = te
        .project_to_logits(&hidden)
        .expect("project should succeed");
    assert_eq!(logits.len(), vocab_size);
}

#[test]
fn project_batch_shape() {
    let vocab_size = 8;
    let hidden_dim = 4;
    let batch_size = 3;
    let te = TiedEmbedding::init_kaiming(vocab_size, hidden_dim, 42);
    let hidden = vec![0.5f32; batch_size * hidden_dim];
    let out = te
        .project_batch(&hidden, batch_size)
        .expect("project_batch should succeed");
    assert_eq!(out.len(), batch_size * vocab_size);
}

#[test]
fn embed_then_project_nonzero() {
    let vocab_size = 10;
    let hidden_dim = 8;
    // Use Kaiming init so weights are non-zero
    let te = TiedEmbedding::init_kaiming(vocab_size, hidden_dim, 123);
    let hidden = te.embed(3).expect("embed should succeed");
    let logits = te
        .project_to_logits(&hidden)
        .expect("project should succeed");
    // At minimum, logit[3] should be the squared norm of that row — positive
    let norm_sq: f32 = hidden.iter().map(|&x| x * x).sum();
    assert!(
        logits[3] > 0.0 || norm_sq == 0.0,
        "logit for embedded token should be positive (norm_sq={norm_sq})"
    );
    // Output should be non-trivially non-zero
    let any_nonzero = logits.iter().any(|&x| x.abs() > 1e-7);
    assert!(any_nonzero, "at least one logit should be non-zero");
}

// ─── Memory savings ────────────────────────────────────────────────────────────

#[test]
fn memory_saved_positive() {
    let te = TiedEmbedding::new(50257, 768);
    let saved = te.memory_saved_bytes();
    assert!(saved > 0, "memory saved must be positive");
    // Should be vocab_size * hidden_dim * 4 bytes
    assert_eq!(saved, 50257 * 768 * 4);
}

// ─── Kaiming init ─────────────────────────────────────────────────────────────

#[test]
fn init_kaiming_correct_size() {
    let vocab_size = 100;
    let hidden_dim = 32;
    let te = TiedEmbedding::init_kaiming(vocab_size, hidden_dim, 0);
    assert_eq!(
        te.weights.len(),
        vocab_size * hidden_dim,
        "Kaiming init weights size mismatch"
    );
}

// ─── from_weights error ───────────────────────────────────────────────────────

#[test]
fn from_weights_shape_error() {
    let result = TiedEmbedding::from_weights(vec![1.0f32; 10], 4, 4); // expects 16
    assert!(
        matches!(
            result,
            Err(TyingError::ShapeMismatch {
                expected: 16,
                actual: 10
            })
        ),
        "expected ShapeMismatch error, got: {result:?}"
    );
}

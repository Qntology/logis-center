//! Integration tests for the BPE trainer module.
//!
//! Covers: TrainerConfig, SymbolPair, BpeTrainer, TrainedTokenizer, and
//! error paths.  All 15 tests are deterministic — no random data.

use oxibonsai_tokenizer::{
    trainer::{BpeTrainer, SymbolPair, TrainerConfig, TrainerError},
    OxiTokenizer,
};

// ── 1. TrainerConfig defaults ─────────────────────────────────────────────────

#[test]
fn trainer_config_default() {
    let cfg = TrainerConfig::default();
    assert_eq!(cfg.vocab_size, 1000);
    assert!(cfg.byte_level);
    assert!(cfg.add_special_tokens);
    assert_eq!(cfg.min_frequency, 2);
    assert!(cfg.progress_interval.is_none());
}

// ── 2. TrainerConfig builder methods ─────────────────────────────────────────

#[test]
fn trainer_config_builder() {
    let cfg = TrainerConfig::new(512)
        .with_min_frequency(5)
        .with_special_tokens(false);

    assert_eq!(cfg.vocab_size, 512);
    assert_eq!(cfg.min_frequency, 5);
    assert!(!cfg.add_special_tokens);
}

// ── 3. SymbolPair hash / equality ─────────────────────────────────────────────

#[test]
fn symbol_pair_hash_eq() {
    use std::collections::HashSet;

    let p1 = SymbolPair::new(10, 20);
    let p2 = SymbolPair::new(10, 20);
    let p3 = SymbolPair::new(20, 10);

    assert_eq!(p1, p2);
    assert_ne!(p1, p3);

    let mut set = HashSet::new();
    set.insert(p1.clone());
    set.insert(p2.clone()); // duplicate — set size must stay 1
    assert_eq!(set.len(), 1);
    assert!(!set.contains(&p3));
}

// ── 4. BpeTrainer::new does not panic ────────────────────────────────────────

#[test]
fn bpe_trainer_new() {
    let cfg = TrainerConfig::new(300);
    let _trainer = BpeTrainer::new(cfg);
    // Should reach here without panic.
}

// ── 5. Train on empty corpus returns EmptyCorpus error ───────────────────────

#[test]
fn train_empty_corpus_error() {
    let mut trainer = BpeTrainer::new(TrainerConfig::new(300));
    let result = trainer.train(&[]);
    match result {
        Err(TrainerError::EmptyCorpus) => {}
        other => panic!("expected EmptyCorpus, got {other:?}"),
    }
}

// ── 6. Train on a small corpus succeeds ──────────────────────────────────────

#[test]
fn train_small_corpus() {
    let mut trainer = BpeTrainer::new(TrainerConfig::new(300));
    let corpus = ["hello world", "hello there"];
    let trained = trainer
        .train(&corpus)
        .expect("training on small corpus should succeed");
    // Must have at least the 256 base tokens.
    assert!(trained.vocab_size() >= 256);
}

// ── 7. Trained vocab_size does not exceed config.vocab_size ──────────────────

#[test]
fn train_vocab_size_respected() {
    let target = 280usize;
    let mut trainer = BpeTrainer::new(TrainerConfig::new(target));
    // Use a moderately rich corpus so the trainer has merges to perform.
    let corpus = [
        "the cat sat on the mat",
        "the cat sat on the hat",
        "the cat in the hat",
        "a cat a mat a hat",
    ];
    let trained = trainer.train(&corpus).expect("training should succeed");
    assert!(
        trained.vocab_size() <= target,
        "vocab_size {} exceeded target {}",
        trained.vocab_size(),
        target
    );
}

// ── 8. Merges are returned in training (application) order ───────────────────

#[test]
fn train_merges_ordered() {
    let corpus = ["aaaa bbbb aaaa bbbb aaaa", "cccc dddd cccc"];
    let mut trainer = BpeTrainer::new(TrainerConfig::new(300).with_min_frequency(2));
    let trained = trainer.train(&corpus).expect("training should succeed");

    // IDs of merged tokens must be strictly increasing (each merge gets a
    // fresh, higher ID than all previous tokens).
    let mut last_id = 0u32;
    for rule in &trained.merges {
        assert!(
            rule.merged > last_id || last_id == 0,
            "merge IDs must be increasing; got {} after {}",
            rule.merged,
            last_id
        );
        last_id = rule.merged;
    }
}

// ── 9. TrainingStats::summary returns a non-empty string ─────────────────────

#[test]
fn train_stats_summary_nonempty() {
    let corpus = ["the quick brown fox jumps over the lazy dog"];
    let mut trainer = BpeTrainer::new(TrainerConfig::new(300));
    let trained = trainer.train(&corpus).expect("training should succeed");
    let summary = trained.stats.summary();
    assert!(!summary.is_empty(), "summary should not be empty");
    // Should contain at least the token-count report.
    assert!(
        summary.contains("BPE training"),
        "summary format unexpected: {summary}"
    );
}

// ── 10. All 256 base byte tokens are in the trained vocabulary ────────────────

#[test]
fn trained_vocab_has_base_tokens() {
    let mut trainer = BpeTrainer::new(TrainerConfig::new(300).with_special_tokens(false));
    let corpus = ["hello world"];
    let trained = trainer.train(&corpus).expect("training should succeed");

    // IDs 0–255 must be present.
    for id in 0u32..256 {
        assert!(
            trained.vocab.contains_key(&id),
            "base token id {id} missing from trained vocab"
        );
    }
}

// ── 11. At least one merge is learned on sufficiently repeated data ───────────

#[test]
fn trained_merges_nonempty_with_enough_data() {
    // Repeat "hello" many times so "he", "el", "ll", "lo" pairs exceed
    // min_frequency=2 easily.
    let repeated = "hello ".repeat(20);
    let corpus_str: Vec<&str> = vec![repeated.as_str()];
    let mut trainer = BpeTrainer::new(TrainerConfig::new(300).with_min_frequency(2));
    let trained = trainer.train(&corpus_str).expect("training should succeed");
    assert!(
        !trained.merges.is_empty(),
        "expected at least one merge on repeated data"
    );
}

// ── 12. to_oxi_tokenizer creates an OxiTokenizer without panic ───────────────

#[test]
fn to_oxi_tokenizer_creates() {
    let corpus = ["hello world", "world hello", "hello there"];
    let mut trainer = BpeTrainer::new(TrainerConfig::new(300));
    let trained = trainer.train(&corpus).expect("training should succeed");

    let tok: OxiTokenizer = trained.to_oxi_tokenizer();
    // Must have a reasonable vocabulary.
    assert!(tok.vocab_size() >= 256, "OxiTokenizer vocab too small");
}

// ── 13. merges_to_text produces one line per merge rule ──────────────────────

#[test]
fn merges_to_text_one_per_line() {
    let repeated = "abcabc ".repeat(10);
    let corpus_str: Vec<&str> = vec![repeated.as_str()];
    let mut trainer = BpeTrainer::new(TrainerConfig::new(300).with_min_frequency(2));
    let trained = trainer.train(&corpus_str).expect("training should succeed");

    let text = trained.merges_to_text();
    let num_rules = trained.merges.len();

    if num_rules == 0 {
        assert!(text.is_empty(), "no merges → text should be empty");
    } else {
        // Each rule occupies exactly one line terminated by '\n'.
        let line_count = text.lines().count();
        assert_eq!(
            line_count, num_rules,
            "expected {num_rules} lines in merges text, got {line_count}"
        );
    }
}

// ── 14. More repetition in corpus → more merge rules learned ─────────────────

#[test]
fn train_repeated_text_more_merges() {
    // Corpus A: low repetition.
    let small_corpus = ["hello world", "world hello"];
    let mut trainer_a = BpeTrainer::new(TrainerConfig::new(300).with_min_frequency(2));
    let trained_a = trainer_a
        .train(&small_corpus)
        .expect("training A should succeed");

    // Corpus B: high repetition of the same content.
    let large_text = "hello world ".repeat(30);
    let large_corpus: Vec<&str> = vec![large_text.as_str()];
    let mut trainer_b = BpeTrainer::new(TrainerConfig::new(300).with_min_frequency(2));
    let trained_b = trainer_b
        .train(&large_corpus)
        .expect("training B should succeed");

    // B should have at least as many merges as A because pairs have higher
    // frequency and are thus more likely to exceed min_frequency.
    assert!(
        trained_b.merges.len() >= trained_a.merges.len(),
        "expected more merges with more repetition: a={} b={}",
        trained_a.merges.len(),
        trained_b.merges.len()
    );
}

// ── 15. TrainingStats.corpus_size_chars is positive ──────────────────────────

#[test]
fn train_stats_corpus_size() {
    let corpus = ["some text here", "more text there"];
    let mut trainer = BpeTrainer::new(TrainerConfig::new(300));
    let trained = trainer.train(&corpus).expect("training should succeed");
    assert!(
        trained.stats.corpus_size_chars > 0,
        "corpus_size_chars must be positive"
    );
    // The reported size must be at least the length of the corpus strings.
    let expected_chars: usize = corpus.iter().map(|s| s.len()).sum();
    assert_eq!(trained.stats.corpus_size_chars, expected_chars);
}

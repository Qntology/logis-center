use oxibonsai_rag::{
    ChunkStrategy, ChunkerRegistry, MarkdownChunker, RecursiveCharSplitter, RichChunk,
    SentenceChunker, SlidingWindowChunker,
};

// ── 1. rich_chunk_len ─────────────────────────────────────────────────────────

#[test]
fn rich_chunk_len() {
    let chunk = RichChunk::new("hello world".to_string(), 0, 11, 0);
    assert_eq!(chunk.len(), 11);
    assert!(!chunk.is_empty());
}

// ── 2. rich_chunk_word_count ──────────────────────────────────────────────────

#[test]
fn rich_chunk_word_count() {
    let chunk = RichChunk::new("the quick brown fox".to_string(), 0, 19, 0);
    assert_eq!(chunk.word_count(), 4);
}

// ── 3. sentence_chunker_basic ─────────────────────────────────────────────────

#[test]
fn sentence_chunker_basic() {
    let chunker = SentenceChunker::new(500);
    let text = "This is sentence one. This is sentence two. This is sentence three.";
    let chunks = chunker.chunk(text);
    // With max_chars=500, all sentences fit in one chunk
    assert!(!chunks.is_empty());
    // chunk_index should start at 0
    assert_eq!(chunks[0].chunk_index, 0);
}

// ── 4. sentence_chunker_max_chars ─────────────────────────────────────────────

#[test]
fn sentence_chunker_max_chars() {
    let max = 30usize;
    let chunker = SentenceChunker::new(max);
    let text = "Hello world. This is a longer sentence here. Another one follows. And yet another sentence comes now.";
    let chunks = chunker.chunk(text);
    assert!(!chunks.is_empty());
    for chunk in &chunks {
        // Each chunk should be <= max_chars (or a single oversized sentence)
        assert!(
            chunk.len() <= max || chunk.word_count() == 1 || {
                // It's a single sentence that is longer than max — allowed
                !chunk.text.contains(". ")
            },
            "Chunk '{}' (len={}) exceeds max_chars={}",
            chunk.text,
            chunk.len(),
            max
        );
    }
}

// ── 5. sentence_chunker_overlap ───────────────────────────────────────────────

#[test]
fn sentence_chunker_overlap() {
    let chunker = SentenceChunker::new(60).with_overlap(1);
    // Each sentence is ~20 chars; max_chars=60 fits ~3; overlap=1
    let text = "Sentence one here. Sentence two here. Sentence three here. Sentence four here. Sentence five here.";
    let chunks = chunker.chunk(text);
    // With overlap, we should get more chunks than without
    let no_overlap = SentenceChunker::new(60).chunk(text);
    // With overlap we get >= as many chunks as without
    assert!(chunks.len() >= no_overlap.len());
}

// ── 6. recursive_splitter_short_text ─────────────────────────────────────────

#[test]
fn recursive_splitter_short_text() {
    let splitter = RecursiveCharSplitter::new(1000);
    let text = "short text";
    let chunks = splitter.chunk(text);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, text);
}

// ── 7. recursive_splitter_long_text ──────────────────────────────────────────

#[test]
fn recursive_splitter_long_text() {
    let splitter = RecursiveCharSplitter::new(20);
    let text = "This is a longer piece of text that should be split into multiple chunks by the recursive splitter.";
    let chunks = splitter.chunk(text);
    assert!(
        chunks.len() > 1,
        "Expected multiple chunks, got {}",
        chunks.len()
    );
}

// ── 8. recursive_splitter_no_chunk_exceeds_max ───────────────────────────────

#[test]
fn recursive_splitter_no_chunk_exceeds_max() {
    let max = 25usize;
    let splitter = RecursiveCharSplitter::new(max);
    let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. \
                Ut enim ad minim veniam, quis nostrud exercitation ullamco.";
    let chunks = splitter.chunk(text);
    assert!(!chunks.is_empty());
    for chunk in &chunks {
        assert!(
            chunk.len() <= max,
            "Chunk '{}' (len={}) exceeds max={}",
            chunk.text,
            chunk.len(),
            max
        );
    }
}

// ── 9. sliding_window_basic ───────────────────────────────────────────────────

#[test]
fn sliding_window_basic() {
    let chunker = SlidingWindowChunker::non_overlapping(5);
    let text = "abcdefghij"; // 10 chars
    let chunks = chunker.chunk(text);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].text, "abcde");
    assert_eq!(chunks[1].text, "fghij");
}

// ── 10. sliding_window_overlap ────────────────────────────────────────────────

#[test]
fn sliding_window_overlap() {
    let chunker = SlidingWindowChunker::with_50pct_overlap(4);
    let text = "abcdefgh"; // 8 chars, window=4, step=2
    let chunks = chunker.chunk(text);
    assert!(chunks.len() >= 2);
    // Adjacent chunks should share characters
    let c0 = &chunks[0].text;
    let c1 = &chunks[1].text;
    // c0 ends at char 4, c1 starts at char 2 → overlap is "cd"
    let overlap: String = c0
        .chars()
        .rev()
        .take(2)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    let c1_start: String = c1.chars().take(2).collect();
    assert_eq!(
        overlap, c1_start,
        "expected overlap between '{}' and '{}'",
        c0, c1
    );
}

// ── 11. sliding_window_nonoverlapping ─────────────────────────────────────────

#[test]
fn sliding_window_nonoverlapping() {
    let chunker = SlidingWindowChunker::non_overlapping(3);
    let text = "abcdefghi"; // 9 chars
    let chunks = chunker.chunk(text);
    assert_eq!(chunks.len(), 3);
    // No shared characters between adjacent chunks
    for i in 0..chunks.len() - 1 {
        let end_of_current = chunks[i].char_end;
        let start_of_next = chunks[i + 1].char_start;
        assert_eq!(
            end_of_current,
            start_of_next,
            "chunks {} and {} should be contiguous",
            i,
            i + 1
        );
    }
}

// ── 12. markdown_chunker_splits_on_headers ────────────────────────────────────

#[test]
fn markdown_chunker_splits_on_headers() {
    let chunker = MarkdownChunker::new(1000);
    let text = "# Title\n\nIntro paragraph.\n\n## Section One\n\nContent of section one.\n\n## Section Two\n\nContent of section two.";
    let chunks = chunker.chunk(text);
    assert!(
        chunks.len() >= 2,
        "Expected at least 2 chunks from markdown headers, got {}",
        chunks.len()
    );
}

// ── 13. markdown_chunker_no_headers ──────────────────────────────────────────

#[test]
fn markdown_chunker_no_headers() {
    let max = 30usize;
    let chunker = MarkdownChunker::new(max);
    // No markdown headers — should still split on size
    let text =
        "This is a long paragraph without any headers in it. It should still be chunked by size.";
    let chunks = chunker.chunk(text);
    assert!(!chunks.is_empty());
    for chunk in &chunks {
        assert!(
            chunk.len() <= max,
            "chunk len {} exceeds max {}",
            chunk.len(),
            max
        );
    }
}

// ── 14. chunker_registry_default ─────────────────────────────────────────────

#[test]
fn chunker_registry_default() {
    let registry = ChunkerRegistry::default_registry();
    let strategies = registry.available_strategies();
    assert!(
        strategies.contains(&"sentence"),
        "missing 'sentence' strategy"
    );
    assert!(
        strategies.contains(&"recursive"),
        "missing 'recursive' strategy"
    );
    assert!(
        strategies.contains(&"sliding_window"),
        "missing 'sliding_window' strategy"
    );
    assert!(
        strategies.contains(&"markdown"),
        "missing 'markdown' strategy"
    );
    assert_eq!(strategies.len(), 4);
}

// ── 15. chunker_registry_chunk ────────────────────────────────────────────────

#[test]
fn chunker_registry_chunk() {
    let registry = ChunkerRegistry::default_registry();
    let text = "Hello world. This is a test sentence. Another sentence follows here.";

    // Sentence strategy should return chunks
    let result = registry.chunk("sentence", text);
    assert!(result.is_some(), "sentence strategy should be found");
    assert!(!result.expect("chunker should return result").is_empty());

    // Unknown strategy returns None
    let none = registry.chunk("nonexistent_strategy", text);
    assert!(none.is_none());
}

// ── Additional edge-case tests ────────────────────────────────────────────────

#[test]
fn rich_chunk_is_empty_for_empty_text() {
    let chunk = RichChunk::new(String::new(), 0, 0, 0);
    assert!(chunk.is_empty());
    assert_eq!(chunk.len(), 0);
    assert_eq!(chunk.word_count(), 0);
}

#[test]
fn sliding_window_empty_text() {
    let chunker = SlidingWindowChunker::non_overlapping(10);
    assert!(chunker.chunk("").is_empty());
}

#[test]
fn sentence_chunker_empty_text() {
    let chunker = SentenceChunker::new(200);
    assert!(chunker.chunk("").is_empty());
}

#[test]
fn recursive_splitter_empty_text() {
    let splitter = RecursiveCharSplitter::new(100);
    assert!(splitter.chunk("").is_empty());
}

#[test]
fn chunker_registry_registers_custom() {
    struct MyChunker;
    impl ChunkStrategy for MyChunker {
        fn name(&self) -> &'static str {
            "custom"
        }
        fn chunk(&self, text: &str) -> Vec<RichChunk> {
            if text.is_empty() {
                return Vec::new();
            }
            vec![RichChunk::new(text.to_string(), 0, text.chars().count(), 0)]
        }
    }

    let mut registry = ChunkerRegistry::new();
    registry.register(Box::new(MyChunker));

    let result = registry.chunk("custom", "test");
    assert!(result.is_some());
    let chunks = result.expect("custom strategy should succeed");
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, "test");
}

use oxibonsai_model::chunked_prefill::{
    create_prefill_chunks, peak_memory_estimate, ChunkedPrefillConfig, PrefillAction, PrefillChunk,
    PrefillPriority, PrefillScheduler,
};

#[test]
fn config_default() {
    let cfg = ChunkedPrefillConfig::default();
    assert_eq!(cfg.chunk_size, 512);
}

#[test]
fn create_chunks_short_prompt() {
    let tokens: Vec<u32> = (0..100).collect();
    let cfg = ChunkedPrefillConfig::new(512);
    let chunks = create_prefill_chunks(&tokens, &cfg);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].tokens.len(), 100);
    assert!(chunks[0].is_last);
}

#[test]
fn create_chunks_exact_boundary() {
    let tokens: Vec<u32> = (0..1024).collect();
    let cfg = ChunkedPrefillConfig::new(512);
    let chunks = create_prefill_chunks(&tokens, &cfg);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].tokens.len(), 512);
    assert_eq!(chunks[1].tokens.len(), 512);
}

#[test]
fn create_chunks_with_remainder() {
    let tokens: Vec<u32> = (0..1000).collect();
    let cfg = ChunkedPrefillConfig::new(512);
    let chunks = create_prefill_chunks(&tokens, &cfg);
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0].tokens.len(), 512);
    assert_eq!(chunks[1].tokens.len(), 488);
}

#[test]
fn create_chunks_positions_correct() {
    let tokens: Vec<u32> = (0..1500).collect();
    let cfg = ChunkedPrefillConfig::new(512);
    let chunks = create_prefill_chunks(&tokens, &cfg);
    assert_eq!(chunks[0].start_pos, 0);
    assert_eq!(chunks[0].end_pos, 512);
    assert_eq!(chunks[1].start_pos, 512);
    assert_eq!(chunks[1].end_pos, 1024);
    assert_eq!(chunks[2].start_pos, 1024);
    assert_eq!(chunks[2].end_pos, 1500);
}

#[test]
fn create_chunks_last_flag() {
    let tokens: Vec<u32> = (0..1500).collect();
    let cfg = ChunkedPrefillConfig::new(512);
    let chunks = create_prefill_chunks(&tokens, &cfg);
    for (i, chunk) in chunks.iter().enumerate() {
        if i < chunks.len() - 1 {
            assert!(!chunk.is_last, "chunk {i} should not be last");
        } else {
            assert!(chunk.is_last, "final chunk should be last");
        }
    }
}

#[test]
fn create_chunks_with_overlap() {
    let tokens: Vec<u32> = (0..1024).collect();
    let cfg = ChunkedPrefillConfig::new(512).with_overlap(64);
    let chunks = create_prefill_chunks(&tokens, &cfg);
    // stride = 512 - 64 = 448, so chunks start at 0, 448, 896
    assert!(chunks.len() >= 3);
    // Overlapping tokens: chunk[0] ends at 512, chunk[1] starts at 448
    // so tokens 448..512 are shared.
    let overlap_start = chunks[1].start_pos;
    let overlap_end = chunks[0].end_pos;
    assert!(overlap_end > overlap_start, "chunks should overlap");
    assert_eq!(overlap_end - overlap_start, 64);
}

#[test]
fn scheduler_prefill_first() {
    let tokens: Vec<u32> = (0..1024).collect();
    let cfg = ChunkedPrefillConfig::new(512).with_priority(PrefillPriority::PrefillFirst);
    let mut sched = PrefillScheduler::new(&tokens, cfg);

    // Should get two prefill actions then StartDecode.
    match sched.next_action() {
        PrefillAction::Prefill(c) => assert_eq!(c.chunk_index, 0),
        other => panic!("expected Prefill, got {:?}", other),
    }
    match sched.next_action() {
        PrefillAction::Prefill(c) => assert_eq!(c.chunk_index, 1),
        other => panic!("expected Prefill, got {:?}", other),
    }
    match sched.next_action() {
        PrefillAction::StartDecode => {}
        other => panic!("expected StartDecode, got {:?}", other),
    }
}

#[test]
fn scheduler_interleaved() {
    let tokens: Vec<u32> = (0..1024).collect();
    let cfg = ChunkedPrefillConfig::new(512).with_priority(PrefillPriority::Interleaved);
    let mut sched = PrefillScheduler::new(&tokens, cfg);

    // Prefill -> YieldToDecode -> Prefill -> StartDecode
    match sched.next_action() {
        PrefillAction::Prefill(_) => {}
        other => panic!("expected Prefill, got {:?}", other),
    }
    match sched.next_action() {
        PrefillAction::YieldToDecode => {}
        other => panic!("expected YieldToDecode, got {:?}", other),
    }
    match sched.next_action() {
        PrefillAction::Prefill(_) => {}
        other => panic!("expected Prefill, got {:?}", other),
    }
    // After last chunk processed, next should be StartDecode
    match sched.next_action() {
        PrefillAction::StartDecode => {}
        other => panic!("expected StartDecode, got {:?}", other),
    }
}

#[test]
fn scheduler_is_complete() {
    let tokens: Vec<u32> = (0..512).collect();
    let cfg = ChunkedPrefillConfig::new(512);
    let mut sched = PrefillScheduler::new(&tokens, cfg);
    assert!(!sched.is_complete());
    let _ = sched.next_action(); // consume the one chunk
    assert!(sched.is_complete());
}

#[test]
fn scheduler_progress() {
    let tokens: Vec<u32> = (0..1024).collect();
    let cfg = ChunkedPrefillConfig::new(512);
    let mut sched = PrefillScheduler::new(&tokens, cfg);
    assert!((sched.progress() - 0.0).abs() < f32::EPSILON);
    let _ = sched.next_action();
    assert!((sched.progress() - 0.5).abs() < f32::EPSILON);
    let _ = sched.next_action();
    assert!((sched.progress() - 1.0).abs() < f32::EPSILON);
}

#[test]
fn scheduler_total_chunks() {
    let tokens: Vec<u32> = (0..1500).collect();
    let cfg = ChunkedPrefillConfig::new(512);
    let chunks = create_prefill_chunks(&tokens, &cfg);
    let sched = PrefillScheduler::new(&tokens, cfg);
    assert_eq!(sched.total_chunks(), chunks.len());
}

#[test]
fn peak_memory_estimate_savings() {
    let est = peak_memory_estimate(2048, 512, 4096, 32);
    assert!(est.chunked_prefill_bytes < est.full_prefill_bytes);
    assert!(est.memory_savings_ratio > 0.0);
}

#[test]
fn memory_estimate_summary_nonempty() {
    let est = peak_memory_estimate(2048, 512, 4096, 32);
    let s = est.summary();
    assert!(!s.is_empty());
    assert!(s.contains("MB"));
}

#[test]
fn prefill_chunk_len() {
    let chunk = PrefillChunk {
        tokens: vec![1, 2, 3, 4, 5],
        start_pos: 0,
        end_pos: 5,
        chunk_index: 0,
        is_last: true,
    };
    assert_eq!(chunk.len(), 5);
    assert!(!chunk.is_empty());
}

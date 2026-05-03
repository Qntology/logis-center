//! Integration tests for gradient checkpointing.

use oxibonsai_model::gradient_checkpoint::{
    CheckpointBudget, CheckpointError, CheckpointSegment, CheckpointStrategy,
    CheckpointedActivation, CheckpointedPipeline,
};

#[test]
fn budget_new() {
    let b = CheckpointBudget::new(4096);
    assert_eq!(b.used_bytes, 0, "fresh budget should have used_bytes = 0");
    assert_eq!(b.max_bytes, 4096);
}

#[test]
fn budget_allocate_ok() {
    let mut b = CheckpointBudget::new(1024);
    b.allocate(256)
        .expect("allocation within budget should succeed");
    assert_eq!(b.used_bytes, 256, "used_bytes should increase");
    b.allocate(256).expect("second allocation should succeed");
    assert_eq!(b.used_bytes, 512, "used_bytes should accumulate");
}

#[test]
fn budget_allocate_exceed() {
    let mut b = CheckpointBudget::new(100);
    let result = b.allocate(200);
    assert!(
        matches!(result, Err(CheckpointError::BudgetExceeded { .. })),
        "exceeding budget must return BudgetExceeded"
    );
    assert_eq!(b.used_bytes, 0, "failed alloc must not change used_bytes");
}

#[test]
fn budget_free() {
    let mut b = CheckpointBudget::new(1024);
    b.allocate(512).expect("allocation should succeed");
    b.free(256);
    assert_eq!(b.used_bytes, 256, "free should decrease used_bytes");
    b.free(512); // saturating
    assert_eq!(b.used_bytes, 0, "free should saturate at 0");
}

#[test]
fn budget_utilization() {
    let mut b = CheckpointBudget::new(1000);
    b.allocate(250).expect("allocation should succeed");
    let util = b.utilization();
    assert!(
        (util - 0.25).abs() < 1e-6,
        "utilization should be 0.25, got {util}"
    );
}

#[test]
fn segment_forward_shape() {
    let seg = CheckpointSegment::init_lcg("test", 4, 8, 42);
    let input = vec![1.0f32; 4];
    let out = seg.forward(&input).expect("forward should succeed");
    assert_eq!(out.len(), 8, "output should have out_dim elements");
}

#[test]
fn segment_forward_deterministic() {
    let seg = CheckpointSegment::init_lcg("det", 4, 8, 99);
    let input = vec![0.5f32, -0.5, 1.0, -1.0];
    let out1 = seg.forward(&input).expect("first call should succeed");
    let out2 = seg.forward(&input).expect("second call should succeed");
    assert_eq!(out1, out2, "forward must be deterministic");
}

#[test]
fn checkpointed_activation_recompute() {
    let seg = CheckpointSegment::init_lcg("recomp", 3, 6, 7);
    let input = vec![1.0f32, 2.0, 3.0];
    let expected = seg.forward(&input).expect("forward should succeed");
    let act = CheckpointedActivation::new(CheckpointSegment::init_lcg("recomp", 3, 6, 7), input);
    let got = act.recompute().expect("recompute should succeed");
    assert_eq!(got, expected, "recompute must equal forward");
}

#[test]
fn checkpointed_activation_memory_savings() {
    // in_dim=4, out_dim=16 → input takes 16 bytes, output takes 64 bytes
    // savings = 1 - 16/80 = 0.8
    let seg = CheckpointSegment::init_lcg("savings", 4, 16, 0);
    let input = vec![1.0f32; 4];
    let act = CheckpointedActivation::new(seg, input);
    let savings = act.memory_savings();
    assert!(
        savings > 0.0,
        "expanding segment should have positive savings, got {savings}"
    );
}

#[test]
fn pipeline_forward_runs() {
    let seg1 = CheckpointSegment::init_lcg("l1", 4, 8, 1);
    let seg2 = CheckpointSegment::init_lcg("l2", 8, 4, 2);
    let pipe = CheckpointedPipeline::new(vec![seg1, seg2]);
    let input = vec![1.0f32; 4];
    let out = pipe
        .forward(&input)
        .expect("pipeline forward should succeed");
    assert_eq!(out.len(), 4, "output should match final out_dim");
}

#[test]
fn pipeline_num_segments() {
    let seg1 = CheckpointSegment::init_lcg("a", 2, 4, 10);
    let seg2 = CheckpointSegment::init_lcg("b", 4, 8, 11);
    let seg3 = CheckpointSegment::init_lcg("c", 8, 2, 12);
    let pipe = CheckpointedPipeline::new(vec![seg1, seg2, seg3]);
    assert_eq!(pipe.num_segments(), 3);
}

#[test]
fn pipeline_overall_savings() {
    // Expanding pipeline: 4→16→64
    let seg1 = CheckpointSegment::init_lcg("s1", 4, 16, 20);
    let seg2 = CheckpointSegment::init_lcg("s2", 16, 64, 21);
    let pipe = CheckpointedPipeline::new(vec![seg1, seg2]);
    let savings = pipe.overall_savings(4);
    assert!(
        savings > 0.0,
        "expanding pipeline should save memory, got {savings}"
    );
}

#[test]
fn strategy_every_all_layers() {
    let layers = CheckpointStrategy::Every.select_layers(5);
    assert_eq!(layers, vec![0, 1, 2, 3, 4]);
}

#[test]
fn strategy_every_nth() {
    let layers = CheckpointStrategy::EveryNth(2).select_layers(6);
    assert_eq!(layers, vec![0, 2, 4], "every 2nd layer from 6 total");
}

#[test]
fn strategy_sqrt_count() {
    let layers = CheckpointStrategy::Sqrt.select_layers(16);
    // sqrt(16) = 4
    assert_eq!(layers.len(), 4, "sqrt(16) should select 4 layers");
    // Should be sorted
    for w in layers.windows(2) {
        assert!(w[0] < w[1], "layers should be sorted");
    }
}

#[test]
fn strategy_none_empty() {
    let layers = CheckpointStrategy::None.select_layers(10);
    assert!(layers.is_empty(), "None strategy should select no layers");
}

//! Integration tests for cross-attention.

use oxibonsai_model::layers::cross_attention::{
    compute_attention_weights, cross_attention_forward, single_head_cross_attention,
    CrossAttentionConfig, CrossAttnError,
};

const EPS: f32 = 1e-5;

#[test]
fn config_hidden_dim() {
    let cfg = CrossAttentionConfig::new(4, 8);
    assert_eq!(cfg.hidden_dim(), 32, "hidden_dim = num_heads * head_dim");
}

#[test]
fn cross_attention_output_shape() {
    let num_heads = 2;
    let head_dim = 4;
    let dec_seq = 3;
    let enc_seq = 5;
    let cfg = CrossAttentionConfig::new(num_heads, head_dim);
    let dec = vec![0.1f32; dec_seq * cfg.hidden_dim()];
    let enc = vec![0.2f32; enc_seq * cfg.hidden_dim()];
    let out = cross_attention_forward(&dec, &enc, dec_seq, enc_seq, &cfg, None)
        .expect("cross_attention_forward should succeed");
    assert_eq!(out.len(), dec_seq * cfg.hidden_dim());
}

#[test]
fn cross_attention_with_mask() {
    let num_heads = 1;
    let head_dim = 2;
    let dec_seq = 1;
    let enc_seq = 3;
    let cfg = CrossAttentionConfig::new(num_heads, head_dim);

    let dec = vec![1.0f32, 0.0];
    // Encoder: 3 positions, only position 0 unmasked
    let enc = vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0];
    let mask = vec![true, false, false];

    let out = cross_attention_forward(&dec, &enc, dec_seq, enc_seq, &cfg, Some(&mask))
        .expect("masked cross attention should succeed");

    // Only position 0 contributes (weight=1.0), value=[1.0, 0.0]
    assert!(
        (out[0] - 1.0).abs() < EPS,
        "output[0] = {} expected 1.0",
        out[0]
    );
    assert!(
        (out[1] - 0.0).abs() < EPS,
        "output[1] = {} expected 0.0",
        out[1]
    );
}

#[test]
fn single_head_output_shape() {
    let dec_seq = 3;
    let enc_seq = 5;
    let head_dim = 4;
    let q = vec![0.1f32; dec_seq * head_dim];
    let k = vec![0.2f32; enc_seq * head_dim];
    let v = vec![0.3f32; enc_seq * head_dim];
    let out = single_head_cross_attention(&q, &k, &v, dec_seq, enc_seq, head_dim, 0.5, None)
        .expect("single head should succeed");
    assert_eq!(out.len(), dec_seq * head_dim);
}

#[test]
fn single_head_deterministic() {
    let dec_seq = 2;
    let enc_seq = 3;
    let head_dim = 4;
    let q: Vec<f32> = (0..dec_seq * head_dim).map(|i| i as f32 * 0.1).collect();
    let k: Vec<f32> = (0..enc_seq * head_dim).map(|i| i as f32 * 0.05).collect();
    let v: Vec<f32> = (0..enc_seq * head_dim).map(|i| (i as f32).sin()).collect();
    let out1 = single_head_cross_attention(&q, &k, &v, dec_seq, enc_seq, head_dim, 0.5, None)
        .expect("first call should succeed");
    let out2 = single_head_cross_attention(&q, &k, &v, dec_seq, enc_seq, head_dim, 0.5, None)
        .expect("second call should succeed");
    assert_eq!(out1, out2, "single_head must be deterministic");
}

#[test]
fn attention_weights_shape() {
    let dec_seq = 3;
    let enc_seq = 5;
    let head_dim = 4;
    let q = vec![0.1f32; dec_seq * head_dim];
    let k = vec![0.2f32; enc_seq * head_dim];
    let weights = compute_attention_weights(&q, &k, dec_seq, enc_seq, head_dim, 0.5)
        .expect("compute_attention_weights should succeed");
    assert_eq!(weights.len(), dec_seq * enc_seq);
}

#[test]
fn attention_weights_rows_sum_to_one() {
    let dec_seq = 4;
    let enc_seq = 6;
    let head_dim = 8;
    let q: Vec<f32> = (0..dec_seq * head_dim)
        .map(|i| (i as f32) * 0.1 - 1.0)
        .collect();
    let k: Vec<f32> = (0..enc_seq * head_dim).map(|i| (i as f32) * 0.05).collect();
    let weights = compute_attention_weights(&q, &k, dec_seq, enc_seq, head_dim, 0.5)
        .expect("compute_attention_weights should succeed");

    for dq in 0..dec_seq {
        let row_sum: f32 = weights[dq * enc_seq..(dq + 1) * enc_seq].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < EPS,
            "row {dq} sums to {row_sum}, expected ~1.0"
        );
    }
}

#[test]
fn cross_attn_invalid_head_dim_error() {
    let cfg = CrossAttentionConfig::new(2, 0);
    let dec = vec![];
    let enc = vec![];
    let result = cross_attention_forward(&dec, &enc, 1, 1, &cfg, None);
    assert!(
        matches!(result, Err(CrossAttnError::InvalidHeadDim)),
        "head_dim=0 should return InvalidHeadDim"
    );
}

#[test]
fn cross_attn_decoder_dim_mismatch_error() {
    let cfg = CrossAttentionConfig::new(2, 4);
    let dec = vec![0.0f32; 3]; // should be 1 * 2 * 4 = 8
    let enc = vec![0.0f32; 8];
    let result = cross_attention_forward(&dec, &enc, 1, 1, &cfg, None);
    assert!(
        matches!(result, Err(CrossAttnError::DecoderDimMismatch { .. })),
        "wrong decoder size should return DecoderDimMismatch"
    );
}

#[test]
fn cross_attn_encoder_dim_mismatch_error() {
    let cfg = CrossAttentionConfig::new(2, 4);
    let dec = vec![0.0f32; 8];
    let enc = vec![0.0f32; 3]; // should be 1 * 2 * 4 = 8
    let result = cross_attention_forward(&dec, &enc, 1, 1, &cfg, None);
    assert!(
        matches!(result, Err(CrossAttnError::EncoderDimMismatch { .. })),
        "wrong encoder size should return EncoderDimMismatch"
    );
}

#[test]
fn cross_attention_no_mask() {
    let num_heads = 1;
    let head_dim = 2;
    let dec_seq = 2;
    let enc_seq = 2;
    let cfg = CrossAttentionConfig::new(num_heads, head_dim);
    let dec = vec![0.0f32; dec_seq * cfg.hidden_dim()];
    let enc = vec![1.0f32; enc_seq * cfg.hidden_dim()];
    let out = cross_attention_forward(&dec, &enc, dec_seq, enc_seq, &cfg, None)
        .expect("no-mask cross attention should succeed");
    assert_eq!(out.len(), dec_seq * cfg.hidden_dim());
    // With zero queries and uniform encoder, output should be the encoder values
    for v in &out {
        assert!(
            (v - 1.0).abs() < EPS,
            "with zero queries and uniform encoder, output should equal encoder values"
        );
    }
}

//! Property-based tests for invariants that should hold for every input.

use std::collections::HashMap;

use oxibonsai_rag::metadata_filter::{MetadataFilter, MetadataValue};
use oxibonsai_rag::{Distance, RagError};
use proptest::prelude::*;

// ── helpers ─────────────────────────────────────────────────────────────────

fn finite_vec_strategy(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    prop::collection::vec(-1000.0f32..1000.0f32, dim..=dim)
}

// ── Distance symmetry ───────────────────────────────────────────────────────

proptest! {
    #[test]
    fn cosine_is_symmetric(a in finite_vec_strategy(4), b in finite_vec_strategy(4)) {
        let ab = Distance::Cosine.compute(&a, &b).expect("compute ab");
        let ba = Distance::Cosine.compute(&b, &a).expect("compute ba");
        prop_assert!((ab - ba).abs() < 1e-4, "cosine(a,b)={ab} cosine(b,a)={ba}");
    }

    #[test]
    fn euclidean_is_symmetric(a in finite_vec_strategy(4), b in finite_vec_strategy(4)) {
        let ab = Distance::Euclidean.compute(&a, &b).expect("compute ab");
        let ba = Distance::Euclidean.compute(&b, &a).expect("compute ba");
        prop_assert!((ab - ba).abs() < 1e-3);
    }

    #[test]
    fn dot_product_is_symmetric(a in finite_vec_strategy(4), b in finite_vec_strategy(4)) {
        let ab = Distance::DotProduct.compute(&a, &b).expect("compute ab");
        let ba = Distance::DotProduct.compute(&b, &a).expect("compute ba");
        prop_assert!((ab - ba).abs() < 1e-2);
    }

    #[test]
    fn angular_is_symmetric(a in finite_vec_strategy(4), b in finite_vec_strategy(4)) {
        let ab = Distance::Angular.compute(&a, &b).expect("compute ab");
        let ba = Distance::Angular.compute(&b, &a).expect("compute ba");
        prop_assert!((ab - ba).abs() < 1e-4);
    }

    #[test]
    fn hamming_is_symmetric(a in finite_vec_strategy(4), b in finite_vec_strategy(4)) {
        let ab = Distance::Hamming.compute(&a, &b).expect("compute ab");
        let ba = Distance::Hamming.compute(&b, &a).expect("compute ba");
        prop_assert!((ab - ba).abs() < 1e-6);
    }
}

// ── Identity: d(a, a) = min for distances ───────────────────────────────────

proptest! {
    #[test]
    fn cosine_self_similarity_is_one(a in finite_vec_strategy(4)) {
        // Skip near-zero vectors (cosine is defined as 0 there).
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assume!(norm > 1e-3);
        let score = Distance::Cosine.compute(&a, &a).expect("compute");
        prop_assert!((score - 1.0).abs() < 1e-3, "got {score}");
    }

    #[test]
    fn euclidean_self_distance_is_zero(a in finite_vec_strategy(4)) {
        let d = Distance::Euclidean.compute(&a, &a).expect("compute");
        prop_assert!(d.abs() < 1e-3, "got {d}");
    }

    #[test]
    fn hamming_self_distance_is_zero(a in finite_vec_strategy(4)) {
        let d = Distance::Hamming.compute(&a, &a).expect("compute");
        prop_assert_eq!(d, 0.0);
    }
}

// ── Euclidean triangle inequality ───────────────────────────────────────────

proptest! {
    #[test]
    fn euclidean_triangle_inequality(
        a in finite_vec_strategy(3),
        b in finite_vec_strategy(3),
        c in finite_vec_strategy(3),
    ) {
        let ab = Distance::Euclidean.compute(&a, &b).expect("ab");
        let bc = Distance::Euclidean.compute(&b, &c).expect("bc");
        let ac = Distance::Euclidean.compute(&a, &c).expect("ac");
        // Use a looser epsilon to absorb f32 rounding.
        prop_assert!(ac <= ab + bc + 1e-2, "ac={ac} ab+bc={}", ab + bc);
    }
}

// ── Normalisation idempotence ───────────────────────────────────────────────

proptest! {
    #[test]
    fn l2_normalize_is_idempotent(a in finite_vec_strategy(5)) {
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assume!(norm > 1e-3);
        let mut v = a.clone();
        oxibonsai_rag::vector_store::l2_normalize(&mut v);
        let mut w = v.clone();
        oxibonsai_rag::vector_store::l2_normalize(&mut w);
        for (x, y) in v.iter().zip(w.iter()) {
            prop_assert!((x - y).abs() < 1e-5);
        }
    }
}

// ── Angular lies in [0, 1] ──────────────────────────────────────────────────

proptest! {
    #[test]
    fn angular_is_bounded_01(a in finite_vec_strategy(4), b in finite_vec_strategy(4)) {
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        prop_assume!(na > 1e-3 && nb > 1e-3);
        let d = Distance::Angular.compute(&a, &b).expect("compute");
        prop_assert!((0.0..=1.0).contains(&d), "got {d}");
    }
}

// ── NaN propagation always becomes NonFinite ────────────────────────────────

proptest! {
    #[test]
    fn nan_always_rejected(i in 0usize..4) {
        let mut a = vec![1.0f32; 4];
        let b = vec![0.5f32; 4];
        a[i] = f32::NAN;
        prop_assert!(matches!(
            Distance::Cosine.compute(&a, &b),
            Err(RagError::NonFinite)
        ));
    }
}

// ── Filter negation duality ─────────────────────────────────────────────────

proptest! {
    #[test]
    fn equals_and_not_equals_are_disjoint(value in "[a-z]{1,8}") {
        let mut m: HashMap<String, MetadataValue> = HashMap::new();
        m.insert("k".into(), MetadataValue::from(value.as_str()));
        let eq = MetadataFilter::eq("k", value.as_str());
        let neq = MetadataFilter::neq("k", value.as_str());
        prop_assert!(eq.matches(&m) != neq.matches(&m));
    }
}

// ── Empty All / non-empty Any ───────────────────────────────────────────────

proptest! {
    #[test]
    fn all_empty_always_matches(value in "[a-z]{1,8}") {
        let mut m: HashMap<String, MetadataValue> = HashMap::new();
        m.insert("k".into(), MetadataValue::from(value.as_str()));
        prop_assert!(MetadataFilter::All(vec![]).matches(&m));
    }

    #[test]
    fn any_empty_never_matches(value in "[a-z]{1,8}") {
        let mut m: HashMap<String, MetadataValue> = HashMap::new();
        m.insert("k".into(), MetadataValue::from(value.as_str()));
        prop_assert!(!MetadataFilter::Any(vec![]).matches(&m));
    }
}

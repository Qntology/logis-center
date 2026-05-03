//! Integration tests for grammar-constrained decoding.

use oxibonsai_runtime::constrained_decoding::{
    ConstrainedSampler, ConstrainedSamplerBuilder, ConstraintError, JsonConstraint, JsonParseState,
    NoConstraint, RegexConstraint, TokenConstraint,
};
use oxibonsai_runtime::sampling_advanced::SamplerChain;

// ─────────────────────────────────────────────────────────────────────────────
// NoConstraint
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_no_constraint_allows_all() {
    let nc = NoConstraint;
    assert!(nc.allowed_tokens(&[], 32).is_none());
    assert!(nc.allowed_tokens(&[1, 2, 3], 1000).is_none());
}

// ─────────────────────────────────────────────────────────────────────────────
// JsonConstraint
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_json_constraint_initial_state() {
    let jc = JsonConstraint::new();
    assert_eq!(*jc.current_state(), JsonParseState::Start);
    assert_eq!(jc.depth(), 0);
    assert!(!jc.is_in_string());
    assert!(!jc.is_complete());
}

#[test]
fn test_json_constraint_valid_object_chars() {
    let jc = JsonConstraint::new();
    let chars = jc.valid_next_chars();
    assert!(chars.contains(&'{'));
    assert!(chars.contains(&'['));
    assert!(chars.contains(&'"'));
    assert!(chars.contains(&'t')); // true
    assert!(chars.contains(&'f')); // false
    assert!(chars.contains(&'n')); // null
}

#[test]
fn test_json_constraint_tracks_depth() {
    let mut jc = JsonConstraint::new();
    assert_eq!(jc.depth(), 0);

    jc.advance('{' as u32);
    assert_eq!(jc.depth(), 1);

    // Add a nested object: `"k": {}`
    jc.advance('"' as u32);
    jc.advance('k' as u32);
    jc.advance('"' as u32);
    jc.advance(':' as u32);
    jc.advance('{' as u32);
    assert_eq!(jc.depth(), 2);

    jc.advance('}' as u32);
    assert_eq!(jc.depth(), 1);

    jc.advance('}' as u32);
    assert_eq!(jc.depth(), 0);
}

#[test]
fn test_json_constraint_detects_completion() {
    let mut jc = JsonConstraint::new();
    assert!(!jc.is_complete());

    jc.advance('{' as u32);
    assert!(!jc.is_complete());

    jc.advance('}' as u32);
    assert!(jc.is_complete());
}

#[test]
fn test_json_constraint_in_string_state() {
    let mut jc = JsonConstraint::new();
    assert!(!jc.is_in_string());

    jc.advance('"' as u32);
    assert!(jc.is_in_string());

    jc.advance('h' as u32);
    assert!(jc.is_in_string());

    jc.advance('"' as u32);
    assert!(!jc.is_in_string());
}

// ─────────────────────────────────────────────────────────────────────────────
// RegexNfa (via RegexConstraint::is_match)
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_regex_nfa_literal_match() {
    assert!(RegexConstraint::is_match("abc", "abc"));
    assert!(!RegexConstraint::is_match("abc", "ab"));
    assert!(!RegexConstraint::is_match("abc", "abcd"));
    assert!(!RegexConstraint::is_match("abc", "xyz"));
}

#[test]
fn test_regex_nfa_dot_match() {
    assert!(RegexConstraint::is_match("a.c", "abc"));
    assert!(RegexConstraint::is_match("a.c", "a1c"));
    assert!(!RegexConstraint::is_match("a.c", "ac"));
    assert!(!RegexConstraint::is_match("a.c", "abbc"));
}

#[test]
fn test_regex_nfa_star_quantifier() {
    assert!(RegexConstraint::is_match("ab*c", "ac"));
    assert!(RegexConstraint::is_match("ab*c", "abc"));
    assert!(RegexConstraint::is_match("ab*c", "abbbc"));
    assert!(!RegexConstraint::is_match("ab*c", "xbc"));
    assert!(!RegexConstraint::is_match("ab*c", "ab"));
}

#[test]
fn test_regex_nfa_alternation() {
    assert!(RegexConstraint::is_match("cat|dog", "cat"));
    assert!(RegexConstraint::is_match("cat|dog", "dog"));
    assert!(!RegexConstraint::is_match("cat|dog", "cow"));
    assert!(!RegexConstraint::is_match("cat|dog", "catdog"));
}

// ─────────────────────────────────────────────────────────────────────────────
// RegexConstraint
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_regex_constraint_is_match() {
    assert!(RegexConstraint::is_match("he+llo", "hello"));
    assert!(RegexConstraint::is_match("he+llo", "heeeello"));
    assert!(!RegexConstraint::is_match("he+llo", "hllo"));
    assert!(!RegexConstraint::is_match("he+llo", "hello world"));
}

#[test]
fn test_regex_constraint_allows_valid_chars() {
    let rc = RegexConstraint::new("abc").expect("valid pattern");
    // 'a' is the only valid first character
    assert!(rc.char_is_valid('a'));
    // 'b' cannot be the first char for pattern "abc"
    assert!(!rc.char_is_valid('b'));
    assert!(!rc.char_is_valid('c'));
    assert!(!rc.char_is_valid('z'));
}

// ─────────────────────────────────────────────────────────────────────────────
// ConstrainedSampler
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_constrained_sampler_masks_logits() {
    // Custom constraint: only allow even-indexed tokens.
    struct AllowEvens;
    impl TokenConstraint for AllowEvens {
        fn allowed_tokens(&self, _: &[u32], vocab_size: usize) -> Option<Vec<bool>> {
            Some((0..vocab_size).map(|i| i % 2 == 0).collect())
        }
        fn advance(&mut self, _: u32) -> bool {
            true
        }
        fn is_complete(&self) -> bool {
            true
        }
        fn reset(&mut self) {}
        fn name(&self) -> &str {
            "AllowEvens"
        }
    }

    let chain = SamplerChain::greedy();
    let mut sampler = ConstrainedSampler::new(chain, Box::new(AllowEvens), 4);

    // Token 1 has the highest raw logit, but it's odd → masked.
    // Token 0 is the highest-logit even token.
    let mut logits = vec![2.0_f32, 10.0, 1.0, 0.5];
    let tok = sampler.sample(&mut logits);
    assert_eq!(tok, 0, "token 0 should win after masking odd tokens");
}

#[test]
fn test_constrained_sampler_greedy_json() {
    let chain = SamplerChain::greedy();
    let mut sampler = ConstrainedSampler::new(chain, Box::new(JsonConstraint::new()), 256);

    assert!(!sampler.is_complete());

    // Drive the sampler to emit `{}`
    let mut logits_open = vec![0.0_f32; 256];
    logits_open['{' as usize] = 100.0;
    sampler.sample(&mut logits_open);

    let mut logits_close = vec![0.0_f32; 256];
    logits_close['}' as usize] = 100.0;
    sampler.sample(&mut logits_close);

    assert!(
        sampler.is_complete(),
        "constraint should be satisfied after `{{}}`"
    );
    assert_eq!(sampler.generated_text_len(), 2);
}

#[test]
fn test_constrained_sampler_reset() {
    let chain = SamplerChain::greedy();
    let mut sampler = ConstrainedSampler::new(chain, Box::new(JsonConstraint::new()), 256);

    let mut logits = vec![0.0_f32; 256];
    logits['{' as usize] = 100.0;
    sampler.sample(&mut logits);

    assert_eq!(sampler.generated_text_len(), 1);

    sampler.reset();
    assert_eq!(sampler.generated_text_len(), 0);
    assert!(
        !sampler.is_complete(),
        "after reset constraint should not be complete"
    );
}

#[test]
fn test_constrained_sampler_builder_json() {
    let sampler = ConstrainedSamplerBuilder::new(256, 42).with_json_constraint();
    assert_eq!(sampler.constraint_name(), "JsonConstraint");
    assert!(!sampler.is_complete());
}

#[test]
fn test_constrained_sampler_builder_unconstrained() {
    let sampler = ConstrainedSamplerBuilder::new(256, 42).unconstrained();
    assert_eq!(sampler.constraint_name(), "NoConstraint");
    assert!(sampler.is_complete(), "NoConstraint is always complete");
}

// ─────────────────────────────────────────────────────────────────────────────
// ConstraintError
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_constraint_error_display() {
    let e1 = ConstraintError::InvalidPattern("bad[".to_string());
    assert!(
        e1.to_string().contains("bad["),
        "should contain pattern text"
    );

    let e2 = ConstraintError::InvalidSchema("missing type".to_string());
    assert!(e2.to_string().contains("missing type"));

    let e3 = ConstraintError::Violated {
        token: 99,
        reason: "oops".to_string(),
    };
    let s = e3.to_string();
    assert!(s.contains("99"));
    assert!(s.contains("oops"));
}

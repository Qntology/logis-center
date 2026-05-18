/// Constrained Viterbi decoder for BIOES span labelling.
///
/// The decoder enforces valid BIOES transitions:
///   O  → O, B-*, S-*
///   B-X → I-X, E-X
///   I-X → I-X, E-X
///   E-X → O, B-*, S-*
///   S-X → O, B-*, S-*

use super::config::ViterbiConfig;

const NEG_INF: f64 = -1e30;

/// Get the BIOES tag type for a label index.
/// Returns: 'O', 'B', 'I', 'E', or 'S'
fn label_tag(label: usize) -> char {
    if label == 0 {
        'O'
    } else {
        match (label - 1) % 4 {
            0 => 'B',
            1 => 'I',
            2 => 'E',
            3 => 'S',
            _ => unreachable!(),
        }
    }
}

/// Get the category index for a label. Returns None for O.
fn label_category(label: usize) -> Option<usize> {
    if label == 0 {
        None
    } else {
        Some((label - 1) / 4)
    }
}

/// Check if transition from `prev` to `curr` is valid under BIOES constraints.
fn is_valid_transition(prev: usize, curr: usize) -> bool {
    let prev_tag = label_tag(prev);
    let curr_tag = label_tag(curr);
    let prev_cat = label_category(prev);
    let curr_cat = label_category(curr);

    match prev_tag {
        'O' | 'E' | 'S' => {
            // Can go to O, B-*, or S-*
            matches!(curr_tag, 'O' | 'B' | 'S')
        }
        'B' | 'I' => {
            // Must continue or end the SAME category
            match curr_tag {
                'I' | 'E' => prev_cat == curr_cat,
                _ => false,
            }
        }
        _ => false,
    }
}

/// Get the transition bias for a given transition.
fn transition_bias(prev: usize, curr: usize, config: &ViterbiConfig) -> f64 {
    let prev_tag = label_tag(prev);
    let curr_tag = label_tag(curr);

    match (prev_tag, curr_tag) {
        ('O', 'O') => config.transition_bias_background_stay,
        ('O', 'B') | ('O', 'S') => config.transition_bias_background_to_start,
        ('B', 'I') | ('I', 'I') => config.transition_bias_inside_to_continue,
        ('B', 'E') | ('I', 'E') => config.transition_bias_inside_to_end,
        ('E', 'O') | ('S', 'O') => config.transition_bias_end_to_background,
        ('E', 'B') | ('E', 'S') | ('S', 'B') | ('S', 'S') => config.transition_bias_end_to_start,
        _ => 0.0,
    }
}

/// Run constrained Viterbi decoding on logits.
pub fn viterbi_decode(logits: &[f32], seq_len: usize, num_labels: usize, config: &ViterbiConfig) -> Vec<usize> {
    if seq_len == 0 {
        return vec![];
    }

    let mut dp = vec![vec![NEG_INF; num_labels]; seq_len];
    let mut bp = vec![vec![0usize; num_labels]; seq_len];

    for s in 0..num_labels {
        let tag = label_tag(s);
        if matches!(tag, 'O' | 'B' | 'S') {
            dp[0][s] = logits[s] as f64;
        }
    }

    for t in 1..seq_len {
        for curr in 0..num_labels {
            let emission = logits[t * num_labels + curr] as f64;
            let mut best_score = NEG_INF;
            let mut best_prev = 0;

            for prev in 0..num_labels {
                if !is_valid_transition(prev, curr) {
                    continue;
                }
                let score = dp[t - 1][prev]
                    + transition_bias(prev, curr, config)
                    + emission;
                if score > best_score {
                    best_score = score;
                    best_prev = prev;
                }
            }

            dp[t][curr] = best_score;
            bp[t][curr] = best_prev;
        }
    }

    let mut best_final = 0;
    let mut best_score = NEG_INF;
    for s in 0..num_labels {
        let tag = label_tag(s);
        if matches!(tag, 'O' | 'E' | 'S') && dp[seq_len - 1][s] > best_score {
            best_score = dp[seq_len - 1][s];
            best_final = s;
        }
    }

    let mut path = vec![0usize; seq_len];
    path[seq_len - 1] = best_final;
    for t in (1..seq_len).rev() {
        path[t - 1] = bp[t][path[t]];
    }

    path
}

/// A detected privacy span.
#[derive(Debug, Clone)]
pub struct PrivacySpan {
    pub entity_group: String,
    pub score: f32,
    pub word: String,
    pub start: usize,
    pub end: usize,
}

/// Extract spans from Viterbi-decoded label path.
pub fn extract_spans(
    label_path: &[usize],
    logits: &[f32],
    num_labels: usize,
    label_list: &[String],
    tokens: &[String],
    offsets: &[(usize, usize)],
    input_text: &str,
) -> Vec<PrivacySpan> {
    let mut spans = Vec::new();
    let seq_len = label_path.len();
    let mut i = 0;

    while i < seq_len {
        let label = label_path[i];
        let tag = label_tag(label);

        match tag {
            'S' => {
                // Single-token span
                let label_name = &label_list[label];
                let cat_name = label_name.strip_prefix("S-").unwrap_or(label_name);
                
                let score = compute_span_score(logits, num_labels, &[i], label_path);
                let (start, end) = offsets[i];
                let word = if end > start && end <= input_text.len() {
                    input_text[start..end].to_string()
                } else {
                    tokens[i].clone()
                };
                spans.push(PrivacySpan {
                    entity_group: cat_name.to_string(),
                    score,
                    word,
                    start,
                    end,
                });
                i += 1;
            }
            'B' => {
                // Begin of multi-token span
                let label_name = &label_list[label];
                let cat_name = label_name.strip_prefix("B-").unwrap_or(label_name);
                let cat = label_category(label).unwrap();
                
                let span_start = i;
                let char_start = offsets[i].0;
                i += 1;

                while i < seq_len {
                    let next_label = label_path[i];
                    let next_tag = label_tag(next_label);
                    if (next_tag == 'I' || next_tag == 'E') && label_category(next_label) == Some(cat) {
                        i += 1;
                        if next_tag == 'E' { break; }
                    } else {
                        break;
                    }
                }

                let span_end = i;
                let char_end = offsets[span_end - 1].1;
                let token_indices: Vec<usize> = (span_start..span_end).collect();
                let score = compute_span_score(logits, num_labels, &token_indices, label_path);
                let word = if char_end > char_start && char_end <= input_text.len() {
                    input_text[char_start..char_end].to_string()
                } else {
                    tokens[span_start..span_end].join("")
                };
                spans.push(PrivacySpan {
                    entity_group: cat_name.to_string(),
                    score,
                    word,
                    start: char_start,
                    end: char_end,
                });
            }
            _ => {
                i += 1;
            }
        }
    }

    spans
}

/// Compute average softmax confidence for a span.
fn compute_span_score(logits: &[f32], num_labels: usize, token_indices: &[usize], label_path: &[usize]) -> f32 {
    if token_indices.is_empty() {
        return 0.0;
    }

    let mut total_score = 0.0;
    for &t in token_indices {
        let offset = t * num_labels;
        let label = label_path[t];

        let max_val = logits[offset..offset + num_labels]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits[offset..offset + num_labels]
            .iter()
            .map(|&v| (v - max_val).exp())
            .sum();
        let prob = (logits[offset + label] - max_val).exp() / exp_sum;
        total_score += prob;
    }

    total_score / token_indices.len() as f32
}

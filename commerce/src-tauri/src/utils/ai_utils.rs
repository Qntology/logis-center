pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 { 0.0 } else { dot_product / (norm_a * norm_b) }
}

pub fn extract_pug_context(lines: &[&str], target_idx: usize) -> String {
    if lines.is_empty() { return String::new(); }
    let mut parent_idx = target_idx;
    let target_indent = lines[target_idx].chars().take_while(|c| c.is_whitespace()).count();
    
    for i in (0..target_idx).rev() {
        let indent = lines[i].chars().take_while(|c| c.is_whitespace()).count();
        if indent < target_indent && !lines[i].trim().is_empty() {
            parent_idx = i;
            break;
        }
    }

    let parent_indent = lines[parent_idx].chars().take_while(|c| c.is_whitespace()).count();
    let mut context_lines = vec![lines[parent_idx]];
    
    for i in (parent_idx + 1)..lines.len() {
        if lines[i].trim().is_empty() { continue; }
        let indent = lines[i].chars().take_while(|c| c.is_whitespace()).count();
        if indent <= parent_indent {
            break;
        }
        context_lines.push(lines[i]);
    }
    
    context_lines.join("\n")
}
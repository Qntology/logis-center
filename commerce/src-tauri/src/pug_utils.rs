// 추출된 DOM 뭉치가 실제 원본 PUG의 몇 번째 줄부터 몇 번째 줄까지인지 정확하게 매핑합니다.
pub fn find_block_indices_in_pug<S: AsRef<str>>(full_lines: &[S], block_pug: &str) -> Option<(usize, usize)> {
    let b_lines: Vec<&str> = block_pug.lines().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    if b_lines.is_empty() { return None; }
    
    for i in 0..full_lines.len() {
        if full_lines[i].as_ref().trim() == b_lines[0] {
            let mut match_count = 1;
            let mut j = i + 1;
            let mut k = 1;
            while j < full_lines.len() && k < b_lines.len() {
                if full_lines[j].as_ref().trim().is_empty() { j += 1; continue; }
                if full_lines[j].as_ref().trim() == b_lines[k] { match_count += 1; k += 1; } 
                else { break; }
                j += 1;
            }
            if match_count == b_lines.len() { return Some((i, j - 1)); }
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct GridCell {
    pub row: usize,
    pub col: usize,
    pub rowspan: usize,
    pub colspan: usize,
    pub text: String,
    pub line_indices: Vec<usize>,
}

pub fn parse_pug_grid(lines: &[String]) -> Vec<GridCell> {
    let mut cells = Vec::new();
    let mut current_row = 0;
    let mut occupied = std::collections::HashSet::new();
    
    let mut i = 0;
    let mut in_tr = false;
    let mut tr_indent = 0;
    
    let re_row = regex::Regex::new(r#"rowspan="(\d+)""#).unwrap();
    let re_col = regex::Regex::new(r#"colspan="(\d+)""#).unwrap();
    
    while i < lines.len() {
        let line = &lines[i];
        let trimmed = line.trim();
        if trimmed.is_empty() { i += 1; continue; }
        let indent = line.chars().take_while(|c| c.is_whitespace()).count();
        
        if trimmed.starts_with("tr") && (trimmed.len() == 2 || trimmed.starts_with("tr[") || trimmed.starts_with("tr ")) {
            if in_tr { current_row += 1; }
            in_tr = true;
            tr_indent = indent;
            i += 1;
            continue;
        }
        
        if in_tr && indent <= tr_indent && !trimmed.starts_with("tr") {
            in_tr = false;
            current_row += 1;
        }
        
        if trimmed.starts_with("th") || trimmed.starts_with("td") {
            let is_tag = trimmed.starts_with("th[") || trimmed.starts_with("th ") || trimmed == "th" ||
                         trimmed.starts_with("td[") || trimmed.starts_with("td ") || trimmed == "td" ||
                         trimmed.starts_with("th|") || trimmed.starts_with("td|");
                         
            if is_tag {
                let mut rowspan = 1;
                let mut colspan = 1;
                
                if let Some(cap) = re_row.captures(trimmed) { rowspan = cap[1].parse().unwrap_or(1); }
                if let Some(cap) = re_col.captures(trimmed) { colspan = cap[1].parse().unwrap_or(1); }
                
                let mut current_col = 0;
                while occupied.contains(&(current_row, current_col)) {
                    current_col += 1;
                }
                
                for r in 0..rowspan {
                    for c in 0..colspan {
                        occupied.insert((current_row + r, current_col + c));
                    }
                }
                
                let mut text_acc = String::new();
                let mut line_indices = vec![i];
                
                if let Some(idx) = trimmed.find('|') {
                    text_acc.push_str(trimmed[idx + 1..].trim());
                    text_acc.push(' ');
                }
                
                let cell_indent = indent;
                let mut j = i + 1;
                while j < lines.len() {
                    let sub_line = &lines[j];
                    let sub_trim = sub_line.trim();
                    if sub_trim.is_empty() { j += 1; continue; }
                    let sub_indent = sub_line.chars().take_while(|c| c.is_whitespace()).count();
                    
                    if sub_indent <= cell_indent { break; }
                    
                    if let Some(idx) = sub_trim.find('|') {
                        text_acc.push_str(sub_trim[idx + 1..].trim());
                        text_acc.push(' ');
                    }
                    
                    line_indices.push(j);
                    j += 1;
                }
                
                cells.push(GridCell {
                    row: current_row, col: current_col, rowspan, colspan,
                    text: text_acc.trim().to_string(), line_indices
                });
                
                i = j;
                continue;
            }
        }
        i += 1;
    }
    cells
}
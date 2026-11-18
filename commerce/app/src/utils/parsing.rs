//! HTML → PUG 변환 · 레이아웃 · 스키마 필드는 crates/logis-pug 로 옮겼습니다. 예전 경로(crate::parsing · crate::utils::parsing)를
//! 그대로 쓰고, 토크나이저나 임베딩 모델(LogisModel)이 필요한 함수만 여기 남습니다.
pub use logis_pug::utils::parsing::*;

use crate::tokenizer;

pub fn truncate_pug_by_tokens(pug: &str, max_tokens: usize, tokenizer: &tokenizer::TokenizerModel, bottom_drop_tokens: Option<usize>) -> String {
    let mut lines: Vec<&str> = pug.lines().collect();
    if lines.is_empty() { return String::new(); }

    
    // 의미 있는 자식(input, option, td 등)을 품고 있는 구조적 부모(form, table, select 등)를 찾아내어
    // 절단기(Truncator)가 이 블록을 반토막 내지 못하도록 "Unbreakable Block"으로 묶어버립니다.
    #[derive(Clone, Copy)]
    struct Block { start: usize, end: usize }
    let mut unbreakable_blocks = Vec::new();
    let target_tags = ["form", "table", "ul", "ol", "dl", "fieldset"];
    let meaningful_children = ["input", "button", "textarea", "th", "td", "li", "dt", "dd", "a", "img", "label"];

    for i in 0..lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.is_empty() { continue; }
        
        let indent = lines[i].chars().take_while(|c| c.is_whitespace()).count();
        let tag_name = trimmed.split(|c| c == '[' || c == ' ' || c == '(').next().unwrap_or("").to_lowercase();
        
        if target_tags.contains(&tag_name.as_str()) {
            let mut end_idx = i;
            let mut has_meaningful = false;
            
            for j in (i + 1)..lines.len() {
                let child_line = lines[j];
                if child_line.trim().is_empty() { continue; }
                let child_indent = child_line.chars().take_while(|c| c.is_whitespace()).count();
                
                if child_indent <= indent {
                    break; // 부모의 들여쓰기와 같거나 작아지면 블록 종료
                }
                
                let child_tag = child_line.trim().split(|c| c == '[' || c == ' ' || c == '(').next().unwrap_or("").to_lowercase();
                if meaningful_children.contains(&child_tag.as_str()) {
                    has_meaningful = true;
                }
                end_idx = j;
            }
            
            // 유의미한 자식이 하나라도 있다면 이 구역은 절대 잘려선 안 되는 보호 구역으로 지정합니다.
            if has_meaningful {
                unbreakable_blocks.push(Block { start: i, end: end_idx });
            }
        }
    }

    // 중첩된 보호 구역(예: form 안에 table)들을 하나의 거대한 보호 구역으로 병합 매핑합니다.
    let mut block_of_line: Vec<Option<(usize, usize)>> = vec![None; lines.len()];
    for block in &unbreakable_blocks {
        for idx in block.start..=block.end {
            if let Some(existing) = block_of_line[idx] {
                let new_start = existing.0.min(block.start);
                let new_end = existing.1.max(block.end);
                for k in new_start..=new_end {
                    block_of_line[k] = Some((new_start, new_end));
                }
            } else {
                block_of_line[idx] = Some((block.start, block.end));
            }
        }
    }

    
    if let Some(drop_limit) = bottom_drop_tokens {
        let mut low = 0;
        let mut high = lines.len();
        let mut cut_idx = lines.len();

        // 1. 하단 버리기(bottom_drop) 이진 탐색
        while low <= high {
            let mid = low + (high - low) / 2;
            let bottom_part = lines[mid..].join("\n");
            let tokens = tokenizer.text_encode_vec(bottom_part, false).map(|v| v.len()).unwrap_or(0);

            if tokens > drop_limit {
                low = mid + 1;
            } else {
                cut_idx = mid;
                if mid == 0 { break; }
                high = mid - 1;
            }
        }
        
        if cut_idx < lines.len() {
            if let Some((_, b_end)) = block_of_line[cut_idx] {
                cut_idx = (b_end + 1).min(lines.len());
            }
        }

        // 문서가 너무 짧아 통째로 날아가는 것을 방지하기 위해 최소 1줄은 남깁니다.
        let safe_cut_idx = cut_idx.min(lines.len().saturating_sub(1));
        lines.truncate(safe_cut_idx);
    }

    let mut low = 0;
    let mut high = lines.len();
    let mut start_keep_idx = 0;

    // 2. 최대 토큰(max_tokens) 제한 이진 탐색
    while low <= high {
        let mid = low + (high - low) / 2;
        let part = lines[mid..].join("\n");
        let tokens = tokenizer.text_encode_vec(part, false).map(|v| v.len()).unwrap_or(0);

        if tokens > max_tokens {
            low = mid + 1;
        } else {
            start_keep_idx = mid;
            if mid == 0 { break; }
            high = mid - 1;
        }
    }
    
    if start_keep_idx < lines.len() && start_keep_idx > 0 {
        if let Some((b_start, _)) = block_of_line[start_keep_idx] {
            start_keep_idx = b_start;
        }
    }
    
    let mut final_kept_lines = Vec::new();
    let mut last_valid_indent = None;

    for i in start_keep_idx..lines.len() {
        final_kept_lines.push(format!("{}\n", lines[i]));
        if last_valid_indent.is_none() && !lines[i].trim().is_empty() {
            last_valid_indent = Some(lines[i].chars().take_while(|c| c.is_whitespace()).count());
        }
    }
    
    // 2. [복구 단계] 절단면 위쪽으로 거슬러 올라가며 필수 부모 껍데기 구출
    let mut extracted_title = None;
    // 🌟 [역추적 한계선 연결] 파일 상단 주석이 선언한 규칙(html/body/section/main 등을 만나면
    //    역추적 중단)이 실제로는 한 번도 호출되지 않아 is_root_layout_element 가 죽은 코드였습니다.
    //    한계선 위의 전역 뼈대는 토큰만 먹고 컨텍스트를 희석하므로 삽입을 멈춥니다.
    //    다만 title 은 head 안에 있어 body 보다 위쪽 인덱스에 있으므로 루프 자체는 계속 돌립니다.
    let mut root_reached = false;

    if let Some(mut target_indent) = last_valid_indent {
        for i in (0..start_keep_idx).rev() {
            let line = lines[i];
            let trimmed = line.trim();
            if trimmed.is_empty() { continue; }
            
            let current_indent = line.chars().take_while(|c| c.is_whitespace()).count();
            let tag_name = trimmed.split(|c| c == '[' || c == ' ' || c == '(').next().unwrap_or("").to_lowercase();
            
            if tag_name == "title" && extracted_title.is_none() {
                let mut title_block = format!("{}\n", line);
                if i + 1 < lines.len() && lines[i+1].trim().starts_with('|') {
                    title_block.push_str(&format!("{}\n", lines[i+1]));
                }
                extracted_title = Some(title_block);
            }
            if !root_reached && current_indent < target_indent && !is_void_element(line) {
                final_kept_lines.insert(0, format!("{}\n", line));
                target_indent = current_indent;
                if is_root_layout_element(line) { root_reached = true; }
            }
        }
    }
    
    if let Some(title_str) = extracted_title {
        final_kept_lines.insert(0, title_str);
    }
    
    // 3. [정렬 단계] 수집된 라인을 정방향으로 유지한 채 다이내믹 뎁스 정렬 수행
    if !final_kept_lines.is_empty() {
        let mut current_shift = final_kept_lines.iter()
            .find(|line| !line.trim().is_empty())
            .map(|line| line.chars().take_while(|c| c.is_whitespace()).count())
            .unwrap_or(0);
        
        for line in final_kept_lines.iter_mut() {
            if line.trim().is_empty() { continue; }
            let original_indent = line.chars().take_while(|c| c.is_whitespace()).count();
            
            if original_indent < current_shift {
                current_shift = original_indent;
            }
            
            let remove_count = current_shift.min(original_indent);
            *line = line.chars().skip(remove_count).collect();
        }
    }
    
    final_kept_lines.concat()
}

pub async fn canonicalize_trade_columns_with_embedding(
    missed_headers: &[(usize, String)],
    doc_lang: &str,
    model: &crate::model::LogisModel,
) -> Vec<(usize, String)> {
    if missed_headers.is_empty() {
        return Vec::new();
    }
    let header_texts: Vec<String> = missed_headers.iter().map(|(_, t)| t.clone()).collect();
    let header_embs = match model.get_embedding_batch(header_texts).await {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let mut field_names: Vec<String> = Vec::new();
    let mut field_embs: Vec<Vec<f32>> = Vec::new();
    for (field, aliases) in TRADE_COLUMN_ALIASES.iter() {
        let lang_aliases: Vec<String> = aliases
            .iter()
            .filter(|a| alias_lang_matches(a, doc_lang))
            .map(|a| a.to_string())
            .collect();
        if lang_aliases.is_empty() {
            continue;
        }
        let emb = match model.get_embedding(lang_aliases.join(", ")).await {
            Ok(e) => e,
            Err(_) => continue,
        };
        if emb.iter().all(|&v| v == 0.0) {
            continue;
        }
        field_names.push(field.to_string());
        field_embs.push(emb);
    }
    let mut results: Vec<(usize, String)> = Vec::new();
    for (ei, (orig_idx, _)) in missed_headers.iter().enumerate() {
        let h_emb = &header_embs[ei];
        if h_emb.iter().all(|&v| v == 0.0) {
            continue;
        }
        let mut best: f32 = -1.0;
        let mut second: f32 = -1.0;
        let mut best_field: String = String::new();
        for fi in 0..field_embs.len() {
            let sim = crate::utils::ai_utils::cosine_similarity(h_emb, &field_embs[fi]);
            if sim > best {
                second = best;
                best = sim;
                best_field = field_names[fi].clone();
            } else if sim > second {
                second = sim;
            }
        }
        if best >= 0.72 && (best - second) >= 0.03 && !best_field.is_empty() {
            println!(
                "   🔤 [EMBEDDING FALLBACK] '{}' → '{}' (cos {:.4}, margin {:+.4})",
                missed_headers[ei].1,
                best_field,
                best,
                best - second
            );
            results.push((*orig_idx, best_field));
        } else {
            println!(
                "   ⚪ [EMBEDDING FALLBACK SKIP] '{}' 최고 {:.4} / 마진 {:+.4} 로 확정 불가",
                missed_headers[ei].1,
                best,
                best - second
            );
        }
    }
    results
}

/// 🌟 [HEADER BAND v2 — ASYNC] 임베딩 폴백만 수행합니다.
///    `&Html` / `ElementRef` 가 이 함수에 전혀 등장하지 않으므로
///    `tokio::spawn` 내부에서 안전하게 `.await` 할 수 있습니다.
pub async fn extract_doc_table_headers_async(
    grid: Vec<Vec<String>>,
    pending_embedding: Vec<(usize, String)>,
    doc_lang: &str,
    model: &crate::model::LogisModel,
) -> Vec<Vec<String>> {
    if !pending_embedding.is_empty() {
        let emb_results = canonicalize_trade_columns_with_embedding(
            &pending_embedding,
            doc_lang,
            model,
        ).await;
        for (_idx, field) in emb_results {
            // 임베딩 폴백 결과는 alt= 주입 시 사용됩니다.
            // grid 는 이미 title 로 채워져 있으므로 여기서 별도 반영 불필요.
            let _ = field;
        }
    }
    grid
}

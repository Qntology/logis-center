use scraper::{Html, Node, Selector};
use ego_tree::NodeRef;
use regex::Regex;
use serde_json::Value;

pub use crate::prompts::*; // 🌟 분리된 프롬프트 함수들을 외부에서 그대로 사용할 수 있도록 재수출

use crate::tokenizer;

// 🌟 [호환성 재수출] 기존 crate::parsing:: 경로로 접근하던 코드를 위해
// 각 전용 모듈의 함수를 이 모듈에서 그대로 재수출합니다.
pub use crate::utils::bias_schema::{
    BIAS_DICT,
    get_localized_page_type,
    get_layout_bias,
    get_combinatorial_layout_bias,
    get_page_type_full_bias,
    get_page_type_classification_bias,
    get_title_bias,
    get_list_schema_fields,
    get_detail_schema_fields,
    get_vision_tracking_bias,
    get_layout_prompt_hints,
    get_multi_pass_contexts,
};
pub use crate::utils::json_parse::{
    sanitize_llm_input as other_sanitize_llm_input,
    normalize_to_json_string,
    parse_json_from_llm,
};
pub use crate::utils::nl_convert::json_to_natural_language;
pub use crate::utils::time_guide::get_deterministic_time_guide;

#[derive(PartialEq, Clone, Copy)]
pub enum PugMode {
    StructureOnly,
    FullContent,
    DetailMode,
    TheadMode,
    ListMode, 
    NoAttributesMode, 
}

pub fn sanitize_llm_input(text: &str) -> String {
    // 1. Filter out non-printable and non-ASCII/Korean characters if they are broken
    // But we want to keep Korean. Let's filter out known problematic control codes.
    let cleaned: String = text.chars()
        .filter(|c| {
            let u = *c as u32;
            // Keep: standard ASCII, Korean Hangul Jamo/Syllables, Common Punctuation
            (u >= 32 && u <= 126) || // Basic ASCII
            (u >= 0xAC00 && u <= 0xD7A3) || // Hangul Syllables
            (u >= 0x1100 && u <= 0x11FF) || // Hangul Jamo
            (u >= 0x3130 && u <= 0x318F) || // Hangul Compatibility Jamo
            u == 10 || u == 13 || u == 9     // \n, \r, \t
        })
        .collect();

    // 2. Prevent internal special tokens from being interpreted
    cleaned.replace("<|", "< |").replace("|>", "| >")
}

pub fn pre_clean_html(html: &str) -> String {
    // 1. 주석 제거
    let re_comm = Regex::new(r"(?s)<!--.*?-->").unwrap();
    let html = re_comm.replace_all(html, "");

    // 2. 불필요한 태그 및 내부 콘텐츠 통째로 제거
    // JS filter list: script, style, link, noscript, iframe, svg
    let re_tags = Regex::new(r"(?is)<(script|style|link|noscript|iframe|svg)\b[^>]*>.*?</(script|style|link|noscript|iframe|svg)>").unwrap();
    let html = re_tags.replace_all(&html, "");

    // 3. 단일 태그 및 불필요한 메타 태그 정리 (input은 제외하고 보존)
    let re_single = Regex::new(r"(?is)<(meta|link|br|hr|source)\b[^>]*>").unwrap();
    let clean = re_single.replace_all(&html, "");

    // 4. 허용된 속성 외 모두 제거 (지정된 16개 속성만 보존)
    let re_tag = Regex::new(r"(?i)<([a-zA-Z0-9\-]+)([^>]*)>").unwrap();
    
    // 🌟 [CRITICAL FIX] 정규식의 Alternation(|) 우선순위 버그 수정!
    // rows가 앞단에 있으면 rowspan을 만났을 때 rows 부분만 매칭되고 pan="2"가 잘려나가는 현상을 원천 방지하기 위해 긴 단어를 먼저 배치합니다.
    let re_attr = Regex::new(r#"(?i)\b(id|class|src|href|type|name|value|placeholder|checked|selected|disabled|readonly|rowspan|colspan|rows|cols|scope)(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+))?"#).unwrap();
    
    let clean = re_tag.replace_all(&clean, |caps: &regex::Captures| {
        let tag_name = &caps[1];
        let attrs_str = &caps[2];
        
        let mut keep_attrs = String::new();
        for attr_cap in re_attr.captures_iter(attrs_str) {
            keep_attrs.push(' ');
            keep_attrs.push_str(&attr_cap[0]);
        }
        
        if attrs_str.trim_end().ends_with('/') {
            keep_attrs.push_str(" /");
        }
        
        format!("<{}{}>", tag_name, keep_attrs)
    }).to_string();

    // 5. 연속된 줄바꿈 및 불필요한 공백 제거
    let re_whitespace = Regex::new(r"(?m)^\s*\n").unwrap();
    let clean = re_whitespace.replace_all(&clean, "");
    
    clean.trim().to_string()
}


pub fn convert_doc_to_clean_pug(document: &Html, mode: PugMode, base_url: Option<&str>) -> String {
    let mut pug_output = String::new();
    pug_output.reserve(1024 * 50);
    
    
    let mut ctx = Some(TableContext {
        base_url: base_url.map(|s| s.to_string()),
        ..Default::default()
    });

    // Discovery 모드(StructureOnly)일 때는 body 내부만 집중
    let mut found_body = false;
    for child in document.tree.root().children() {
        if let Some(element) = child.value().as_element() {
            if element.name() == "body" {
                generate_pug_lines(child, 0, &mut pug_output, &mode, &mut ctx);
                found_body = true;
                break;
            }
        }
    }
    if !found_body {
        for child in document.tree.root().children() {
            generate_pug_lines(child, 0, &mut pug_output, &mode, &mut ctx);
        }
    }
    sanitize_llm_input(&pug_output)
}
    
pub fn convert_to_clean_pug(html: &str, mode: PugMode, base_url: Option<&str>) -> String {
    let document = Html::parse_document(html);
    convert_doc_to_clean_pug(&document, mode, base_url)
}

pub fn convert_doc_to_clean_pug_selector(document: &Html, selector_str: &str, mode: PugMode, base_url: Option<&str>) -> String {
    let selector = match Selector::parse(selector_str) {
        Ok(s) => s,
        Err(_) => return String::new(),
    };
    let mut pug_output = String::new();
    pug_output.reserve(1024 * 5);
    
    let mut ctx = Some(TableContext {
        base_url: base_url.map(|s| s.to_string()),
        ..Default::default()
    });
    
    for node in document.tree.root().descendants() {
        if let Some(element_ref) = scraper::ElementRef::wrap(node) {
            if selector.matches(&element_ref) {
                 generate_pug_lines(node, 0, &mut pug_output, &mode, &mut ctx);
                 break;
            }
        }
    }
    pug_output
}

pub fn convert_to_clean_pug_selector(html: &str, selector_str: &str, mode: PugMode, base_url: Option<&str>) -> String {
    let document = Html::parse_document(html);
    convert_doc_to_clean_pug_selector(&document, selector_str, mode, base_url)
}


// 1. Void Elements (자식 불가 태그): area, base, br, col, embed, hr, img, input, link, meta, param, source, track, wbr 및 PUG 텍스트(|)
// 2. Root/Layout Elements (역추적 한계선): html, body, head, main, section, article, aside, nav, header, footer
// 3. Container Elements (합법적 부모): 위 1번과 2번을 제외한 모든 태그 (div, table, ul, li, span, p 등)
// 이 원칙을 바탕으로 잘려나간 PUG의 잃어버린 뎁스(부모 껍데기)를 역추적하여 100% 복구하고 불필요한 전체 뼈대는 버립니다.

/// HTML 명세상 자식을 가질 수 없는 단일 태그(Void Elements) 판별
fn is_void_element(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() { return true; }
    
    // 1. PUG 텍스트 노드(|)는 부모가 될 수 없음
    if trimmed.starts_with('|') { return true; }
    
    // 2. HTML Void Elements 14개 리스트 완벽 적용
    let void_tags = [
        "area", "base", "br", "col", "embed", "hr", "img", "input", 
        "link", "meta", "param", "source", "track", "wbr"
    ];
    
    // 태그 이름만 추출 (예: "img[src='...']" -> "img")
    let tag_name = trimmed.split(|c| c == '[' || c == ' ' || c == '(').next().unwrap_or("").to_lowercase();
    
    void_tags.contains(&tag_name.as_str())
}

/// 문서의 최상위 골격을 형성하는 레이아웃 태그 판별 (만나면 역추적 중단)
fn is_root_layout_element(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('|') { return false; }
    
    // 태그 이름만 정확히 추출
    let tag_name = trimmed.split(|c| c == '[' || c == ' ' || c == '(').next().unwrap_or("").to_lowercase();
    
    // HTML5 시맨틱 레이아웃/구역 경계 태그 10개 리스트
    let root_tags = [
        "html", "body", "head", 
        "main", "section", "article", "aside", "nav", 
        "header", "footer"
    ];
    
    root_tags.contains(&tag_name.as_str())
}


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

            if current_indent < target_indent && !is_void_element(line) {
                final_kept_lines.insert(0, format!("{}\n", line));
                target_indent = current_indent; 
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

#[derive(Default, Clone)]
pub struct TableContext {
    pub headers: Vec<Vec<String>>, // Row -> Col -> Title
    pub current_row_idx: usize,
    pub current_col_idx: usize,
    pub is_in_tbody: bool,
    pub base_url: Option<String>, 
}

// 🌟 [STRUCTURE GUARD] el_ref.text() 는 '텍스트 노드'만 수집하므로
//    <tr><th>이름</th><td><input value="세글만"></td></tr> 같은 폼 행은
//    trim() 이후 "이름" 한 덩어리가 되어 인라인 병합 조건을 통과해 버립니다.
//    그 순간 td 와 input[value] 는 출력조차 되지 않고 문서에서 영구 소멸합니다.
//    (로그의 'tr | 이름', 'tr | 핸드폰', 'tr | E-mail' 이 전부 이 경로입니다)
//    따라서 "텍스트로 환원되지 않는 데이터"를 자손으로 가진 노드는
//    절대 한 줄로 압축하지 않고 반드시 자식까지 펼쳐서 출력합니다.
fn has_data_bearing_descendant(node: NodeRef<scraper::Node>) -> bool {
    for desc in node.descendants() {
        if desc.id() == node.id() { continue; }
        if let Some(el) = desc.value().as_element() {
            let t = el.name().to_lowercase();
            // 셀/폼컨트롤/미디어는 그 자체가 독립된 데이터 단위입니다.
            if ["tr", "td", "th", "input", "select", "option", "textarea", "button", "img"]
                .contains(&t.as_str())
            {
                return true;
            }
            // 텍스트가 아닌 속성에 값이 실려 있는 노드(링크/리소스/폼값)
            if el.attr("href").map_or(false, |v| !v.trim().is_empty()) { return true; }
            if el.attr("src").map_or(false, |v| !v.trim().is_empty()) { return true; }
            if el.attr("value").map_or(false, |v| !v.trim().is_empty()) { return true; }
        }
    }
    false
}

pub fn generate_pug_lines(node: NodeRef<scraper::Node>, indent_level: usize, output: &mut String, mode: &PugMode, ctx: &mut Option<TableContext>) {
    if indent_level > 50 { return; }
    let indent = "    ".repeat(indent_level);
    
    match node.value() {
        Node::Element(element) => {
            let tag_name = element.name().to_lowercase();

            if let Some(style) = element.attr("style") {
                let style_lower = style.to_lowercase();
                if style_lower.contains("position") && 
                   (style_lower.contains("absolute") || style_lower.contains("fixed")) 
                {
                    return;
                }
            }

            // --- base64 이미지를 포함하는 img 태그 제외 ---
            if tag_name == "img" {
                if let Some(src) = element.attr("src") {
                    if src.contains("base64") {
                        return;
                    }
                }
            }

            // 불필요한 태그들을 만나면 건너뛰기 (svg 추가)
            if ["script", "style", "link", "noscript", "iframe", "svg"].contains(&tag_name.as_str()) {
                return;
            }

            
            if tag_name == "option" && !element.attrs().any(|(k, _)| k.to_lowercase() == "selected") {
                return;
            }

            
            if *mode == PugMode::NoAttributesMode {
                if ["select", "datalist", "option"].contains(&tag_name.as_str()) {
                    return;
                }
            }

            // Context Management
            if tag_name == "tbody" { if let Some(c) = ctx.as_mut() { c.is_in_tbody = true; c.current_row_idx = 0; } }
            if tag_name == "tr" { if let Some(c) = ctx.as_mut() { c.current_col_idx = 0; } }

            
            // 껍데기 태그 자체가 출력되지 않고 자식에게 뎁스(indent)를 그대로 패스합니다.
            let useless_wrappers = [
                "div", "span", "section", "article", "main", "aside", 
                "header", "footer", "nav", "p", "strong", "b", "em", "i", "center", "font"
            ];
            
            let is_useless = useless_wrappers.contains(&tag_name.as_str());
            
            let has_meaningful_attrs = if *mode == PugMode::NoAttributesMode {
                
                element.attrs().any(|(k, _)| ["colspan", "rowspan", "scope"].contains(&k.to_lowercase().as_str()))
            } else {
                element.attrs().any(|(k, _)| {
                    let k_lower = k.to_lowercase();
                    ["src", "href", "type", "name", "value", "placeholder", "checked", "selected", "disabled", "readonly", "rows", "cols", "rowspan", "colspan", "scope"].contains(&k_lower.as_str()) || k_lower.starts_with("data-")
                })
            };

            
            let valid_children: Vec<_> = node.children().filter(|n| {
                match n.value() {
                    Node::Element(_) => {
                        let void_tags = ["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"];
                        let preserve_empty = ["td", "th", "textarea", "select", "button"];
                        
                        // 하위에 텍스트나 필수 보존 태그가 단 하나라도 존재하는지 확인 (빈 껍데기 필터링)
                        n.descendants().any(|desc| match desc.value() {
                            Node::Text(t) => !t.trim().is_empty(),
                            Node::Element(de) => {
                                let d_tag = de.name().to_lowercase();
                                void_tags.contains(&d_tag.as_str()) || preserve_empty.contains(&d_tag.as_str())
                            },
                            _ => false
                        })
                    },
                    Node::Text(t) => !t.trim().is_empty(),
                    _ => false
                }
            }).collect();

            let void_tags = ["area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"];
            let preserve_empty_tags = ["td", "th", "textarea", "select", "button"]; // 폼이나 표의 구조적 형태 유지를 위해 빈 셀/입력창은 예외적으로 보존
            
            
            if valid_children.is_empty() && !void_tags.contains(&tag_name.as_str()) && !preserve_empty_tags.contains(&tag_name.as_str()) {
                return;
            }

            // 현재 태그가 무의미한 껍데기이고 유효한 자식이 딱 1개라면, 자신을 숨기고 뎁스 유지
            if is_useless && !has_meaningful_attrs && valid_children.len() == 1 {
                for child in node.children() {
                    generate_pug_lines(child, indent_level, output, mode, ctx);
                }
                return;
            }

            // --- 허용된 속성만 Pug 문법으로 변환 ---
            let mut other_attributes = Vec::new();

            // ID 속성 처리 (DetailMode, TheadMode, ListMode, NoAttributesMode 가 아닐 때만 유지)
            if *mode != PugMode::DetailMode && *mode != PugMode::TheadMode && *mode != PugMode::ListMode && *mode != PugMode::NoAttributesMode {
                if let Some(id) = element.id() {
                    other_attributes.push(format!("id=\"{}\"", id));
                }
            }

            // Class 속성 처리 (DetailMode, TheadMode, ListMode, NoAttributesMode 가 아닐 때만 유지)
            if *mode != PugMode::DetailMode && *mode != PugMode::TheadMode && *mode != PugMode::ListMode && *mode != PugMode::NoAttributesMode {
                if let Some(classes) = element.attr("class") {
                    if !classes.is_empty() {
                        other_attributes.push(format!("class=\"{}\"", classes));
                    }
                }
            }

            // Inject alt from headers for tbody cells
            if tag_name == "td" || tag_name == "th" {
                if let Some(c) = ctx.as_mut() {
                    if c.is_in_tbody && !c.headers.is_empty() {
                        let h_row = &c.headers[c.current_row_idx % c.headers.len()];
                        if let Some(title) = h_row.get(c.current_col_idx) {
                            if !title.is_empty() {
                                other_attributes.push(format!("alt=\"{}\"", title.replace("\"", "'")));
                            }
                        }
                    }
                }
            }

            // 필수 속성 정의
            let always_include = [
                "src", "href", "type", "name", "value", "placeholder", 
                "checked", "selected", "disabled", "readonly", "rows", "cols", "rowspan", "colspan", "scope"
            ];
            let thead_include = ["scope", "rowspan", "colspan"];

            for (name, value) in element.attrs() {
                
                let name_lower = name.to_lowercase();
                let name_str = name_lower.as_str();

                if name_str == "id" || name_str == "class" || name_str == "alt" { continue; }

                
                let should_include = ["colspan", "rowspan", "scope"].contains(&name_str) || if *mode == PugMode::TheadMode {
                    thead_include.contains(&name_str)
                } else if *mode == PugMode::NoAttributesMode {
                    false
                } else {
                    name_str.starts_with("data-") || always_include.contains(&name_str)
                };

                if should_include {
                    if ["checked", "selected", "disabled", "readonly"].contains(&name_str) && (value.is_empty() || value == name) {
                        other_attributes.push(name_str.to_string());
                    } else if !value.is_empty() {
                        let mut safe_value = value.replace("\"", "'");
                        
                        if name_str == "href" || name_str == "src" {
                            if let Some(c) = ctx.as_ref() {
                                if let Some(base) = &c.base_url {
                                    if let Ok(base_url_obj) = url::Url::parse(base) {
                                        if let Ok(resolved_url) = base_url_obj.join(safe_value.trim()) {
                                            safe_value = resolved_url.to_string();
                                        }
                                    }
                                }
                            }
                        }

                        other_attributes.push(format!("{}=\"{}\"", name_str, safe_value));
                    }
                }
            }

            let mut attributes_string = String::new();
            if !other_attributes.is_empty() {
                attributes_string.push_str(&format!("[{}]", other_attributes.join(" ")));
            }

            let should_output_text = *mode == PugMode::FullContent || *mode == PugMode::DetailMode || *mode == PugMode::TheadMode || *mode == PugMode::ListMode || *mode == PugMode::NoAttributesMode;
            
            let mut is_inline_text = false;
            let mut inline_content = String::new();

            // 🌟 [CRITICAL FIX] 텍스트만 포함하는 태그(td, th, label, span 등)를 한 줄로 병합하여 PUG 컨텍스트 밀도를 높입니다.
            //    단, 자손에 '텍스트로 환원되지 않는 데이터'(input[value], option, a[href], img[src], td/th)가 존재하면
            //    병합 시 해당 데이터가 출력조차 되지 않고 소멸하므로 반드시 펼쳐서 출력합니다.
            if should_output_text {
                if tag_name == "input" {
                    if let Some(val) = element.attr("value") {
                        let trimmed = val.trim();
                        if !trimmed.is_empty() && !trimmed.contains('\n') {
                            inline_content = trimmed.to_string();
                            is_inline_text = true;
                        }
                    }
                } else if tag_name == "textarea" {
                    let mut text_buf = String::new();
                    for child in node.children() {
                        if let Node::Text(t) = child.value() { text_buf.push_str(t); }
                    }
                    let clean = text_buf.trim();
                    if !clean.is_empty() && !clean.contains('\n') {
                        inline_content = clean.to_string();
                        is_inline_text = true;
                    }
                } else if !has_data_bearing_descendant(node) {
                    if let Some(el_ref) = scraper::ElementRef::wrap(node) {
                        // 🌟 [요구사항 완벽 반영] 하드코딩된 태그 리스트 검사를 완전히 삭제했습니다!
                        // 어떤 태그이든 상관없이 모든 하위 텍스트를 긁어와 병합 검사를 무조건 실행합니다.
                        let text_buf = el_ref.text().collect::<Vec<_>>().join(" ");
                        let clean = text_buf.trim();
                        
                        // 텍스트가 존재하고, 줄바꿈이 없으며, 너무 길지 않은(150자 이내) 경우에만 인라인 압축을 허용합니다.
                        if !clean.is_empty() && !clean.contains('\n') && clean.len() < 150 {
                            let mut clean_text = clean.replace("\"", "'").replace("  ", " ");
                            if let Ok(re) = regex::Regex::new(r"(\d{1,3}(?:,\d{3})+)(\.\d+)?") {
                                clean_text = re.replace_all(&clean_text, |caps: &regex::Captures| {
                                    let int_part = caps.get(1).map_or("", |m| m.as_str()).replace(",", "");
                                    let dec_part = caps.get(2).map_or("", |m| m.as_str());
                                    format!("{}{}", int_part, dec_part)
                                }).to_string();
                            }
                            inline_content = clean_text;
                            is_inline_text = true;
                        }
                    }
                }
            }

            if is_inline_text {
                // 태그 껍데기와 텍스트를 파이프(|) 기호와 함께 한 줄로 압축합니다. (예: td | 무통장)
                output.push_str(&format!("{}{}{} | {}\n", indent, tag_name, attributes_string, inline_content));
            } else {
                output.push_str(&format!("{}{}{}\n", indent, tag_name, attributes_string));

                if tag_name == "textarea" {
                    let mut text_content = String::new();
                    for child in node.children() {
                        if let Node::Text(t) = child.value() { text_content.push_str(t); }
                    }
                    if !text_content.trim().is_empty() {
                        for line in text_content.lines() {
                            let trimmed = line.trim();
                            if !trimmed.is_empty() {
                                output.push_str(&format!("{}    | {}\n", indent, trimmed));
                            }
                        }
                    }
                } else if tag_name == "input" {
                    if let Some(val) = element.attr("value") {
                        let trimmed = val.trim();
                        if !trimmed.is_empty() {
                            for line in trimmed.lines() {
                                let t_line = line.trim();
                                if !t_line.is_empty() {
                                    output.push_str(&format!("{}    | {}\n", indent, t_line));
                                }
                            }
                        }
                    }
                } else {
                    for child in node.children() {
                        generate_pug_lines(child, indent_level + 1, output, mode, ctx);
                    }
                }
            }

            // End of Tag Updates
            if tag_name == "tr" { if let Some(c) = ctx.as_mut() { if c.is_in_tbody { c.current_row_idx += 1; } } }
            if tag_name == "td" || tag_name == "th" { if let Some(c) = ctx.as_mut() { if c.is_in_tbody { c.current_col_idx += 1; } } }
            if tag_name == "tbody" { if let Some(c) = ctx.as_mut() { c.is_in_tbody = false; } }
        }
        Node::Text(text) => {
            if *mode == PugMode::FullContent || *mode == PugMode::DetailMode || *mode == PugMode::TheadMode || *mode == PugMode::ListMode || *mode == PugMode::NoAttributesMode {
                let text_content = text.trim();
                if !text_content.is_empty() {
                    
                    let mut clean_text = text_content.replace("\"", "'");
                    
                    // 정규식: 1~3자리 숫자 뒤에 (콤마 + 3자리 숫자)가 1번 이상 반복되고, 선택적으로 소수점이 붙는 패턴
                    if let Ok(re) = regex::Regex::new(r"(\d{1,3}(?:,\d{3})+)(\.\d+)?") {
                        clean_text = re.replace_all(&clean_text, |caps: &regex::Captures| {
                            let int_part = caps.get(1).map_or("", |m| m.as_str()).replace(",", "");
                            let dec_part = caps.get(2).map_or("", |m| m.as_str());
                            format!("{}{}", int_part, dec_part)
                        }).to_string();
                    }
                    
                    output.push_str(&format!("{}| {}\n", indent, clean_text));
                }
            }
        }
        _ => {}
    }
}

pub fn split_doc_to_pug_list(document: &Html, selector_str: &str, mode: PugMode) -> Vec<String> {
    split_doc_to_pug_list_advanced(document, selector_str, mode, None, None)
}

pub fn split_doc_to_pug_list_advanced(document: &Html, selector_str: &str, mode: PugMode, headers: Option<Vec<Vec<String>>>, base_url: Option<&str>) -> Vec<String> {
    let selector = match Selector::parse(selector_str) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let mut pug_list = Vec::new();
    
    // rowspan으로 묶인 다음 행들을 병합하기 위한 버퍼와 카운터
    let mut skip_next_n_rows = 0;
    let mut combined_pug_buffer = String::new();

    for node in document.tree.root().descendants() {
        if let Some(element_ref) = scraper::ElementRef::wrap(node) {
            if selector.matches(&element_ref) {
                
                let mut pug_output = String::new();
                pug_output.reserve(2048);
                
                
                let mut ctx = Some(TableContext {
                    headers: headers.clone().unwrap_or_default(),
                    is_in_tbody: true,
                    base_url: base_url.map(|s| s.to_string()),
                    ..Default::default()
                });
                
                // 현재 노드의 PUG 라인 생성
                generate_pug_lines(node, 0, &mut pug_output, &mode, &mut ctx);

                // 개별로 바로 push 하지 않고 rowspan 검사
                if !pug_output.trim().is_empty() {
                    // 현재 노드 내부에 rowspan 속성이 있는지 확인
                    let mut current_rowspan = 1;
                    if let Ok(td_selector) = scraper::Selector::parse("td, th") {
                        for cell in element_ref.select(&td_selector) {
                            if let Some(span_str) = cell.value().attr("rowspan") {
                                if let Ok(span) = span_str.parse::<usize>() {
                                    if span > current_rowspan {
                                        current_rowspan = span;
                                    }
                                }
                            }
                        }
                    }

                    if skip_next_n_rows > 0 {
                        // 이전 행에 rowspan이 있어서 현재 행을 합쳐야 하는 경우
                        combined_pug_buffer.push_str(&pug_output);
                        skip_next_n_rows -= 1;
                        
                        // 대기 중인 행을 다 합쳤다면 최종 리스트에 추가
                        if skip_next_n_rows == 0 {
                            pug_list.push(combined_pug_buffer.clone());
                            combined_pug_buffer.clear();
                        }
                    } else if current_rowspan > 1 {
                        // 새로운 rowspan 시작 지점
                        combined_pug_buffer.push_str(&pug_output);
                        skip_next_n_rows = current_rowspan - 1;
                    } else {
                        // 평범한 단일 행
                        pug_list.push(pug_output);
                    }
                }
            }
        }
    }
    
    // 혹시 버퍼에 남은 게 있다면 털어줌
    if !combined_pug_buffer.is_empty() {
        pug_list.push(combined_pug_buffer);
    }
    
    pug_list
}

pub fn extract_table_headers(html: &str, table_selector: &str) -> Vec<Vec<String>> {
    let document = Html::parse_document(html);
    let mut all_headers = Vec::new();
    
    if let Ok(sel) = Selector::parse(table_selector) {
        if let Some(first_match) = document.select(&sel).next() {
            let mut current = first_match.parent();
            while let Some(parent) = current {
                if let Some(el) = parent.value().as_element() {
                    if el.name() == "table" {
                        if let Some(table_ref) = scraper::ElementRef::wrap(parent) {
                            if let Ok(thead_sel) = Selector::parse("thead") {
                                if let Some(thead) = table_ref.select(&thead_sel).next() {
                                    if let Ok(tr_sel) = Selector::parse("tr") {
                                        for tr in thead.select(&tr_sel) {
                                            let mut row_headers = Vec::new();
                                            if let Ok(cell_sel) = Selector::parse("th, td") {
                                                for cell in tr.select(&cell_sel) {
                                                    row_headers.push(cell.text().collect::<Vec<_>>().join(" ").trim().to_string());
                                                }
                                            }
                                            if !row_headers.is_empty() { all_headers.push(row_headers); }
                                        }
                                    }
                                }
                            }
                        }
                        break;
                    }
                }
                current = parent.parent();
            }
        }
    }
    all_headers
}

pub fn split_html_to_pug_list(html: &str, selector_str: &str, mode: PugMode) -> Vec<String> {
    let document = Html::parse_document(html);
    split_doc_to_pug_list(&document, selector_str, mode)
}

/// 🌟 [TRADE SLICE CONFIG] 무역 서식별 크롭 좌표.
///  ── 출처 ──
///   app-logis-center/rust/src-tauri/src/model.rs 의 get_slice_config 를 이식했습니다.
///   원본은 Mission 구조체를 돌려주지만, cron 의 extract_from_image 가
///   (카테고리, top, bottom) 튜플을 기대하므로 형태만 맞췄습니다.
///
///  ── 왜 이식하는가 ──
///   cron 은 CI/PI/BL/AWB 4종 + 폴백 1개만 갖고 있었고,
///   폴백에는 cargo / financials / logistics 슬라이스가 없어
///   LC·ED·ID·PHYTO·DGD 등 22종에서 화물·금액·운송 정보가
///   구조적으로 추출되지 않았습니다.
pub fn get_trade_doc_slice_config(doc_type: &str) -> Vec<(&'static str, f32, f32)> {
    match doc_type {
        // --- 1. 계약 · 결제 ---
        "CI" | "PI" | "SC" => vec![
            ("header",     0.00, 0.25),
            ("parties",    0.00, 0.40),
            ("logistics",  0.20, 0.50),
            ("items",      0.30, 0.70),
            ("items",      0.50, 0.85),
            ("financials", 0.70, 0.95),
            ("conditions", 0.80, 1.00),
        ],
        "PO" => vec![
            ("header",     0.00, 0.25),
            ("parties",    0.00, 0.40),
            ("logistics",  0.20, 0.50),
            ("items",      0.30, 0.80),
            ("financials", 0.70, 0.95),
            ("conditions", 0.80, 1.00),
        ],
        "LC" => vec![
            ("header",     0.00, 0.30),
            ("parties",    0.00, 0.40),
            ("financials", 0.20, 0.60), // L/C 는 금융 조항 밀도가 가장 높습니다
            ("logistics",  0.40, 0.70),
            ("conditions", 0.50, 1.00), // 본문 대부분이 조건절입니다
        ],

        // --- 2. 선적 · 운송 ---
        "PL" | "SA" => vec![
            ("header",     0.00, 0.20),
            ("parties",    0.00, 0.40),
            ("logistics",  0.20, 0.50),
            ("items",      0.30, 0.80),
            ("cargo",      0.60, 0.95),
            ("conditions", 0.85, 1.00),
        ],
        "BL" => vec![
            ("header",     0.00, 0.20),
            ("parties",    0.00, 0.60),
            ("logistics",  0.35, 0.65),
            ("cargo",      0.50, 0.90),
            ("conditions", 0.80, 1.00),
        ],
        "AWB" => vec![
            ("header",     0.00, 0.15),
            ("parties",    0.00, 0.40),
            ("logistics",  0.10, 0.40),
            ("cargo",      0.30, 0.70),
            ("financials", 0.60, 0.90),
            ("conditions", 0.85, 1.00),
        ],
        "BC" => vec![
            ("header",    0.00, 0.25),
            ("parties",   0.00, 0.50),
            ("logistics", 0.30, 0.70),
            ("cargo",     0.50, 0.90),
        ],
        "AN" | "DO" => vec![
            ("header",     0.00, 0.25),
            ("parties",    0.00, 0.50),
            ("logistics",  0.30, 0.70),
            ("financials", 0.50, 0.90), // Arrival Notice 는 로컬 charge 가 핵심
            ("cargo",      0.60, 1.00),
        ],

        // --- 3. 통관 · 신고 ---
        "ED" | "ID" | "CINV" => vec![
            ("header",     0.00, 0.20),
            ("parties",    0.00, 0.30),
            ("logistics",  0.20, 0.50),
            ("financials", 0.40, 0.70),
            ("items",      0.50, 0.90),
            ("conditions", 0.80, 1.00),
        ],
        "CO" => vec![
            ("header",     0.00, 0.20),
            ("parties",    0.00, 0.40),
            ("logistics",  0.30, 0.50),
            ("items",      0.40, 0.80),
            ("conditions", 0.75, 1.00),
        ],

        // --- 4. 검사 · 증명 ---
        "IC" | "WC" | "CA" | "PHYTO" | "HC" | "BEN_CERT" => vec![
            ("header",     0.00, 0.25),
            ("parties",    0.00, 0.40),
            ("items",      0.30, 0.80), // 시험 항목 / 학명 리스트
            ("conditions", 0.70, 1.00), // "We hereby certify..." 선언문
        ],

        // --- 5. 특수 · 법무 ---
        "DGD" | "MSDS" => vec![
            ("header",    0.00, 0.25),
            ("logistics", 0.20, 0.50),
            ("cargo",     0.40, 0.90), // 위험물은 화물 속성이 본문입니다
        ],
        "POA" | "BIZ_LIC" | "INS" => vec![
            ("header",     0.00, 0.30),
            ("parties",    0.10, 0.50),
            ("conditions", 0.40, 1.00), // 법률 문언
        ],

        // --- 폴백 ---
        _ => vec![
            ("header",     0.00, 0.30),
            ("parties",    0.00, 0.50),
            ("items",      0.30, 0.80),
            ("conditions", 0.70, 1.00),
        ],
    }
}
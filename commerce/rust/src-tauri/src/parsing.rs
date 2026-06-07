use scraper::{Html, Node, Selector};
use ego_tree::NodeRef;
use regex::Regex;
use once_cell::sync::Lazy;
use serde_json::Value;

// 🌟 [다국어 지원] 빌드 시점에 bias.json 파일을 읽어와 메모리에 영구 등재합니다.
pub static BIAS_DICT: Lazy<Value> = Lazy::new(|| {
    let json_str = include_str!("bias.json");
    serde_json::from_str(json_str).unwrap_or(serde_json::json!({}))
});

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
    let re_comm = Regex::new(r"(?s)").unwrap();
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


pub fn truncate_pug_by_tokens(pug: &str, max_tokens: usize, tokenizer: &crate::tokenizer::TokenizerModel, bottom_drop_tokens: Option<usize>) -> String {
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

            // 🌟 [CRITICAL FIX] 텍스트만 포함하는 태그(td, th, label, span, input 등)를 한 줄로 병합하여 PUG 컨텍스트 밀도를 비약적으로 높입니다.
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
                } else {
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

pub fn page_type_prompt() -> String { r###"[TASK]
Based on the provided Pug template, identify the primary category of this webpage and its main language.

[SCHEMA DEFINITIONS]
- type: The main category. Must be one of:
  - "order": Order list, Order history, Order details, Checkout success.
  - "goods": Product list, product detail.
  - "tracking": Shipment tracking status, delivery history.
  - "review": Product reviews, feedback list.
  - "coupon": Coupon list, discount events.
  - "event": Promotion pages, event announcements.
  - "": If none of the above match.
- language: ISO 639-1 language code.

[OUTPUT FORMAT]
{
    "type": "String",
    "language": "String"
}"###.to_string() }

pub fn extract_titles_prompt(page_type: &str) -> String {
    let (category_desc, titles_desc, title_desc) = match page_type {
        "goods" => ("product", "product titles", "product title"),
        "order" => ("product", "order product titles", "order product title"),
        "tracking" => ("product", "tracking product titles", "tracking product title"),
        "review" => ("title", "review titles", "review title"),
        "coupon" => ("title", "coupon titles", "coupon title"),
        "event" => ("title", "event titles", "event title"),
        _ => ("title", "titles", "title"),
    };


    let template = r###"[TASK]
Find all the {TITLES} from the following PUG/HTML content.

[SCHEMA DEFINITIONS]
{ 
  {CATEGORY} : ["{TITLE}"]
}

[OUTPUT FORMAT]
{ {CATEGORY} : [...] }

RETURN JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{CATEGORY}", category_desc)
            .replace("{TITLES}", titles_desc)
            .replace("{TITLE}", title_desc)
            .replace("{TYPE}", page_type)
}

pub fn get_localized_page_type(page_type: &str, lang: &str) -> String {
    let lang_lower = lang.to_lowercase();
    
    // zh-tw와 zh-hk 같은 번체자 환경을 정확하게 분리하여 추출
    let lang_code = if lang_lower.starts_with("zh-tw") || lang_lower.starts_with("zh-hk") {
        "zh-tw"
    } else if lang_lower.len() >= 2 {
        &lang_lower[0..2]
    } else {
        "en"
    };

    let matched_str = match lang_code {
        "ko" => match page_type { "order" => "주문", "goods" => "상품", "tracking" => "배송", "review" => "리뷰", "coupon" | "event" => "이벤트", _ => "문서" },
        "zh-tw" => match page_type { "order" => "訂單", "goods" => "商品", "tracking" => "物流", "review" => "評價", "coupon" | "event" => "活動", _ => "文件" },
        "zh" => match page_type { "order" => "订单", "goods" => "商品", "tracking" => "物流", "review" => "评价", "coupon" | "event" => "活动", _ => "文档" },
        "ja" => match page_type { "order" => "注文", "goods" => "商品", "tracking" => "配送", "review" => "レビュー", "coupon" | "event" => "イベント", _ => "ドキュメント" },
        "es" => match page_type { "order" => "pedido", "goods" => "producto", "tracking" => "seguimiento", "review" => "reseña", "coupon" | "event" => "evento", _ => "documento" },
        "pt" => match page_type { "order" => "pedido", "goods" => "produto", "tracking" => "rastreamento", "review" => "avaliação", "coupon" | "event" => "evento", _ => "documento" },
        "de" => match page_type { "order" => "bestellung", "goods" => "produkt", "tracking" => "sendungsverfolgung", "review" => "bewertung", "coupon" | "event" => "event", _ => "dokument" },
        "nl" => match page_type { "order" => "bestelling", "goods" => "product", "tracking" => "tracking", "review" => "beoordeling", "coupon" | "event" => "evenement", _ => "document" },
        "it" => match page_type { "order" => "ordine", "goods" => "prodotto", "tracking" => "tracciamento", "review" => "recensione", "coupon" | "event" => "evento", _ => "documento" },
        "id" | "ms" => match page_type { "order" => "pesanan", "goods" => "produk", "tracking" => "pelacakan", "review" => "ulasan", "coupon" | "event" => "acara", _ => "dokumen" },
        "vi" => match page_type { "order" => "đơn hàng", "goods" => "sản phẩm", "tracking" => "theo dõi", "review" => "đánh giá", "coupon" | "event" => "sự kiện", _ => "tài liệu" },
        "th" => match page_type { "order" => "คำสั่งซื้อ", "goods" => "สินค้า", "tracking" => "การติดตาม", "review" => "รีวิว", "coupon" | "event" => "กิจกรรม", _ => "เอกสาร" },
        "ar" => match page_type { "order" => "طلب", "goods" => "منتج", "tracking" => "تتبع", "review" => "مراجعة", "coupon" | "event" => "حدث", _ => "مستند" },
        "ta" => match page_type { "order" => "ஆர்டர்", "goods" => "தயாரிப்பு", "tracking" => "கண்காணிப்பு", "review" => "விமர்சனம்", "coupon" | "event" => "நிகழ்வு", _ => "ஆவணம்" },
        "te" => match page_type { "order" => "ఆర్డర్", "goods" => "ఉత్పత్తి", "tracking" => "ట్రాకింగ్", "review" => "సమీక్ష", "coupon" | "event" => "ఈవెంట్", _ => "పత్రం" },
        "kn" => match page_type { "order" => "ಆರ್ಡರ್", "goods" => "ಉತ್ಪನ್ನ", "tracking" => "ಟ್ರ್ಯಾಕಿಂಗ್", "review" => "ವಿಮರ್ಶೆ", "coupon" | "event" => "ಈವೆಂಟ್", _ => "ಡಾಕ್ಯುಮೆಂಟ್" },
        "gu" => match page_type { "order" => "ઓર્ડર", "goods" => "ઉત્પાદન", "tracking" => "ટ્રેકિંગ", "review" => "સમીક્ષા", "coupon" | "event" => "ઇવેન્ટ", _ => "દસ્તાવેજ" },
        _ => match page_type { "order" => "order", "goods" => "product", "tracking" => "tracking", "review" => "review", "coupon" | "event" => "event", _ => "document" },
    };
    
    // 최종 결과물에 단 한번만 .to_string()을 호출하여 타입 불일치 에러 완벽 해결
    matched_str.to_string()
}

pub fn get_layout_bias(page_type: &str, lang: &str) -> (String, String) {
    let mut bias = String::from("detail has_list has_form true false");
    let mut prejudice = String::new(); // 🌟 초기화
    
    let lang_code = if lang.len() >= 2 { &lang[0..2].to_lowercase() } else { "en" };
    let localized_type = get_localized_page_type(page_type, lang);
    
    if let Some(localized_obj) = BIAS_DICT.get(lang_code).or_else(|| BIAS_DICT.get("en")).and_then(|l| l.get(page_type).or_else(|| l.get("default"))) {
        if let Some(l_list) = localized_obj.get("layout_list") {
            if let Some(b) = l_list.get("bias").and_then(|v| v.as_str()) {
                bias.push_str(" ");
                bias.push_str(&b.replace("{TYPE}", &localized_type));
            }
            if let Some(p) = l_list.get("prejudice").and_then(|v| v.as_str()) {
                prejudice.push_str(" ");
                prejudice.push_str(&p.replace("{TYPE}", &localized_type));
            }
        }
        if let Some(l_form) = localized_obj.get("layout_form") {
            if let Some(b) = l_form.get("bias").and_then(|v| v.as_str()) {
                bias.push_str(" ");
                bias.push_str(&b.replace("{TYPE}", &localized_type));
                bias.push_str(&format!(" {} input {} select {} textarea", localized_type, localized_type, localized_type));
            }
            if let Some(p) = l_form.get("prejudice").and_then(|v| v.as_str()) {
                prejudice.push_str(" ");
                prejudice.push_str(&p.replace("{TYPE}", &localized_type));
            }
        }
    } else {
        bias.push_str(&format!(" {} input {} select {} textarea", localized_type, localized_type, localized_type));
    }
    
    if prejudice.trim().is_empty() {
        prejudice = String::from("global navigation, menus, footers, aside, search, filter.");
    }
    
    (bias.trim().to_string(), prejudice.trim().to_string())
}

pub fn get_separated_layout_bias(page_type: &str, lang: &str) -> (String, String, String) {
    // 영어 하드코딩을 제거하고 초기값을 비워 bias.json의 데이터를 100% 신뢰하도록 변경합니다.
    let mut list_bias = String::new();
    let mut form_bias = String::new();
    let mut prejudice = String::new();
    
    let lang_code = if lang.len() >= 2 { &lang[0..2].to_lowercase() } else { "en" };
    let localized_type = get_localized_page_type(page_type, lang);
    
    if let Some(localized_obj) = BIAS_DICT.get(lang_code).or_else(|| BIAS_DICT.get("en")).and_then(|l| l.get(page_type).or_else(|| l.get("default"))) {
        if let Some(l_list) = localized_obj.get("layout_list") {
            if let Some(b) = l_list.get("bias").and_then(|v| v.as_str()) {
                list_bias.push_str(&b.replace("{TYPE}", &localized_type));
            }
            if let Some(p) = l_list.get("prejudice").and_then(|v| v.as_str()) {
                prejudice.push_str(&p.replace("{TYPE}", &localized_type));
            }
        }
        if let Some(l_form) = localized_obj.get("layout_form") {
            if let Some(b) = l_form.get("bias").and_then(|v| v.as_str()) {
                form_bias.push_str(&b.replace("{TYPE}", &localized_type));
                form_bias.push_str(&format!(" {} input {} select {} textarea", localized_type, localized_type, localized_type));
            }
            if let Some(p) = l_form.get("prejudice").and_then(|v| v.as_str()) {
                if !prejudice.is_empty() { prejudice.push_str(" "); }
                prejudice.push_str(&p.replace("{TYPE}", &localized_type));
            }
        }
    } 
    
    // bias.json에서 데이터를 찾지 못한 예외적인 상황에만 적용되는 폴백(Fallback) 방어 로직
    if list_bias.trim().is_empty() {
        list_bias = format!("{} list catalog grid repeating multiple table rows items", localized_type);
    }
    if form_bias.trim().is_empty() {
        form_bias = format!("{} detail form single input fields properties configuration input select textarea", localized_type);
    }
    if prejudice.trim().is_empty() {
        prejudice = String::from("global navigation, menus, footers, aside, search, guide, tip, filter.");
    }
    
    (list_bias.trim().to_string(), form_bias.trim().to_string(), prejudice.trim().to_string())
}

pub fn get_page_type_full_bias(page_type: &str, lang: &str) -> String {
    let mut full_bias = String::from(page_type);
    let lang_code = if lang.len() >= 2 { &lang[0..2].to_lowercase() } else { "en" };
    let localized_type = get_localized_page_type(page_type, lang);
    
    if let Some(localized_obj) = BIAS_DICT.get(lang_code).or_else(|| BIAS_DICT.get("en")).and_then(|l| l.get(page_type).or_else(|| l.get("default"))) {
        if let Some(obj) = localized_obj.as_object() {
            for (_, v) in obj {
                if let Some(b) = v.get("bias").and_then(|bv| bv.as_str()) {
                    full_bias.push_str(" ");
                    full_bias.push_str(&b.replace("{TYPE}", &localized_type));
                }
            }
        }
    }
    
    full_bias.trim().to_string()
}

pub fn get_title_bias(page_type: &str, lang: &str) -> (String, String) {
    let lang_code = if lang.len() >= 2 { &lang[0..2].to_lowercase() } else { "en" };
    let localized_type = get_localized_page_type(page_type, lang);
    let mut bias = String::from("title name product");
    let mut prejudice = String::from("address location");
    if let Some(localized_obj) = BIAS_DICT.get(lang_code).or_else(|| BIAS_DICT.get("en")).and_then(|l| l.get(page_type).or_else(|| l.get("default"))) {
        if let Some(t_obj) = localized_obj.get("title") {
            if let Some(b) = t_obj.get("bias").and_then(|v| v.as_str()) { bias = format!("{} {}", bias, b.replace("{TYPE}", &localized_type)); }
            if let Some(p) = t_obj.get("prejudice").and_then(|v| v.as_str()) { prejudice = format!("{} {}", prejudice, p.replace("{TYPE}", &localized_type)); }
        }
    }
    (bias, prejudice)
}

pub fn get_list_schema_fields(page_type: &str, _href: &str, lang: &str) -> Vec<(String, String, String, String)> {
    let mut fields = Vec::new();
    
    let lang_code = if lang.len() >= 2 { &lang[0..2].to_lowercase() } else { "en" };
    let localized_type = get_localized_page_type(page_type, lang);

    let mut add = |key: &str, field_type: &str, en_bias: &str, en_prejudice: &str| {
        let mut final_bias = en_bias.to_string();
        let mut final_prejudice = en_prejudice.to_string();
        let mut semantic_desc = String::new();
        
        if let Some(localized_obj) = BIAS_DICT
            .get(lang_code)
            .and_then(|l| l.get(page_type).or_else(|| l.get("default")))
            .and_then(|p| p.get(key))
        {
            if let Some(semantic) = localized_obj.get("semantic").and_then(|v| v.as_str()) {
                semantic_desc = semantic.to_string();
            }
            if let Some(bias_str) = localized_obj.get("bias").and_then(|v| v.as_str()) {
                final_bias = format!("{} {}", en_bias, bias_str.replace("{TYPE}", &localized_type));
            }
            if let Some(prejudice_str) = localized_obj.get("prejudice").and_then(|v| v.as_str()) {
                final_prejudice = format!("{} {}", en_prejudice, prejudice_str.replace("{TYPE}", &localized_type));
            }
        }
        
        let final_desc = if key == "id,link" {
            "- \"link\": String. Detailed page URL.\n- \"id\": String. Refer to the ID value from the link.".to_string()
        } else if semantic_desc.is_empty() {
            format!("- \"{}\": {}.", key, field_type)
        } else {
            format!("- \"{}\": {}. {}", key, field_type, semantic_desc)
        };
        
        fields.push((key.to_string(), final_desc, final_bias.trim().to_string(), final_prejudice.trim().to_string()));
    };

    match page_type {
        "tracking" => {
            add("id,link", "", "id link tracking", "");
            add("status", "String", "status delivery", "");
            add("title", "String", "title product name", "");
            add("registration_date", "String", "date registration", "");
            add("sender_name", "String", "sender seller name", "");
            add("recipient_name", "String", "recipient buyer name", "");
            add("carrier", "String", "carrier courier", "");
        },
        "goods" => {
            add("id,link", "", "id link", "");
            add("code", "String", "code sku item", "");
            add("status", "String", "status condition", "");
            add("title", "String", "title name product", "");
            add("registration_date", "String", "date registration", "");
            add("sale_price", "Number", "sale price discount", "");
            add("supply_price", "Number", "supply price cost", "");
            add("currency", "String", "currency", "");
            add("quantity", "Number", "quantity inventory stock", "");
            add("stock_keeping_unit", "String", "sku code", "");
            add("main_image_url", "String", "main image thumbnail", "");
        },
        "order" => {
            add("id,link", "", "id link order number", "");
            add("tracking_number", "String", "tracking number", "");
            add("status", "String", "status state", "");
            add("registration_date", "String", "date registration", "");
            add("goods", "Array of Objects [ { title:String, link:String, id:String } ]", "goods product items", "");
            add("sender_name", "String", "sender buyer orderer name", "");
            add("recipient_name", "String", "recipient receiver name", "");
            add("payment_method", "String", "payment method type", "");
            add("payment_date", "String", "payment date", "");
        },
        "coupon" | "event" => {
            add("id,link", "", "id link", "");
            add("type", "String", "type", "");
            add("status", "String", "status state", "");
            add("title", "String", "title name", "");
            add("started_at", "String", "start date time", "");
            add("expired_at", "String", "expire end date time", "");
            add("code", "String", "code", "");
            add("discount", "Number", "discount", "");
            add("quantity", "Number", "quantity amount", "");
        },
        "review" => {
            add("id,link", "", "id link", "");
            add("status", "String", "status state", "");
            add("name", "String", "name reviewer author", "");
            add("title", "String", "title subject product", "");
            add("completed", "Boolean", "completed purchased", "");
            add("registration_date", "String", "date registration time", "");
        },
        _ => {
            add("id,link", "", "id link", "");
            add("title", "String", "title name", "");
            add("status", "String", "status state", "");
            add("registration_date", "String", "date registration", "");
        }
    }
    fields
}

pub fn get_vision_tracking_bias(lang: &str) -> (String, String) {
    let lang_code = if lang.len() >= 2 { &lang[0..2].to_lowercase() } else { "en" };
    let mut bias = String::from("tracking number barcode awb");
    let mut prejudice = String::new();
    if let Some(localized_obj) = BIAS_DICT.get(lang_code).or_else(|| BIAS_DICT.get("en")).and_then(|l| l.get("tracking")) {
        if let Some(id_obj) = localized_obj.get("id,link") {
            if let Some(b) = id_obj.get("bias").and_then(|v| v.as_str()) { bias = format!("{} {}", bias, b); }
        }
        if let Some(c_obj) = localized_obj.get("carrier") {
            if let Some(b) = c_obj.get("bias").and_then(|v| v.as_str()) { bias = format!("{} {}", bias, b); }
        }
    }
    (bias, prejudice)
}

// 🌟 글로벌 언어를 한 번에 섞어 넣지 않고, 전달받은 언어 코드의 힌트만 생성합니다.
pub fn get_layout_prompt_hints(page_type: &str, lang: &str) -> (String, String) {
    let lang_code = if lang.len() >= 2 { &lang[0..2].to_lowercase() } else { "en" };
    let localized_type = get_localized_page_type(page_type, lang);
    let mut list_words = String::new();
    let mut form_words = String::new();
    
    if let Some(localized_obj) = BIAS_DICT.get(lang_code).or_else(|| BIAS_DICT.get("en")).and_then(|l| l.get(page_type).or_else(|| l.get("default"))) {
        if let Some(l_list) = localized_obj.get("layout_list").and_then(|v| v.get("bias")).and_then(|v| v.as_str()) {
            list_words.push_str(&l_list.replace("{TYPE}", &localized_type));
        }
        if let Some(l_form) = localized_obj.get("layout_form").and_then(|v| v.get("bias")).and_then(|v| v.as_str()) {
            form_words.push_str(&l_form.replace("{TYPE}", &localized_type));
        }
    }
    
    let list_hints = if list_words.is_empty() { String::new() } else { format!("\n  (Related keywords in document: {})", list_words.trim()) };
    let form_hints = if form_words.is_empty() { String::new() } else { format!("\n  (Related keywords in document: {})", form_words.trim()) };
    
    (list_hints, form_hints)
}

pub fn is_detail_prompt(page_type: &str, title: &str, lang: &str) -> String {
    let (list_hints, form_hints) = get_layout_prompt_hints(page_type, lang);

    let template = r###"[TASK]
Analyze the provided PUG/HTML content from top to bottom.

[ENTITY CONTEXT: {TYPE}]
Language Context: {LANGUAGE}
You are evaluating a page managing this specific domain entity. Use this context to conceptually understand the abstract structures:
- has_form: A property configuration interface. It features a large overarching form dedicated to inputting or updating the specific attributes of ONE primary entity.{FORM_HINTS}
- has_list: A catalog or inventory interface dedicated to displaying, filtering, or batch-processing multiple DIFFERENT primary entities.{LIST_HINTS}

[FORCED DOCUMENT SCANNING LOGIC]
Read the entire document from top to bottom, applying the following strict filters and evaluations:

1. IGNORE:
   - Strictly ignore global navigation, menus, headers, footers, aside, search, filter.
2. TARGET:
   - Focus purely on the main data payload where "{TYPE}", or actual items are listed.
3. EVALUATE:
   - You MUST evaluate the concluding elements at the very bottom of the main content area first. Check for the following:
     A. Does the page terminate with dataset navigation (pagination, "next/prev") or bulk-action execution elements?
     B. Does the main data area consist of a repeating multi-entity grid?
     C. Does the main data area contain an extensive configuration/input form (inputs, textareas, image uploads, save buttons) for a single entity?

[SCHEMA DEFINITIONS]
- {TYPE}:
    - has_header: Boolean. True if the document contains a header.
    - title: String. Default '{TITLE}'.
    - has_footer: Boolean. True if the document contains a footer.
    - language: String. Default '{LANGUAGE}'.
    - has_list: Boolean. True if the document contains a multi-entity grid, OR if the bottom of main content area has dataset navigation/bulk controls.
    - has_form: Boolean. True if the main data payload is heavily composed of data entry fields (text, select, radio, file uploads) dedicated to creating or updating a single entity.

[OUTPUT FORMAT]
{
  "{TYPE}": {
    "has_header": Boolean,
    "title": String,
    "has_footer": Boolean,
    "language": String,
    "has_list": Boolean,
    "has_form": Boolean
  }
}

JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;
    
    template.replace("{TYPE}", page_type)
        .replace("{TITLE}", title)
        .replace("{LANGUAGE}", lang)
        .replace("{FORM_HINTS}", &form_hints)
        .replace("{LIST_HINTS}", &list_hints)
}

// pub fn para2graph(language: &str) -> String {
//     let template = r###"convert the natural language content to fit the dataset JSON structure. no explanation.
//     {
//         "context" : [
//             {
//                 "language" : "{LANG}",
//                 "type": "sales" | "order" | "goods" | "tracking" | "view" | "review" | "coupon" | "event" | "",
//                 "text": "Segment the natural language content into single-type contexts"
//             }
//         ]
//     }"###;
//     template.replace("{LANG}", language)
// }

pub fn para2graph(language: &str) -> String {
    let template = r###"Translate and convert the given natural language search query into English, then segment it into the specified JSON dataset structure.

[DOCUMENT SCANNING & STRICT SEGMENTATION LOGIC]
1. EXACT COPY: Copy the full original input into 'original_text' without changing anything.
2. TRANSLATE & TAGGED PIPE PLANNING: Translate the query into English. In the 'segmented_plan' field, prefix every translated segment with its assigned type tag in brackets, separated by pipes ('|'). Structure it strictly as '[tag1] english chunk1 | [tag2] english chunk2'.
3. MAXIMAL GROUPING: Group all contiguous words belonging to the same type into a SINGLE English segment. DO NOT split subjects from their numeric conditions. Break the segment ONLY when the type logically shifts.
4. STRICT ARRAY MAPPING: For EVERY tagged English segment in 'segmented_plan', create exactly one object in the 'context' array sequentially.

[SCHEMA DEFINITIONS]
- original_text: String. The exact, unaltered full natural language input.
- segmented_plan: String. Translated English text with '[type] english text | ' format inserted strictly at type boundaries.
- context:
  - 'text': String. The translated English chunk.
  - 'language': String. Default '{LANG}'.
  - 'type': String. Choose one:
    * 'order': Intent to measure sales performance or direct transactions. Triggers: conversion rate, sales volume, checkout, payment, cancellation, refund. (RULE: If the context measures buying success or revenue, classify as 'order' even if the word 'product' or 'item' is present).
    * 'goods': Intent to describe product catalog data, exposure, or traffic metrics. Triggers: page views, clicks, physical attributes, stock limits, unit prices. (RULE: Focuses on item specifications and customer traffic before the actual purchase).
    * 'tracking': Intent to manage logistics and fulfillment. Triggers: shipment status, dispatch, delivery duration, courier information.
    * 'review': Intent to analyze the voice of the customer. Triggers: feedback, ratings, reviews, CS messages, complaints.
    * 'coupon': Intent to manage specific discount vouchers. Triggers: coupon codes, issuance limits, discount amounts applied via coupons.
    * 'event': Intent to manage marketing campaigns or analyze broad operational trends. Triggers: promotions, exhibitions, seasonal sales, overarching managerial analysis requests.
    * '': If none logically apply.

[OUTPUT FORMAT]
{
  "original_text": "String",
  "segmented_plan": "String",
  "context": [...]
}

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;
    template.replace("{LANG}", language)
}

// --- File: src/parsing.rs ---

// ==============================================
// [Zone: src/parsing.rs 교체 코드]
// ==============================================
pub fn extract_numeric_conditions(current: &str, input: &str, seg_type: &str, metrics_json: &str) -> String {
    let template = r###"[Task]
Act as a deterministic semantic parser.
You must extract, transform, and normalize data from the natural language input into the strictly defined JSON output format based on the provided schema.

[DATABASE METRICS CONTEXT]
Use these current database min/max bounds to resolve any relative, proportional, or comparative queries into absolute numeric values.
Metrics: {METRICS}

[SCHEMA DEFINITION]
- operator: A string representing the comparison operator. Allowed values:
  * 'gt': Strictly greater than
  * 'gte': Greater than or equal to
  * 'lt': Strictly less than
  * 'lte': Less than or equal to
  * 'eq': Exact match

[TRANSFORMATION LOGIC - MANDATORY EXECUTION]
1. ATTRIBUTE EXTRACTION: Identify the context of the numbers or comparative words in the text to determine the property type.
2. RELATIVE VALUE CALCULATION (CRITICAL): 
   - If the query contains relative conditions, percentages, or comparative adjectives, you MUST use the [DATABASE METRICS CONTEXT] to calculate the EXACT absolute numeric threshold. 
   - Do NOT output percentages or descriptive text in the `value` field. Always compute and output the final absolute number derived from the min/max metrics.
3. OPERATOR SELECTION: Map the semantic intent to 'gt', 'gte', 'lt', 'lte', or 'eq'.

[FULL QUERY CONTEXT]
{INPUT}

[CURRENT CHUNK TO ANALYZE]
{CURRENT}

[OUTPUT FORMAT]
{
  "condition": {
    "{TYPE}": {
      "is_percent": Boolean,
      "percent_total": is_percent === true ? 100 : 0,
      "value": "...",
      "operator": "..."
    }
  }
}

[ACTION] JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{INPUT}", input)
            .replace("{CURRENT}", current)
            .replace("{TYPE}", seg_type)
            .replace("{METRICS}", metrics_json)
}
// ==============================================

pub fn graph2contexts(current_text: &str, seg_type: &str) -> String {
    
    let status_options = match seg_type {
        "tracking" => "* 'draft': Shipment preparation or pending pickup.
    * 'progress': Currently in transit or out for delivery.
    * 'return': Returning to sender.
    * 'complete': Successfully delivered to the recipient.
    * 'error': Delivery exception, lost, or failed.",
        "goods" => "* 'draft': Product is being created, not yet published.
    * 'show': Visible and available for sale on storefront.
    * 'hide': Hidden from the storefront.
    * 'progress': Currently being restocked or updated.
    * 'stop': Sales temporarily suspended.
    * 'cancel': Product discontinued or cancelled.
    * 'refund': Related to refunded inventory.
    * 'return': Related to returned inventory.
    * 'exchange': Related to exchanged inventory.
    * 'expire': Product expired.
    * 'complete': Completely sold out or finished lifecycle.
    * 'error': Data or system error.",
        "order" => "* 'draft': Pending payment or in cart.
    * 'progress': Order processing or preparing for shipment.
    * 'stop': Order on hold.
    * 'cancel': Order cancelled before fulfillment.
    * 'refund': Payment refunded.
    * 'return': Items returned by customer.
    * 'exchange': Items being exchanged.
    * 'expire': Payment window expired.
    * 'complete': Order fully fulfilled and closed.
    * 'error': Payment or processing error.",
        "coupon" | "event" => "* 'show': Visible to customers.
    * 'progress': Currently active and running.
    * 'hide': Hidden from customers.
    * 'stop': Temporarily paused.
    * 'cancel': Terminated early.
    * 'expire': Passed its expiration date.
    * 'complete': Successfully finished its run.
    * 'error': Configuration error.",
        "review" => "* 'progress': Under moderation or pending approval.
    * 'stop': Blocked or suspended review.
    * 'cancel': Deleted or withdrawn by user.
    * 'refund': Associated with a refunded order.
    * 'return': Associated with a returned order.
    * 'exchange': Associated with an exchanged order.
    * 'expire': Review period expired.
    * 'complete': Published and visible.
    * 'error': Rejected or marked as spam.",
        _ => "* 'show': Visible state.
    * 'progress': Active/Processing state.
    * 'remove': Deleted state.
    * 'hide': Hidden state.
    * 'stop': Paused/Stopped state.
    * 'cancel': Cancelled state.
    * 'refund': Refunded state.
    * 'return': Returned state.
    * 'exchange': Exchanged state.
    * 'expire': Expired state.
    * 'complete': Finished/Completed state.
    * 'error': Error state."
    };

    let template = r###"Analyze the specific text segment and extract the logical attributes based on the defined schema.

[CURRENT SEGMENT]
{TEXT}

[SCHEMA DEFINITIONS]
{TYPE}:
  - status: String. Choose one:
    {STATUS_OPTIONS}
    * '': If none logically apply.
  - substantial: String. Choose one:
    * 'size': Physical dimensions or volume.
    * 'weight': Mass or heaviness.
    * 'shipping_fee': Cost of delivery.
    * 'shipping_duration': Time taken for delivery.
    * 'sale_price': Final selling price to the customer.
    * 'supply_price': Wholesale or original cost.
    * 'low_stock_threshold': Minimum inventory alert level.
    * 'discount': Amount or percentage of price reduction.
    * 'min_order_amount': Minimum spend required to trigger an action.
    * 'max_discount_amount': Maximum cap for a discount.
    * 'usage_limit': Maximum number of times usable globally.
    * 'usage_per': Maximum number of times usable per user.
  - find: String. Choose one:
    * 'many': High quantity, count, or volume.
    * 'few': Low quantity, count, or volume.
    * 'much': High financial value, price, or amount.
    * 'little': Low financial value, price, or amount.
    * 'heavy': High physical weight.
    * 'light': Low physical weight.

[OUTPUT FORMAT]
{
  "{TYPE}" : {
    "status": "...",
    "substantial": "...",
    "find": "..."
  }
}

[ACTION] RETURN STRICTLY VALID JSON ONLY.
NO EXPLANATION. NO THINKING. /no_think"###;

    
    template.replace("{STATUS_OPTIONS}", status_options)
            .replace("{TYPE}", seg_type)
            .replace("{TEXT}", current_text)
}




// 🌟 [CRITICAL FIX] 4개의 반환값(String, String, String, String)으로 확장하여 prejudice를 스케줄러로 넘깁니다.
pub fn get_detail_schema_fields(page_type: &str, _href: &str, lang: &str) -> Vec<(String, String, String, String)> {
    let mut fields = Vec::new();
    
    let lang_code = if lang.len() >= 2 { &lang[0..2].to_lowercase() } else { "en" };
    let localized_type = get_localized_page_type(page_type, lang);
    
    let mut add = |key: &str, field_type: &str, en_bias: &str, en_prejudice: &str| {
        let mut final_bias = en_bias.to_string();
        let mut final_prejudice = en_prejudice.to_string();
        let mut semantic_desc = String::new();
        
        if let Some(localized_obj) = BIAS_DICT
            .get(lang_code)
            .and_then(|l| l.get(page_type).or_else(|| l.get("default")))
            .and_then(|p| p.get(key))
        {
            if let Some(semantic) = localized_obj.get("semantic").and_then(|v| v.as_str()) {
                semantic_desc = semantic.to_string();
            }
            if let Some(bias_str) = localized_obj.get("bias").and_then(|v| v.as_str()) {
                final_bias = format!("{} {}", en_bias, bias_str.replace("{TYPE}", &localized_type));
            }
            if let Some(prejudice_str) = localized_obj.get("prejudice").and_then(|v| v.as_str()) {
                final_prejudice = format!("{} {}", en_prejudice, prejudice_str.replace("{TYPE}", &localized_type));
            }
        }
        
        let final_desc = if key == "id,link" {
            "- \"link\": String. Detailed page URL.\n- \"id\": String. Refer to the ID value from the link.".to_string()
        } else if semantic_desc.is_empty() {
            format!("- \"{}\": {}.", key, field_type)
        } else {
            format!("- \"{}\": {}. {}", key, field_type, semantic_desc)
        };
        
        fields.push((key.to_string(), final_desc, final_bias.trim().to_string(), final_prejudice.trim().to_string()));
    };

    match page_type {
        "tracking" => {
            add("id,link", "", "id link tracking", "");
            add("status", "String", "status delivery", "");
            add("title", "String", "title product name", "");
            add("registration_date", "String", "date registration", "");
            add("shipping_date", "String", "date shipping dispatch", "");
            add("sender_name", "String", "sender seller name", "");
            add("sender_address", "String", "sender address location", "");
            add("sender_phone", "String", "sender phone contact", "");
            add("recipient_name", "String", "recipient buyer name", "");
            add("recipient_address", "String", "recipient address location", "");
            add("recipient_phone", "String", "recipient phone contact", "");
            add("width", "Number", "width dimension", "");
            add("height", "Number", "height dimension", "");
            add("length", "Number", "length dimension", "");
            add("weight", "Number", "weight mass", "");
            add("carrier", "String", "carrier courier", "");
            add("shipping_fee", "Number", "shipping fee cost", "");
            add("shipping_method", "String", "shipping method", "");
            add("shipping_duration", "Number", "shipping duration days", "");
            add("bundle_shipping", "String", "bundle combined shipping", "");
        },
        "goods" => {
            add("id,link", "", "id link", "");
            add("code", "String", "code sku item", "");
            add("status", "String", "status condition", "");
            add("title", "String", "title name product", "");
            add("registration_date", "String", "date registration", "");
            add("payment_method", "String", "payment method", "");
            add("bank", "String", "bank account", "");
            add("card", "String", "card credit", "");
            add("model_name", "String", "model name", "");
            add("brand_name", "String", "brand name", "");
            add("condition", "String", "condition state", "");
            add("description", "String", "description detail", "");
            add("short_description", "String", "short description summary", "");
            add("tags", "Array of Strings", "tags keywords", "");
            add("origin_country", "String", "origin country", "");
            add("manufacturer", "String", "manufacturer", "");
            add("release_date", "String", "release date", "");
            add("manufacture_date", "String", "manufacture date", "");
            add("expiration_date", "String", "expiration date", "");
            add("gtin", "String", "gtin barcode", "");
            add("mpn", "String", "mpn part number", "");
            add("barcode", "String", "barcode", "");
            add("sale_price", "Number", "sale price discount", "");
            add("supply_price", "Number", "supply price cost", "");
            add("currency", "String", "currency", "");
            add("compare_at_price", "Number", "compare original price", "");
            add("quantity", "Number", "quantity inventory stock", "");
            add("stock_keeping_unit", "String", "sku code", "");
            add("low_stock_threshold", "Number", "low stock threshold", "");
            add("unit", "String", "unit measure", "");
            add("tax_included", "Boolean", "tax vat", "");
            add("tax_code", "String", "tax code", "");
            add("main_image_url", "String", "main image thumbnail", "");
            add("additional_image_url", "String", "additional image", "");
            add("video_url", "String", "video url", "");
            add("carrier", "String", "carrier shipping", "");
            add("shipping_fee", "Number", "shipping fee cost", "");
            add("shipping_method", "String", "shipping method", "");
            add("shipping_duration", "Number", "shipping duration time", "");
            add("bundle_shipping", "String", "bundle combined shipping", "");
            add("options", "Array of Objects [ { value:String, inputs:[ { value:String } ] } ]", "options variations", "");
            add("additional_goods", "Array of Objects [ { link:String, id:String } ]", "additional goods related", "");
        },
        "order" => {
            add("id,link", "", "id link order number", "");
            add("tracking_number", "String", "tracking number", "");
            add("status", "String", "status state", "");
            add("registration_date", "String", "date registration", "");
            add("goods", "Array of Objects [ { title:String, link:String, id:String } ]", "goods product items", "");
            add("sender_name", "String", "sender buyer orderer name", "");
            add("sender_address", "String", "sender buyer address", "");
            add("sender_phone", "String", "sender buyer phone", "");
            add("recipient_name", "String", "recipient receiver name", "");
            add("recipient_address", "String", "recipient receiver address", "");
            add("recipient_phone", "String", "recipient receiver phone", "");
            add("bank", "String", "bank account", "");
            add("card", "String", "card credit", "");
            add("order_date", "String", "order date", "");
            add("payment_date", "String", "payment date", "");
            add("payment_method", "String", "payment method type", "");
            add("payment_origin", "String", "payment origin pg gateway", "");
        },
        "coupon" | "event" => {
            add("id,link", "", "id link", "");
            add("type", "String", "type", "");
            add("status", "String", "status state", "");
            add("title", "String", "title name", "");
            add("started_at", "String", "start date time", "");
            add("expired_at", "String", "expire end date time", "");
            add("code", "String", "code", "");
            add("discount", "Number", "discount", "");
            add("quantity", "Number", "quantity amount", "");
            add("usage_limit", "Number", "usage limit max", "");
            add("usage_per", "Number", "usage per customer user", "");
            add("new_customer_only", "Boolean", "new customer only", "");
            add("first_purchase_only", "Boolean", "first purchase only", "");
            add("min_order_amount", "Number", "minimum order amount", "");
            add("max_order_amount", "Number", "maximum order amount", "");
            add("max_discount_amount", "Number", "maximum discount amount", "");
            add("region_restrictions", "String", "region restriction area", "");
            add("number", "String", "number phone contact", "");
            add("address", "String", "address location", "");
            add("registration_date", "String", "date registration", "");
        },
        "review" => {
            add("id,link", "", "id link", "");
            add("status", "String", "status state", "");
            add("name", "String", "name reviewer author", "");
            add("title", "String", "title subject product", "");
            add("completed", "Boolean", "completed purchased", "");
            add("registration_date", "String", "date registration time", "");
        },
        _ => {
            add("id,link", "", "id link", "");
            add("title", "String", "title name", "");
            add("status", "String", "status state", "");
        }
    }
    fields
}

// pub fn extract_single_field_prompt(page_type: &str, field_name: &str, field_desc: &str, language: &str, title: &str) -> String {
//     // 🌟 복수 키(예: "id,link") 대응을 위해 동적 JSON 포맷 생성
//     let mut dynamic_output_keys = String::new();
//     for key in field_name.split(',') {
//         dynamic_output_keys.push_str(&format!("\"{}\": \"...\",\n", key.trim()));
//     }
//     let dynamic_output_keys = dynamic_output_keys.trim_end_matches(",\n");

//     let template = r###"[TASK]
// Analyze the provided PUG/HTML content from top to bottom to extract specific properties into a JSON object.

// [CONTEXT]
// Page Type: {TYPE}
// Language: {LANGUAGE}

// [FORCED DOCUMENT SCANNING LOGIC]
// Read the entire document from top to bottom, applying the following strict filters and evaluations:

// 1. IGNORE:
//    - Strictly ignore global navigation, menus, headers, footers, aside, search, filter, form.
//    - temporary placeholder (such as SKIP READ N, SKIP_READ_N, LINK_SKIP).
// 2. TARGET:
//    - Focus purely on the main data payload where the target properties are located.
// 3. EVALUATE:
//    - You MUST evaluate the concluding elements at the very bottom of the main content area first. Check for the following:
//      A. Does the page terminate with dataset navigation (pagination, "next/prev") or bulk-action execution elements?
//      B. Does the main data area consist of a repeating multi-entity grid?
//      C. Does the main data area contain an extensive configuration/input form (inputs, textareas, image uploads, save buttons) for a single entity?

// [SCHEMA DEFINITIONS]
// - "has_header": Boolean. True if the document contains a header.
// - "has_footer": Boolean. True if the document contains a footer.
// - "title": String. Default '{TITLE}'.
// - "language": String. Detect the language of PUG CONTENT and return ISO 639-1 code.
// {FIELDS}

// [EXTRACTION RULES]
// 1. Return ONLY valid JSON.
// 2. If the field is completely missing in the data, use null.
// 3. Normalize all dates to 'yyyy-MM-ddThh:mm:ss'.
// 4. Extract only numeric values for price, amount, weight, and dimensions.
// 5. Do NOT make up data. Only extract what is present in the Pug structure.

// [OUTPUT FORMAT]
// {
//     "has_header": Boolean,
//     "title" : String,
//     "has_footer": Boolean,
//     "language": String,
//     {DYNAMIC_KEYS}
// }

// [ACTION] RETURN JSON ONLY. NO EXPLANATION. NO COMMENTS IN JSON. /no_think"###;

//     template.replace("{TYPE}", page_type)
//             .replace("{LANGUAGE}", language)
//             .replace("{FIELDS}", field_desc)
//             .replace("{TITLE}", title)
//             .replace("{DYNAMIC_KEYS}", &dynamic_output_keys)
// }
pub fn extract_single_field_prompt(page_type: &str, field_name: &str, field_desc: &str, language: &str, title: &str) -> String {
    // 🌟 복수 키(예: "id,link") 대응을 위해 동적 JSON 포맷 생성
    let mut dynamic_output_keys = String::new();
    for key in field_name.split(',') {
        dynamic_output_keys.push_str(&format!("  \"{}\": \"...\",\n", key.trim()));
    }
    let dynamic_output_keys = dynamic_output_keys.trim_end_matches(",\n");

    // 🌟 인메모리 벡터 검색으로 정제된 컨텍스트에 맞춘 최적화된 프롬프트
    let template = r###"[TASK]
Analyze the provided concentrated PUG/HTML context and extract specific properties into a JSON object.

[CONTEXT]
Page Type: {TYPE}
Language: {LANGUAGE}
Document Title: {TITLE}

[SCHEMA DEFINITIONS]
{FIELDS}

[EXTRACTION RULES]
1. Return ONLY valid JSON containing the requested keys.
2. If the field is completely missing in the data, use null.
3. Normalize all dates to 'yyyy-MM-ddThh:mm:ss'.
4. Extract only numeric values for price, amount, weight, and dimensions.
5. Do NOT make up data. Only extract what is present in the context.

[OUTPUT FORMAT]
{
{DYNAMIC_KEYS}
}

[ACTION] RETURN JSON ONLY. NO EXPLANATION. NO COMMENTS IN JSON. /no_think"###;

    template.replace("{TYPE}", page_type)
            .replace("{LANGUAGE}", language)
            .replace("{FIELDS}", field_desc)
            .replace("{TITLE}", title)
            .replace("{DYNAMIC_KEYS}", &dynamic_output_keys)
}



/// Converts a JSON Value into a human-readable natural language narrative.
/// [STRICT ALIGNMENT] This logic perfectly synchronizes with every column in `parsing.rs`.
pub fn json_to_natural_language(json_val: &serde_json::Value) -> String {
    let mut sentences = Vec::new();

    // 최상위 식별자(ID, Link 등)를 먼저 추출하여 문장 생성
    if let Some(obj) = json_val.as_object() {
        if let Some(id) = obj.get("id").or(obj.get("no")).and_then(|v| v.as_str()) {
            sentences.push(format!("The unique identifier is {}.", id));
        }
        if let Some(link) = obj.get("link").or(obj.get("path")).and_then(|v| v.as_str()) {
            sentences.push(format!("It can be accessed at {}.", link));
        }
    }

    fn parse_node(val: &serde_json::Value, context_name: &str, sentences: &mut Vec<String>) {
        match val {
            serde_json::Value::Object(map) => {
                let title = map.get("title").or(map.get("name")).and_then(|v| v.as_str()).unwrap_or("");
                let item_type = map.get("type").and_then(|v| v.as_str()).unwrap_or("");
                
                let mut intro = String::new();
                if !title.is_empty() {
                    let type_str = if item_type.is_empty() { "item" } else { item_type };
                    intro.push_str(&format!("This {} is titled '{}'.", type_str, title));
                } else if !context_name.is_empty() && context_name != "item" {
                    intro.push_str(&format!("Regarding {},", context_name));
                }
                
                if !intro.is_empty() && !sentences.contains(&intro) {
                    sentences.push(intro);
                }

                for (key, v) in map {
                    // 이미 처리된 핵심 속성 및 시스템 변수는 스킵
                    if ["title", "name", "type", "currency", "text", "json_data", "data", "id", "index", "no", "link", "path", "origin", "mode", "detail"].contains(&key.as_str()) { continue; }
                    if v.is_null() || (v.is_string() && v.as_str().unwrap_or("").trim().is_empty()) { continue; }
                    
                    let clean_key = key.replace("_", " ");
                    
                    if v.is_object() || v.is_array() {
                        parse_node(v, &clean_key, sentences);
                    } else {
                        let val_str = match v {
                            serde_json::Value::String(s) => s.clone(),
                            serde_json::Value::Number(n) => n.to_string(),
                            serde_json::Value::Bool(b) => b.to_string(),
                            _ => String::new(),
                        };
                        
                        if ["sale_price", "supply_price", "price", "amount", "shipping_fee", "discount"].contains(&key.as_str()) {
                            let curr = map.get("currency").and_then(|c| c.as_str()).unwrap_or("");
                            let curr_str = if curr.is_empty() { String::new() } else { format!(" {}", curr) };
                            sentences.push(format!("The {} is {}{}.", clean_key, val_str, curr_str));
                        } else if key == "status" {
                            sentences.push(format!("It is currently in '{}' status.", val_str));
                        } else {
                            sentences.push(format!("Its {} is {}.", clean_key, val_str));
                        }
                    }
                }
            },
            serde_json::Value::Array(arr) => {
                let mut arr_vals = Vec::new();
                for item in arr {
                    if item.is_string() || item.is_number() || item.is_boolean() {
                        arr_vals.push(item.as_str().map(|s| s.to_string()).unwrap_or_else(|| item.to_string()));
                    } else {
                        parse_node(item, context_name, sentences);
                    }
                }
                if !arr_vals.is_empty() {
                    sentences.push(format!("The {} includes: {}.", context_name, arr_vals.join(", ")));
                }
            },
            _ => {
                let val_str = val.as_str().map(|s| s.to_string()).unwrap_or_else(|| val.to_string());
                if !val_str.is_empty() {
                    sentences.push(format!("The {} is {}.", context_name, val_str));
                }
            }
        }
    }

    parse_node(json_val, "item", &mut sentences);
    
    let mut unique_sentences = Vec::new();
    for s in sentences {
        if !unique_sentences.contains(&s) { unique_sentences.push(s); }
    }
    
    unique_sentences.join(" ").replace("  ", " ").trim().to_string()
}

pub fn normalize_to_json_string(input: &str) -> String {
    let mut s = input.replace(&['\u{00A0}', '\u{200B}', '\u{202F}', '\u{FEFF}'][..], " ").trim().to_string();

    // 🌟 [CRITICAL FIX] LLM이 흔히 생성하는 스마트 따옴표 및 전각 기호를 표준 ASCII 기호로 일괄 치환합니다.
    s = s.replace('“', "\"")
         .replace('”', "\"")
         .replace('‘', "'")
         .replace('’', "'")
         .replace('，', ",")
         .replace('：', ":");

    // 🌟 [LLM ESCAPE FIX] Qwen 모델이 JSON 내부에서 "has_list\":false 처럼 큰따옴표 앞에 오염시킨 백슬래시(\)를 원천 제거합니다.
    s = s.replace("\\\"", "\"");

    // 1. Backticks to quotes
    let re_backtick = Regex::new(r"`([\s\S]*?)`").unwrap();
    s = re_backtick.replace_all(&s, |caps: &regex::Captures| {
        format!("\"{}\"", caps[1].replace("\"", "\\\""))
    }).to_string();

    // 2. Key quotes correction (key: -> "key":)
    // 🌟 [CRITICAL FIX] 줄 시작 지점(^), 중괄호({), 쉼표(,) 뒤에 나오는 Unquoted Key를 모두 확실히 잡아내어 따옴표로 감쌉니다.
    let re_keys = Regex::new(r"(?m)(^|[{,])\s*([a-zA-Z0-9_]+)\s*:").unwrap();
    s = re_keys.replace_all(&s, "$1\"$2\":").to_string();

    // 3. Single quotes to double quotes for values
    let re_single_vals = Regex::new(r":\s*'([^']*)'").unwrap();
    s = re_single_vals.replace_all(&s, |caps: &regex::Captures| {
        format!(": \"{}\"", caps[1].replace("\"", "\\\""))
    }).to_string();

    // 4. CSS selector/nested quotes protection (Simplified for Rust regex)
    let re_nested = Regex::new(r#"="([^"]*)""#).unwrap();
    s = re_nested.replace_all(&s, "=\\\"$1\\\"").to_string();

    // 5. Trailing Comma removal
    let re_trailing = Regex::new(r",\s*([\]}])").unwrap();
    s = re_trailing.replace_all(&s, "$1").to_string();

    
    let re_artifact = Regex::new(r",?\s*\.\.\.\s*([\]}])").unwrap();
    s = re_artifact.replace_all(&s, "$1").to_string();

    
    let mut in_string = false;
    let mut escape = false;
    let mut stack = Vec::new();

    for c in s.chars() {
        if escape {
            escape = false;
        } else if c == '\\' {
            escape = true;
        } else if c == '"' {
            in_string = !in_string;
        } else if !in_string {
            match c {
                '{' => stack.push('}'),
                '[' => stack.push(']'),
                '}' | ']' => {
                    if stack.last() == Some(&c) {
                        stack.pop();
                    }
                }
                _ => {}
            }
        }
    }

    // 문자열이 끊겼을 경우 닫는 따옴표 추가 (끝부분 이스케이프 문자 방어)
    if in_string {
        if s.ends_with('\\') {
            s.pop(); 
        }
        s.push('"');
    }

    // 뒤에 쉼표가 남은 상태로 끊겼을 경우 쉼표 제거
    s = s.trim_end().to_string();
    if s.ends_with(',') {
        s.pop();
    }

    // 남은 괄호 스택을 순서대로 역전하여 전부 조립
    while let Some(c) = stack.pop() {
        s.push(c);
    }

    s
}

pub fn parse_json_from_llm(text: &str) -> serde_json::Value {
    // [CLEANUP] Remove <think>...</think> tags if they exist
    let mut clean_text = text.to_string();
    if let Some(start_think) = clean_text.find("<think>") {
        if let Some(end_think) = clean_text.find("</think>") {
            if start_think < end_think {
                clean_text.replace_range(start_think..end_think + 8, "");
            }
        } else {
            clean_text.replace_range(start_think.., "");
        }
    }
    let clean_text = clean_text.trim();

    // 1. First attempt: Direct parse
    if let Ok(v) = serde_json::from_str(clean_text) { return v; }

    // 2. Extract JSON part and attempt direct parse
    let mut extracted = String::new();
    if let Some(start) = clean_text.find("{") {
        if let Some(end) = clean_text.rfind("}") {
            if start < end {
                let attempt = clean_text[start..=end].to_string();
                if let Ok(v) = serde_json::from_str(&attempt) { return v; }
            }
        }
        
        extracted = clean_text[start..].to_string();
    } else if let Some(start) = clean_text.find("[") {
        if let Some(end) = clean_text.rfind("]") {
            if start < end {
                let attempt = clean_text[start..=end].to_string();
                if let Ok(v) = serde_json::from_str(&attempt) { return v; }
            }
        }
        
        extracted = clean_text[start..].to_string();
    }

    // 3. Last attempt: Normalize/Repair then parse
    let to_repair = if extracted.is_empty() { clean_text } else { &extracted };
    let repaired = normalize_to_json_string(to_repair);
    
    if let Ok(v) = serde_json::from_str(&repaired) {
        println!("[Parsing] Success: JSON successfully repaired and parsed!");
        return v;
    } else {
        // Final fallback: try extracting from repaired string again
        if let Some(start) = repaired.find("{") {
            if let Some(end) = repaired.rfind("}") {
                if let Ok(v) = serde_json::from_str(&repaired[start..=end]) { 
                    println!("[Parsing] Success: JSON successfully repaired and parsed on final fallback!");
                    return v; 
                }
            }
        }

        
        println!("[Parsing] Attempting aggressive character-by-character truncation repair...");
        let mut shrink_attempt = to_repair.to_string();
        let mut attempts = 0;
        
        // 시스템 랙(Freezing)을 방지하기 위해 최대 500번(글자)까지만 깎아내며 재시도합니다. (최소 5글자 유지)
        while shrink_attempt.len() > 5 && attempts < 500 {
            shrink_attempt.pop(); // 맨 끝 글자 하나 제거
            attempts += 1;
            
            let attempt_repaired = normalize_to_json_string(&shrink_attempt);
            if let Ok(v) = serde_json::from_str(&attempt_repaired) {
                println!("[Parsing] Success: JSON repaired by aggressive truncation after dropping {} characters!", attempts);
                return v;
            }
        }
    }

    println!("[Parsing] Warning: Failed to repair dirty JSON: {}", clean_text);
    serde_json::json!({})
}







pub fn get_trade_doc_classification_prompt() -> String {
    // TRACKING을 선택지에 명시적으로 추가
    r###"Classify document type. Choose strictly from: PI, CI, BL, AWB, PL, CO, LC, TRACKING, Unknown. 
Return JSON exactly like: {"doc_type": "BL"}
NO EXPLANATION."###.to_string()
}

pub fn get_trade_category_schema(category: &str, doc_type: &str) -> String {
    let schema = match category {
        "header" => r#"{
  "document_type": "CLASSIFIED TYPE {String}",
  "document_number": "Primary Identifier (B/L No, Invoice No) {String}",
  "issue_date": "Date of Creation (YYYY-MM-DD) {String}",
  "reference_number": "Export Ref, Booking No, PO No {String}"
}"#,
        "parties" => r#"{
  "supplier_name": "Shipper, Seller, Exporter {String}",
  "supplier_address": "Address of Supplier {String}",
  "buyer_name": "Consignee, Buyer, Importer {String}",
  "buyer_address": "Address of Buyer {String}",
  "notify_party_name": "Notify Party Name {String}"
}"#,
        "logistics" => r#"{
  "vehicle_name": "Vessel Name, Flight No {String}",
  "voyage_number": "Voyage No {String}",
  "location_port_of_loading": "POL, Airport of Departure {String}",
  "location_port_of_discharge": "POD, Airport of Destination {String}"
}"#,
        "conditions" => r#"{
  "incoterms_code": "FOB, CIF, EXW, DDP {String}",
  "freight_payment_term": "Freight Prepaid, Freight Collect {String}"
}"#,
        "financials" => r#"{
  "currency_code": "Currency Symbol (USD, EUR) {String}",
  "amount_total": "Grand Total Amount {Number}"
}"#,
        "cargo" => r#"{
  "package_count": "Total Quantity (NOT Money) {Number}",
  "weight_gross": "Total Gross Weight {Number}",
  "volume_measurement": "Total Volume (CBM) {Number}",
  "marks_and_numbers": "Marks & Nos {String}"
}"#,
        "items" => r#"[ {
  "description": "Description of Goods {String}",
  "quantity": "Line Item Quantity {Number}",
  "hs_code": "HS Code / Tariff No {String}"
} ]"#,
        "containers" => r#"[ {
  "container_number": "Container No (4 char + 7 digit) {String}",
  "seal_number": "Seal No {String}",
  "type_description": "Type (20GP, 40HC) {String}"
} ]"#,
        _ => "{}"
    };

    format!("RULES: Follow comments strictly. Output JSON ONLY. MISSION: Extract data for category '{}'.\nSCHEMA:\n{}", category.to_uppercase(), schema)
}

// ==========================================
// [수정] 무역 문서(Trade Doc) 검색을 위한 Shipping Condition 업그레이드
// ==========================================
pub fn extract_shipping_conditions(query: &str, language: &str) -> String {
    let template = r###"Task: Act as a deterministic shipping and trade logistics semantic parser.
Extract the logistics filters from the natural language query into the JSON format.

[SCHEMA DEFINITION]
Extract the following tracking/trade properties if semantically present in the text:
- "no": Tracking number, B/L number, Invoice number.
- "status": Shipping status (draft, progress, return, complete, error).
- "vessel": Vessel name, Flight No, or Carrier.
- "pol": Port of Loading, Origin, Departure point.
- "pod": Port of Discharge, Destination, Arrival point.
- "sender_name": Shipper, Seller, or Exporter name.
- "recipient_name": Consignee, Buyer, or Importer name.
- "incoterms": Incoterms (e.g., FOB, CIF, EXW).
- "weight": Cargo or gross weight.
- "amount": Total financial amount or price.

[TRANSFORMATION LOGIC]
For EVERY extracted field, wrap it in an operator object:
{ "operator": "eq" | "gt" | "lt" | "gte" | "lte" | "contains", "value": <extracted_value> }
- Use "contains" for text fields, names, ports, vessels.
- Use "eq" for strict identifiers or status.

[QUERY]
{QUERY}

[OUTPUT FORMAT]
{ "<property_name>": { "operator": "...", "value": "..." } }
JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{QUERY}", query).replace("{LANGUAGE}", language)
}

pub fn get_image_extraction_prompt(region: &str, language: &str, page_type: &str, address: &str) -> String {
    if page_type == "tracking" || page_type == "goods" {
        let template = r###"[TASK]
Convert the image to fit the structured JSON format. 

[CONTEXT]
Region: {REGION}
Recipient Address: {ADDRESS}
Current Language: {LANGUAGE}

[INSTRUCTION]
1. Extract the tracking_number or document number.
2. Set recipient_match to true if the label address matches the context address.
3. Extract all visible barcodes into an array.

[OUTPUT FORMAT]
{ "tracking_number": "string", "recipient_match": boolean, "barcodes": ["string"] }"###;
        template.replace("{REGION}", region).replace("{ADDRESS}", address).replace("{LANGUAGE}", language)
    } else {
        String::new()
    }
}

pub fn extract_table_structure_prompt(page_type: &str, item_selector: &str, pug_content: &str, reference_row: &str) -> String {
    let template = r###"[PUG CONTENT]
{PUG_CONTENT}

[Reference: Row Structure]
{REFERENCE_ROW}

[Instruction]
Locate the main table wrapper, its body container, and its corresponding header container within the [PUG CONTENT].

[Rules]
1. Tag Agnostic: Do NOT assume traditional <table> tags. The structure could be built using <div>, <ul>/<li>, or other semantic tags. Analyze logically.
2. Fill out the `table` selector FIRST to logically establish the common parent wrapper that encompasses both the header (thead) and the items (tbody).
3. The `tbody` selector is exactly "{ITEM_SELECTOR}". Return it as provided.
4. Provide the final exact CSS selector for the `thead` based on your analysis within that table wrapper.

[Expected Output Format]
{
  "{TYPE}" : {
    "tbody" : {
      "selector" : "{ITEM_SELECTOR}"
    },
    "table" : {
        "selector" : "..."
    },
    "thead" : {
      "selector" : "..."
    }
  }
}

[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think"###;

    template.replace("{TYPE}", page_type)
            .replace("{ITEM_SELECTOR}", item_selector)
            .replace("{PUG_CONTENT}", pug_content)
            .replace("{REFERENCE_ROW}", reference_row)
}


pub fn analytic_report_prompt() -> String {
    r###"[TASK]
You are a User Behavior Analysis Expert. Interpret raw HTML interactions to understand the user's specific intent and analyze the selection context within a list or a group of items.

Analyze the parallel arrays of 'Clicked HTML' (the selected element) and 'Related HTML' (the surrounding structure).
If 'Previous Analysis' is provided, use it to infer the user's behavioral flow and connect past actions with the current click.

[ANALYSIS GUIDELINES & CHAIN OF THOUGHT]
Fill out the JSON keys in the exact order specified below. Use 'analysis_*' keys to logically establish the context before finalizing the outputs.

1. analysis_target: Identify the primary entity name and its key attributes from the Clicked HTML.
2. analysis_surroundings: Identify the neighboring items or alternatives displayed in the Related HTML that were NOT selected.
3. action: Determine the specific user intent for clicking the item. Must explicitly include the primary entity name and key attributes. Output as a short verb phrase.
4. relate: Summarize the surrounding unselected items to capture the context of the choice. Do not summarize the clicked item itself in this field.
5. summary: Provide a detailed explanation of what the user aimed to accomplish on this page. Must explicitly reference the extracted primary entity and its key attributes.

[OUTPUT FORMAT]
{
    "actions": {
        "https://hostname.com/pathname?search=parameter": {
            "records": [
                {
                    "id": "...",
                    "analysis_target": "...",
                    "analysis_surroundings": "...",
                    "relate": [...],
                    "action": "..."
                }
            ],
            "summary": "..."
        }
    },
    "cross_action_flow": "...",
    "intent_evolution": "...",
    "consistent_preferences": "..."
}

[ACTION] RETURN JSON ONLY. NO EXPLANATION. /no_think"###.to_string()
}
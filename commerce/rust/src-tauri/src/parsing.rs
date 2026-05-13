use scraper::{Html, Node, Selector};
use ego_tree::NodeRef;
use regex::Regex;

#[derive(PartialEq, Clone, Copy)]
pub enum PugMode {
    StructureOnly,
    FullContent,
    DetailMode,
    TheadMode,
    ListMode, 
    NoAttributesMode, // 🌟 구조 판별을 위해 HTML의 모든 속성을 완벽히 비워버리는 전용 모드
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
    let re_attr = Regex::new(r#"(?i)\b(id|class|src|href|type|name|value|placeholder|checked|selected|disabled|readonly|rows|cols|rowspan|colspan)(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+))?"#).unwrap();
    
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
    
    // 🌟 [CRITICAL FIX] 상세 모드(Detail Mode) 등에서도 상대 주소를 절대 주소로 치환하기 위한 컨텍스트 주입
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

// 🌟 [HTML5 부모/자식 뎁스 판별 절대 규칙 (Parent Trace Rule)]
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

// 🌟 [CRITICAL FIX] Token Optimizer를 주입받아 앞단을 잘라내고, 필수 부모를 복구하며, 최상위 껍데기를 버리는 완전체 함수
pub fn truncate_pug_by_tokens(pug: &str, max_tokens: usize, tokenizer: &crate::tokenizer::TokenizerModel, bottom_drop_tokens: Option<usize>) -> String {
    let mut lines: Vec<&str> = pug.lines().collect();
    if lines.is_empty() { return String::new(); }

    // 🌟 0. 지능형 트리 스캔 (Pre-scan)
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

    // 🌟 1. 아래서 한 번 자르기 (bottom_drop_tokens 가 주어지면 뒤에서부터 해당 토큰 수만큼 버림)
    if let Some(drop_limit) = bottom_drop_tokens {
        let mut dropped_tokens = 0;
        let mut cut_idx = lines.len();
        
        for i in (0..lines.len()).rev() {
            let line_with_newline = format!("{}\n", lines[i]);
            let token_count = tokenizer.text_encode_vec(line_with_newline, false).map(|v| v.len()).unwrap_or(0);
            
            if dropped_tokens + token_count > drop_limit {
                cut_idx = i + 1;
                break; 
            }
            dropped_tokens += token_count;
            cut_idx = i;
        }
        
        // 🌟 [지능형 보호 개입] 절단선이 보호 구역 한가운데를 지나간다면, 절단선을 구역 밖(아래쪽)으로 밀어내어 구역을 살려냅니다.
        if cut_idx < lines.len() {
            if let Some((_, b_end)) = block_of_line[cut_idx] {
                cut_idx = (b_end + 1).min(lines.len());
            }
        }

        // 문서가 너무 짧아 통째로 날아가는 것을 방지하기 위해 최소 1줄은 남깁니다.
        let safe_cut_idx = cut_idx.min(lines.len().saturating_sub(1));
        lines.truncate(safe_cut_idx);
    }

    // 🌟 2. 위에서 한 번 자르기 (남은 덩어리에서 밑에서부터 max_tokens 만큼 수집하여 앞단을 버림)
    let mut current_tokens = 0;
    let mut start_keep_idx = lines.len();

    for i in (0..lines.len()).rev() {
        let line_with_newline = format!("{}\n", lines[i]);
        let token_count = tokenizer.text_encode_vec(line_with_newline, false).map(|v| v.len()).unwrap_or(0);
        
        if current_tokens + token_count > max_tokens {
            start_keep_idx = i + 1;
            break;
        }
        current_tokens += token_count;
        start_keep_idx = i;
    }
    
    // 🌟 [지능형 보호 개입] 시작선이 보호 구역 한가운데를 지나간다면, 시작선을 구역 꼭대기로 끌어올려 전체 껍데기를 살려냅니다.
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
    if let Some(mut target_indent) = last_valid_indent {
        for i in (0..start_keep_idx).rev() {
            let line = lines[i];
            if line.trim().is_empty() { continue; }
            let current_indent = line.chars().take_while(|c| c.is_whitespace()).count();
            
            if is_root_layout_element(line) {
                break;
            }

            if current_indent < target_indent && !is_void_element(line) {
                final_kept_lines.insert(0, format!("{}\n", line));
                target_indent = current_indent; 
            }
        }
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
    pub base_url: Option<String>, // 🌟 상대 주소를 절대 주소로 치환하기 위한 base_url 추가
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

            // 🌟 PugMode::NoAttributesMode일 때 select, datalist, option, input, textarea 태그와 그 자식들을 원천 제거합니다.
            if *mode == PugMode::NoAttributesMode {
                if ["select", "datalist", "option"].contains(&tag_name.as_str()) {
                    return;
                }
            }

            // Context Management
            if tag_name == "tbody" { if let Some(c) = ctx.as_mut() { c.is_in_tbody = true; c.current_row_idx = 0; } }
            if tag_name == "tr" { if let Some(c) = ctx.as_mut() { c.current_col_idx = 0; } }

            // 🌟 [새로운 태그 축약(평탄화) 로직] 
            // 껍데기 태그 자체가 출력되지 않고 자식에게 뎁스(indent)를 그대로 패스합니다.
            let useless_wrappers = [
                "div", "span", "section", "article", "main", "aside", 
                "header", "footer", "nav", "p", "strong", "b", "em", "i", "center", "font"
            ];
            
            let is_useless = useless_wrappers.contains(&tag_name.as_str());
            
            let has_meaningful_attrs = if *mode == PugMode::NoAttributesMode {
                // 🌟 [CRITICAL FIX] 구조 판별을 위해 colspan, rowspan, scope 속성은 예외적으로 보존합니다.
                element.attrs().any(|(k, _)| ["colspan", "rowspan", "scope"].contains(&k))
            } else {
                element.attrs().any(|(k, _)| {
                    ["src", "href", "type", "name", "value", "placeholder", "checked", "selected", "disabled", "readonly", "rows", "cols", "rowspan", "colspan"].contains(&k) || k.starts_with("data-")
                })
            };

            // 🌟 [빈 태그 원천 삭제 로직] 내부 자식을 깊게 탐색하여 의미 있는 컨텐츠가 있는지 검사합니다.
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
            
            // 🌟 유효한 자식(텍스트, 내부 엘리먼트 등)이 전혀 없는 빈 껍데기 태그는 렌더링하지 않고 즉시 폐기합니다.
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
                "checked", "selected", "disabled", "readonly", "rows", "cols", "rowspan", "colspan"
            ];
            let thead_include = ["scope", "rowspan", "colspan"];

            for (name, value) in element.attrs() {
                if name == "id" || name == "class" || name == "alt" { continue; }

                let should_include = if *mode == PugMode::TheadMode {
                    thead_include.contains(&name)
                } else if *mode == PugMode::NoAttributesMode {
                    // 🌟 [CRITICAL FIX] 구조 판별을 위해 colspan, rowspan, scope 속성은 예외적으로 보존합니다.
                    ["colspan", "rowspan", "scope"].contains(&name)
                } else {
                    name.starts_with("data-") || always_include.contains(&name)
                };

                if should_include {
                    if ["checked", "selected", "disabled", "readonly"].contains(&name) && (value.is_empty() || value == name) {
                        other_attributes.push(name.to_string());
                    } else if !value.is_empty() {
                        let mut safe_value = value.replace("\"", "'");
                        
                        // 🌟 [CRITICAL FIX] href, src 속성에 들어있는 상대 경로(./ 또는 /)를 완벽한 절대 경로로 자동 치환합니다.
                        if name == "href" || name == "src" {
                            if let Some(c) = ctx.as_ref() {
                                if let Some(base) = &c.base_url {
                                    if let Ok(base_url_obj) = url::Url::parse(base) {
                                        // 🌟 지저분한 HTML의 공백(띄어쓰기) 때문에 URL 파싱이 실패하여 상대경로가 그대로 남는 현상을 방어하기 위해 trim() 추가
                                        if let Ok(resolved_url) = base_url_obj.join(safe_value.trim()) {
                                            safe_value = resolved_url.to_string();
                                        }
                                    }
                                }
                            }
                        }

                        other_attributes.push(format!("{}=\"{}\"", name, safe_value));
                    }
                }
            }

            let mut attributes_string = String::new();
            if !other_attributes.is_empty() {
                attributes_string.push_str(&format!("[{}]", other_attributes.join(" ")));
            }

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
            } else {
                for child in node.children() {
                    generate_pug_lines(child, indent_level + 1, output, mode, ctx);
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
                    // 🌟 [CRITICAL FIX 1] 숫자 사이의 콤마(,) 제거 (소수점은 완벽히 보존)
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
                
                // 🌟 [CRITICAL FIX] headers가 없더라도 base_url을 전달받아 절대 주소 치환이 가능하도록 구조체를 무조건 생성합니다.
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
Based on the provided Pug template, identify the primary category of this webpage.

[SCHEMA DEFINITIONS]
- type: The main category. Must be one of:
  - "order": Order list, Order history, Order details, Checkout success.
  - "goods": Product list, product detail.
  - "tracking": Shipment tracking status, delivery history.
  - "review": Product reviews, feedback list.
  - "coupon": Coupon list, discount events.
  - "event": Promotion pages, event announcements.
  - "": If none of the above match.

[OUTPUT FORMAT]
{
    type: String
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

pub fn is_detail_prompt(page_type: &str) -> String {
    let template = r###"[TASK]
Analyze the provided PUG/HTML content from top to bottom. Determine if the main content represents a "{TYPE} Detail/{TYPE} Edit Form/{TYPE} Manage Form" (true) or a "{TYPE} List/Index Page/Home Page/Dashboard Page" (false).

[ENTITY CONTEXT: {TYPE}]
You are evaluating a page managing this specific domain entity. Use this context to conceptually understand the abstract structures:
- has_form: A property configuration interface. It features a large overarching form dedicated to inputting or updating the specific attributes of ONE primary entity.
- has_list: A catalog or inventory interface dedicated to displaying, filtering, or batch-processing multiple DIFFERENT primary entities.

[FORCED DOCUMENT SCANNING LOGIC]
Read the entire document from top to bottom, applying the following strict filters and evaluations:

1. IGNORE:
   - Strictly ignore global navigation, menus, headers, footers, aside, search, filter, form.
2. TARGET:
   - Focus purely on the main data payload where "{CATEGORY}", "{TYPE}", or actual items are listed.
3. EVALUATE:
   - You MUST evaluate the concluding elements at the very bottom of the main content area first. Check for the following:
     A. Does the page terminate with dataset navigation (pagination, "next/prev") or bulk-action execution elements?
     B. Does the main data area consist of a repeating multi-entity grid?
     C. Does the main data area contain an extensive configuration/input form (inputs, textareas, image uploads, save buttons) for a single entity?

[SCHEMA DEFINITIONS]
- {TYPE}:
    - has_list: Boolean. True if the document contains a multi-entity grid, OR if the bottom of main content area has dataset navigation/bulk controls.
    - has_form: Boolean. True if the main data payload is heavily composed of data entry fields (text, select, radio, file uploads) dedicated to creating or updating a single entity.
    - detail: Boolean. True ONLY if has_list is false AND has_form is true.

[OUTPUT FORMAT]
{
  "{TYPE}": {
    "has_list": Boolean,
    "has_form": Boolean,
    "detail": Boolean
  }
}

JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;
    template.replace("{TYPE}", page_type)
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
    let template = r###"Convert the given natural language content into the specified JSON dataset structure by segmenting it into granular semantic chunks.

[DOCUMENT SCANNING & STRICT SEGMENTATION LOGIC]
1. EXACT COPY: Copy the full input sentence into 'original_text' without changing anything.
2. TAGGED PIPE PLANNING: In the 'segmented_plan' field, you MUST prefix every segment with its assigned type tag in brackets, followed by the exact substring, separated by pipes ('|'). Structure it strictly as '[tag1] chunk1 | [tag2] chunk2'.
3. MAXIMAL GROUPING: Group all contiguous words belonging to the same type into a SINGLE segment. DO NOT split subjects from their numeric conditions. Break the segment ONLY when the type logically shifts.
4. STRICT ARRAY MAPPING: For EVERY tagged segment in 'segmented_plan', create exactly one object in the 'context' array sequentially.

[SCHEMA DEFINITIONS]
- original_text: String. The exact, unaltered full natural language input.
- segmented_plan: String. The original text with '[type] text | ' format inserted strictly at type boundaries.
- context:
  - 'text': String.
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
pub fn english_summary_for_fts_prompt(korean_text: &str, page_type: &str) -> String {
    let template = r###"Convert the given Korean natural language content into English by segmenting it into granular semantic chunks.

[DOCUMENT SCANNING & STRICT SEGMENTATION LOGIC]
1. EXACT COPY: Copy the full Korean input sentence into 'original_text' without changing anything.
2. TRANSLATE & TAGGED PIPE PLANNING: Translate the content into English. In the 'segmented_plan' field, you MUST prefix every translated segment with the assigned type tag [{TYPE}], followed by the exact English substring, separated by pipes ('|'). Structure it strictly as '[{TYPE}] english chunk1 | [{TYPE}] english chunk2'.
3. MAXIMAL GROUPING: Group all contiguous words belonging to the same entity into a SINGLE translated segment. Break the segment ONLY when the context logically shifts.
4. STRICT ARRAY MAPPING: For EVERY tagged segment in 'segmented_plan', create exactly one object in the 'context' array sequentially.

[SCHEMA DEFINITIONS]
- original_text: String. The exact, unaltered full Korean natural language input.
- segmented_plan: String. The translated English text with '[{TYPE}] english text | ' format inserted strictly at context boundaries.
- context:
  - 'text': String. The translated English chunk.
  - 'type': String. Always use '{TYPE}'.

[INPUT DATA]
{INPUT}

[OUTPUT FORMAT]
{
  "original_text": "String",
  "segmented_plan": "String",
  "context": [...]
}

[ACTION] JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{INPUT}", korean_text).replace("{TYPE}", page_type)
}

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
    // 🌟 도메인(Type)별로 허용되는 상태(Status) 값을 완벽히 분리하여 매핑합니다.
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

    // 🌟 {STATUS_OPTIONS} 를 먼저 치환한 뒤 {TYPE}을 치환합니다.
    template.replace("{STATUS_OPTIONS}", status_options)
            .replace("{TYPE}", seg_type)
            .replace("{TEXT}", current_text)
}


pub fn item2json(page_type: &str, href: &str, language: &str) -> String {
    let schema = match page_type {
    "tracking" => r###"- "{TYPE}":
    - "id":tracking number | string
    - "link":'{HREF}' | string
    - "status":'draft' or 'progress' or 'return' or 'complete' or 'error' | string
    - "title":tracking product title | string
    - "registration_date":yyyy-MM-ddThh:mm:ss | string
    - "shipping_date":yyyy-MM-ddThh:mm:ss | string
    - "sender_name":sender_name | string
    - "sender_address":sender_address | string
    - "sender_phone":sender_phone | string
    - "recipient_name":recipient_name | string
    - "recipient_address":recipient_address | string
    - "recipient_phone":recipient_phone | string
    - "width":Package width | number
    - "height":Package height | number
    - "length":Package length | number
    - "weight":Package weight | number
    - "carrier":carrier name translated into English | string
    - "shipping_fee":Shipping cost | number
    - "shipping_method":'standard' or 'express' or 'same_day' or 'pick_up' or 'freight' or 'prepaid' | string
    - "shipping_duration":Estimated delivery days | number
    - "bundle_shipping":Allow combined shipping | string"###.to_string(),
    "goods" => r###"- "{TYPE}":
    - "id":Refer to the ID value from the link | string
    - "link":'{HREF}' | string
    - "code":product code | string
    - "status":'show' or 'remove' or 'hide' or 'stop' or 'exchange' or 'expire' | string
    - "title":product name | string
    - "registration_date":yyyy-MM-ddThh:mm:ss | string
    - "payment_method":payment method | string
    - "bank":bank company name or '' | string
    - "card":card company name or '' | string
    - "model_name":product Model name | string
    - "brand_name":product Brand name | string
    - "condition":['new' or 'used' or 'lease' or 'rental' or 'refurbish']
    - "description":product Full description (HTML allowed) | string
    - "short_description":product short description | string
    - "tags":[{ tag : product keyword or tag | string }]
    - "origin_country":product Country of origin/manufacture | string
    - "manufacturer":product Manufacturer name | string
    - "release_date":Product release date(yyyy-MM-ddThh:mm:ss) | string
    - "manufacture_date":product Date(yyyy-MM-ddThh:mm:ss) of manufacture | string
    - "expiration_date":product Expiration or use-by date(yyyy-MM-ddThh:mm:ss) | string
    - "gtin":product Global Trade Item Number | string
    - "mpn":product Manufacturer Part Number | string
    - "barcode":product Barcode value | string
    - "sale_price":product sale price | number
    - "supply_price":product supply price | number
    - "currency":ISO 4217 Currency Code | string
    - "compare_at_price":product Original price for showing discounts | number
    - "quantity":product Inventory quantity | number
    - "stock_keeping_unit":Stock Keeping Unit | string
    - "low_stock_threshold":product Low stock alert threshold | number
    - "unit":product Selling unit | string
    - "tax_included":product Whether tax | number
    - "tax_code":product Tax code for region-specific rules | string
    - "main_image_url":Main product image URL | string
    - "additional_image_url":additional product image URL | string
    - "video_url":product Promotional video URL | string
    - "carrier":product carrier name translated into English | string
    - "shipping_fee":product Shipping cost | number
    - "shipping_method":'standard' or 'express' or 'same_day' or 'pick_up' or 'freight' or 'prepaid' | string
    - "shipping_duration":product Estimated delivery days | number
    - "bundle_shipping":product Allow combined shipping | string
    - "width":Package width(cm) | number
    - "height":Package height(cm) | number
    - "length":Package length(cm) | number
    - "weight":Package weight(kg) | number
    - "options":[ 
        { 
            value:product option name | string, 
            inputs:[
                { value:product option input value | string }
            ] 
        }
    ]
    - "additional_goods":[ 
        { 
            path:value:URL includes a additional product manage path, an administrative or additional product edit Link | string, 
            id:Refer to the additional product no value from the link or an attribute or additional product input value | string, 
            link:value:Refer to the ID to find a URL that includes a additional product manage link | string
        }
    ]"###.to_string(),

        "order" => r###"- "{TYPE}":
    - "id":Refer to the ID value from the link | string
    - "link":'{HREF}' | string
    - "tracking_number":tracking number | string
    - "status":'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' | string
    - "registration_date":yyyy-MM-ddThh:mm:ss | string
    - "goods":[{ title:{ value:product title | string }, path:{ value:URL includes a manage path, an administrative or edit Link | string }, id:{ value:Refer to the product no value from the link or an attribute or input value | string }, link:{ value:Refer to the ID to find a URL that includes a manage link | string } }]
    - "sender_name":sender_name | string
    - "sender_address":sender_address, Filter the addresses to District-level and up | string
    - "sender_phone":sender_phone | string
    - "recipient_name":recipient_name | string
    - "recipient_address":recipient_address, Filter the addresses to District-level and up | string
    - "recipient_phone":recipient_phone | string
    - "bank":bank company name | string
    - "card":card company name | string
    - "order_date":order date | string
    - "payment_date":payment date or '' | string
    - "payment_method":'C.O.D.' or 'CARD' or 'BANK' or '' | string
    - "payment_origin":Payment Gateway Service Name or '' | string"###.to_string(),
    "coupon" | "event" => r###"- "{TYPE}":
    - "id":Refer to the ID value from the link | string
    - "link":'{HREF}' | string
    - "type":'percentage' or 'fixed_amount' or 'free_shipping' or '' | string
    - "status":'draft' or 'progress' or 'stop' or 'cancel' or 'expire' or 'complete' or 'error' | string
    - "title":{TYPE} title | string
    - "started_at":yyyy-MM-ddThh:mm:ss | string
    - "expired_at":yyyy-MM-ddThh:mm:ss | string
    - "code":{TYPE} code used at checkout | string
    - "discount":Discount value | number
    - "quantity":{TYPE} quantity | number
    - "usage_limit":Total usage limit for the coupon | number
    - "usage_per":Usage limit per customer | number
    - "new_customer_only":new customer only | boolean
    - "first_purchase_only":first purchase only | boolean
    - "min_order_amount":Minimum order amount required to apply coupon | number
    - "max_order_amount":Maximum order amount allowed to apply coupon | number
    - "max_discount_amount":Maximum discount limit allowed for the coupon | number
    - "region_restrictions":region restrictions | boolean
    - "number":contact phone number | string
    - "address":offline location address | string
    - "registration_date":yyyy-MM-ddThh:mm:ss | string"###.to_string(),
    "review" => r###"- "{TYPE}":
    - "id":Refer to the ID value from the link | string
    - "link":'{HREF}' | string
    - "status":'progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' | string
    - "name":reviewer name | string
    - "title":reviewer item title | string
    - "completed":order complete | boolean
    - "registration_date":yyyy-MM-ddThh:mm:ss | string"###.to_string(),
        _ => r###"- "{TYPE}":
    - "id": Unique identifier | string
    - "title": General name or title | string
    - "status": Current state | string"###.to_string()
    };

    let template = r###"[TASK]
Extract detailed information from the provided Pug template into a single structured JSON object.

[CONTEXT]
Page Type: {TYPE}
current Link: {HREF}

[SCHEMA DEFINITIONS]
{SCHEMA}

[EXTRACTION RULES]
1. Return ONLY valid JSON. No preamble, no postscript.
2. If a field is missing in the data, use null.
3. Normalize all dates to 'yyyy-MM-ddThh:mm:ss'.
4. Extract only numeric values for price, amount, weight, and dimensions.
5. Do NOT make up data. Only extract what is present in the Pug structure.

[OUTPUT FORMAT]
{...}

[ACTION] RETURN JSON ONLY. 
NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{SCHEMA}", &schema)
            .replace("{TYPE}", page_type)
            .replace("{HREF}", href)
            .replace("{LANGUAGE}", language)
}

pub fn list2json(page_type: &str, href: &str, language: &str, head_pug: &str, item_pug: &str) -> String {
    let schema = match page_type {
    "order" => r###"- "order":
    - "path":{HREF}
    - "id":Refer to the ID value from the link or an attribute | string
    - "link":Refer to the ID to find a URL that includes a manage order link or tracking detail link | string
    - "currency":ISO 4217 Currency Code | string
    - "sale_price":sale price | number
    - "tracking_number":tracking Number or shipping number | string
    - "registration_date":yyyy-MM-ddThh:mm:ss | string
    - "status":tracking status('progress' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete') | string
    - "title":title | string"###.to_string(),

    "goods" => r###"- "goods":
    - "path":{HREF}
    - "id":Refer to the ID value from the link or an attribute | string
    - "link":Refer to the ID to find a URL that includes a manage product link | string
    - "currency":ISO 4217 Currency Code | string
    - "sale_price":sale price | number
    - "registration_date":yyyy-MM-ddThh:mm:ss | string
    - "status":'show' or 'remove' or 'hide' or 'stop' or 'exchange' or 'expire' | string
    - "title":title | string"###.to_string(),
    
    "tracking" | "review" => r###"- "{TYPE}":
    - "path":{HREF}
    - "id":Refer to the ID value from the link or an attribute | string
    - "link":Refer to the ID to find a URL that includes a manage {TYPE} link | string
    - "registration_date":yyyy-MM-ddThh:mm:ss | string
    - "status":'start' or 'progress' or 'stop' or 'cancel' or 'return' | string
    - "title":author and content | string"###.to_string(),
    
    "coupon" | "event" => r###"- "{TYPE}":
    - "path":{HREF}
    - "id":Refer to the ID value from the link or an attribute | string
    - "link":Refer to the ID to find a URL that includes a manage {TYPE} link | string
    - "started_at":yyyy-MM-ddThh:mm:ss | string
    - "expired_at":yyyy-MM-ddThh:mm:ss | string
    - "registration_date":yyyy-MM-ddThh:mm:ss | string
    - "status":'show' or 'progress' or 'hide' or 'stop' or 'cancel' or 'expire' or 'complete' | string
    - "title": title | string"###.to_string(),
    
        _ => r###"- "{TYPE}":
    - "path":{HREF}
    - "id":Refer to the ID value from the link or an attribute | string
    - "link":Refer to the ID to find a URL that includes a manage {TYPE} link | string
    - "registration_date":yyyy-MM-ddThh:mm:ss | string
    - "status":'show' or 'progress' or 'remove' or 'hide' or 'stop' or 'cancel' or 'refund' or 'return' or 'exchange' or 'expire' or 'complete' | string
    - "title":title | string"###.to_string()
    };

    // 🌟 [CRITICAL FIX] item_pug 내용에 링크(href)가 없을 경우 스키마에서 link와 path 요구 조건을 동적으로 제거합니다.
    let mut final_schema = schema;
    if !item_pug.contains("href=") && !item_pug.contains("href=\"") {
        final_schema = final_schema.lines()
            .filter(|line| !line.contains("\"link\":") && !line.contains("\"path\":"))
            .collect::<Vec<_>>()
            .join("\n");
    }

    // 🌟 [최종 반영] .txt 파일 구조와 동일하게 thead/tbody 태그 및 계층형 들여쓰기 적용
    let mut final_pug = String::new();

    // 1. 헤더 영역 (thead 태그 추가 및 내부 1단계 들여쓰기)
    if !head_pug.is_empty() {
        final_pug.push_str("thead\n");
        for line in head_pug.lines() {
            final_pug.push_str(&format!("    {}\n", line));
        }
    }

    // 2. 바디 영역 (tbody 태그 추가 및 내부 1단계 들여쓰기)
    if !item_pug.is_empty() {
        final_pug.push_str("tbody\n");
        for line in item_pug.lines() {
            final_pug.push_str(&format!("    {}\n", line));
        }
    }

    // 3. 전체 PUG를 프롬프트에 넣기 위해 일괄적으로 4칸 들여쓰기 적용
    let pug_content = final_pug.trim_end().lines()
        .map(|line| format!("    {}", line))
        .collect::<Vec<_>>()
        .join("\n");

    let template = r###"[TASK]
Extract detailed information from the provided Pug tbody into a single structured JSON object.

[PUG CONTENT]
{PUG_CONTENT}

[CONTEXT]
current Link: {HREF}
Language: {LANGUAGE}

[SCHEMA DEFINITIONS]
{SCHEMA}

[OUTPUT FORMAT]
{...}

[ACTION] RETURN JSON ONLY. 
NO EXPLANATION. NO THINKING. /no_think"###;

    // 🌟 [CRITICAL FIX] moved 에러 해결: 기존 schema 대신 조건부로 처리된 final_schema를 참조합니다.
    template.replace("{SCHEMA}", &final_schema)
            .replace("{TYPE}", page_type)
            .replace("{HREF}", href)
            .replace("{LANGUAGE}", language)
            .replace("{PUG_CONTENT}", &pug_content)
}

/// Converts a JSON Value into a human-readable natural language narrative.
/// [STRICT ALIGNMENT] This logic perfectly synchronizes with every column in `parsing.rs`.
pub fn json_to_natural_language(json_val: &serde_json::Value) -> String {
    let mut output = String::new();

    match json_val {
        serde_json::Value::Object(map) => {
            for (key, val) in map {
                // 1. null 이나 빈 값은 검색에 도움 안 되므로 무시
                if val.is_null() || (val.is_string() && val.as_str().unwrap_or("").trim().is_empty()) {
                    continue;
                }

                // 2. Key 이름을 읽기 좋게 정제 (sale_price -> sale price)
                let clean_key = key.replace("_", " ");

                // 3. Value가 객체나 배열일 경우 내부로 재귀 탐색
                if val.is_object() {
                    let sub_text = json_to_natural_language(val);
                    if !sub_text.is_empty() {
                        output.push_str(&format!("{}: {}. ", clean_key, sub_text));
                    }
                } else if val.is_array() {
                    let arr = val.as_array().unwrap();
                    let mut arr_items = Vec::new();
                    for item in arr {
                        let sub_text = json_to_natural_language(item);
                        if !sub_text.is_empty() {
                            arr_items.push(sub_text);
                        }
                    }
                    if !arr_items.is_empty() {
                        output.push_str(&format!("{}: [{}]. ", clean_key, arr_items.join(", ")));
                    }
                } else {
                    // 4. 일반 문자열/숫자일 경우 최종 조립
                    let val_str = match val {
                        serde_json::Value::String(s) => s.clone(),
                        serde_json::Value::Number(n) => n.to_string(),
                        serde_json::Value::Bool(b) => b.to_string(),
                        _ => String::new(),
                    };
                    output.push_str(&format!("{}: {}. ", clean_key, val_str));
                }
            }
        },
        serde_json::Value::Array(arr) => {
            for item in arr {
                let sub = json_to_natural_language(item);
                if !sub.is_empty() {
                    output.push_str(&format!("{} ", sub));
                }
            }
        },
        _ => {
            if let Some(s) = json_val.as_str() {
                output.push_str(s);
            } else {
                output.push_str(&json_val.to_string());
            }
        }
    }

    output.trim().to_string()
}

pub fn normalize_to_json_string(input: &str) -> String {
    let mut s = input.replace(&['\u{00A0}', '\u{200B}', '\u{202F}', '\u{FEFF}'][..], " ").trim().to_string();

    // 1. Backticks to quotes
    let re_backtick = Regex::new(r"`([\s\S]*?)`").unwrap();
    s = re_backtick.replace_all(&s, |caps: &regex::Captures| {
        format!("\"{}\"", caps[1].replace("\"", "\\\""))
    }).to_string();

    // 2. Key quotes correction (key: -> "key":)
    let re_keys = Regex::new(r"([{,])\s*([a-zA-Z0-9_]+)\s*:").unwrap();
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

    // 🌟 [추가] LLM이 뱉어낸 말줄임표(..., ...) 가비지를 닫는 괄호 앞에서 깔끔하게 제거합니다.
    let re_artifact = Regex::new(r",?\s*\.\.\.\s*([\]}])").unwrap();
    s = re_artifact.replace_all(&s, "$1").to_string();

    // 🌟 6. Force close braces, arrays, and strings (Stack-based repair)
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
        // 🌟 닫는 괄호가 없어도 여는 괄호 이후부터 끝까지 가져와 복구 시도
        extracted = clean_text[start..].to_string();
    } else if let Some(start) = clean_text.find("[") {
        if let Some(end) = clean_text.rfind("]") {
            if start < end {
                let attempt = clean_text[start..=end].to_string();
                if let Ok(v) = serde_json::from_str(&attempt) { return v; }
            }
        }
        // 🌟 닫는 괄호가 없어도 여는 괄호 이후부터 끝까지 가져와 복구 시도
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

        // 🌟 [추가] Absolute Final Fallback: 맨 끝 글자를 하나씩 지워가며 에러가 나지 않을 때까지 파싱을 재시도합니다.
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
Your task is to analyze the [Reference: Row Structure] and locate both its body container and its corresponding header container within the [PUG CONTENT].
Generate the exact CSS selectors for both containers.

[Rules]
1. Analyze Reference: Carefully examine the [Reference: Row Structure] to understand the context, structure, and attributes of a single data item/row.
2. Locate Body Container: The `tbody` selector is already provided in the output format as "{ITEM_SELECTOR}". You MUST return it exactly as is. DO NOT modify it.
3. Locate Header Container: Identify the wrapper or container element that acts as the "Header" for this list (the element containing the column titles or labels). Use the body container selector ("{ITEM_SELECTOR}") as a reference point to locate the matching header container, which is typically placed just above the body container or as a sibling. Generate a precise CSS selector for this header container.
4. Tag Agnostic: Do NOT assume the structure uses traditional <table>, <thead>, or <tbody> tags. It could be built using <div>, <ul>/<li>, or other semantic tags. Analyze the relationship logically.
5. Strict JSON Output: Output the result strictly in valid JSON format exactly matching the structure below. Do not include any other text, markdown formatting, or explanations.

[Expected Output Format]
{
  "{TYPE}" : {
    "table" : {
      "tbody" : {
        "selector" : "{ITEM_SELECTOR}"
      },
      "thead" : {
        "selector" : "..."
      }
    }
  }
}

[ACTION] RETURN JSON ONLY. NO EXPLANATION. NO THINKING. /no_think"###;

    template.replace("{TYPE}", page_type)
            .replace("{ITEM_SELECTOR}", item_selector)
            .replace("{PUG_CONTENT}", pug_content)
            .replace("{REFERENCE_ROW}", reference_row)
}


pub fn analytic_report_prompt() -> String {
    r###"## Role
You are a **User Behavior Analysis Expert**. Your goal is to interpret raw HTML interactions to understand the user's specific intent and analyze the selection context within a list.

## Task
Analyze the **one or more pairs** of 'Clicked HTML' (the selected item) and 'Related HTML' (the surrounding list structure). The inputs are provided as parallel arrays of HTML strings, meaning the Nth item in the 'Clicked HTML' array corresponds to the Nth item in the 'Related HTML' array.

**For each pair**, perform the analysis according to the guidelines below.  
If 'Previous Analysis' is provided, use it to infer the user's behavioral flow.

Use this information to connect the previous action with the current click.

## Analysis Guidelines
1. **action (User Intent)**
    - Determine the specific user intent for clicking the item.
    - If 'Previous Analysis' exists, assume a continuous flow (e.g., "Search" -> "Select Result") to refine the intent.
    - **Must explicitly include the 상품 제목(product title) 및 옵션 정보(option attributes) extracted from the clicked HTML.**
    - Output as a short verb phrase (Korean).

2. **relate (Neighboring Items Context)**
    - Treat 'Related HTML' as a **list or collection of items** where the user made a selection.
    - Identify **neighboring items** (siblings) that were displayed near the clicked element but *not* selected.
    - Summarize these surrounding items to capture the context of the choice (e.g., competitors, other options, or list categories).
    - **Constraint**: Do not summarize the clicked item itself in this field; focus on what surrounds it.

3. **summary (Page-level Goal)**
    - Provide a detailed explanation of what the user aimed to accomplish on this page.
    - **Must explicitly reference the 상품 제목(product title) 및 옵션 정보(option attributes) to explain the user’s goal.**

## Output Format
Output ONLY a raw JSON object, where the outer structure is an array of analysis objects, corresponding to each analyzed pair:
{
    "actions": {
        "https://hostname.com/pathname?search=parameter": {
            "records": [
                {
                    "id": "String",
                    "relate": ["String"],
                    "action": "String"
                }
            ],
            "summary": "String"
        }
    },
    "cross_action_flow": "String",
    "intent_evolution": "String",
    "consistent_preferences": "String"
}
"###.to_string()
}
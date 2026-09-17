use scraper::{Html, Node, Selector};
use ego_tree::NodeRef;
use regex::Regex;
use serde_json::Value;

pub use crate::prompts::*; // 🌟 분리된 프롬프트 함수들을 외부에서 그대로 사용할 수 있도록 재수출

use crate::tokenizer;

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
    let cleaned: String = text.chars()
        .filter(|c| {
            let u = *c as u32;
            // 개행/캐리지리턴/탭은 PUG 들여쓰기 구조 그 자체이므로 무조건 보존
            if u == 9 || u == 10 || u == 13 { return true; }
            // C0 / C1 제어문자 제거
            if u < 0x20 || (0x7F..=0x9F).contains(&u) { return false; }
            // BOM / zero-width / word-joiner : 토크나이저가 단어를 쪼개는 원인
            if matches!(u, 0xFEFF | 0x200B | 0x200C | 0x200D | 0x2060) { return false; }
            // bidi override : 시각 순서를 조작하는 프롬프트 인젝션 문자
            if (0x202A..=0x202E).contains(&u) || (0x2066..=0x2069).contains(&u) { return false; }
            // private use area : 폰트 깨짐 잔재
            if (0xE000..=0xF8FF).contains(&u) { return false; }
            true
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
    // 🌟 [COLUMN SIGNAL 보존] 추가된 속성의 근거
    //    headers : HTML 표준에서 '이 셀의 열 헤더가 누구인지' 를 셀이 직접 선언하는 속성.
    //              컬럼명 추출에서 가장 신뢰도가 높은데 여기서 지워지고 있었습니다.
    //    abbr    : th 의 짧은 정식 명칭. 'UNIT WEIGHT' 대신 'net weight' 가 실려 옵니다.
    //    alt/title: 아이콘 컬럼(중량/수량)의 유일한 텍스트 단서.
    //    data-*  : generate_pug_lines 의 has_meaningful_attrs / always_include 가
    //              starts_with("data-") 를 검사하는데, 여기서 먼저 지워져
    //              그 분기가 도달 불가능한 죽은 코드가 되어 있었습니다.
    //    뒤쪽 (?=[\s/>]|$) 는 format= 안의 for, formaction= 안의 for 처럼
    //    접두사만 걸리는 오검출을 차단합니다.
    let re_attr = Regex::new(r#"(?i)\b(data-[a-z0-9\-]+|placeholder|rowspan|colspan|disabled|readonly|selected|summary|headers|checked|class|scope|title|value|abbr|href|type|name|rows|cols|alt|for|src|id)\b(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+))?"#).unwrap();
    
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

#[derive(Default, Clone)]
pub struct TableContext {
    pub headers: Vec<Vec<String>>, // Row -> Col -> Title
    pub current_row_idx: usize,
    pub current_col_idx: usize,
    pub is_in_tbody: bool,
    pub base_url: Option<String>,
    pub doc_lang: String, // 🌟 canonicalize_trade_column 언어 필터용. 빈 문자열이면 전체 매칭.
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

            // 🌟 [COLUMN LABEL v2] 헤더 행을 '본문 행 번호 % 헤더 행 수' 로 고르던 식은 두 가지로 틀렸습니다.
            //    ① 2단 헤더 표에서 본문 2행부터 라벨이 통째로 어긋납니다.
            //       (본문 r0→헤더 r0, 본문 r1→헤더 r1, 본문 r2→헤더 r0 … 순환)
            //    ② split_doc_to_pug_list_advanced 는 행마다 ctx 를 새로 만들어
            //       current_row_idx 가 항상 0 입니다. 결국 언제나 헤더 첫 행만 쓰게 되어
            //       'UNIT / VALUE' 2단 헤더에서 상위 'UNIT' 만 붙고 'VALUE' 가 사라집니다.
            //    한 열의 라벨은 '그 열의 모든 헤더 행을 위에서 아래로 이어붙인 것' 하나뿐입니다.
            //    인쇄 라벨과 스키마 필드명을 파이프로 함께 실어 LLM 이 매핑을 추측하지 않게 합니다.
            if tag_name == "td" || tag_name == "th" {
                if let Some(c) = ctx.as_mut() {
                    if c.is_in_tbody && !c.headers.is_empty() {
                        let col = c.current_col_idx;
                        let mut parts: Vec<&str> = Vec::new();
                        for h_row in c.headers.iter() {
                            if let Some(seg) = h_row.get(col) {
                                let seg = seg.trim();
                                if !seg.is_empty() && !parts.contains(&seg) {
                                    parts.push(seg);
                                }
                            }
                        }
                        let title = parts.join(" ");
                        if !title.is_empty() {
                            let safe = title.replace("\"", "'");
                            let canonical = canonicalize_trade_column(&title, &c.doc_lang);
                            if canonical.is_empty() {
                                other_attributes.push(format!("alt=\"{}\"", safe));
                            } else {
                                other_attributes.push(format!("alt=\"{}|{}\"", safe, canonical));
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
            // 🌟 [COLSPAN CURSOR] 셀 하나를 언제나 1열로 세면 colspan 셀 이후의 모든 열이
            //    한 칸씩 밀려 alt 라벨이 옆 열 것으로 붙습니다.
            //    (예: '금액' 이 2열을 덮으면 그 뒤 '단가' 셀에 '수량' 라벨이 붙습니다)
            //    실제 점유 열 수만큼 전진시켜 헤더 격자와 본문 격자를 같은 좌표계로 맞춥니다.
            if tag_name == "td" || tag_name == "th" {
                let span = element.attr("colspan")
                    .and_then(|v| v.trim().parse::<usize>().ok())
                    .unwrap_or(1)
                    .clamp(1, 64);
                if let Some(c) = ctx.as_mut() { if c.is_in_tbody { c.current_col_idx += span; } }
            }
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

pub static TRADE_COLUMN_ALIASES: &[(&str, &[&str])] = &[
    ("description", &[
        // English
        "description of goods", "description", "goods description", "commodity",
        "commodity description", "description of merchandise", "article", "articles",
        "item description", "product", "product name", "name of goods", "nature of goods",
        // Korean
        "품명", "상품명", "품목", "화물명", "물품명",
        // Chinese (Simplified)
        "货物描述", "品名", "商品描述", "货物名称", "品目名称", "商品名",
        // Chinese (Traditional)
        "貨物描述", "品名", "商品描述", "貨物名稱",
        // Japanese
        "品名", "品目", "貨物名", "商品名", "品目名",
        // Spanish
        "descripción de mercancías", "descripción", "mercancía", "producto",
        // German
        "warenbezeichnung", "beschreibung", "warenbeschreibung", "artikel",
        // French
        "désignation des marchandises", "désignation", "description des marchandises",
        // Portuguese
        "descrição das mercadorias", "descrição",
        // Italian
        "descrizione delle merci", "descrizione", "merce",
        // Dutch
        "omschrijving van goederen", "omschrijving", "productomschrijving",
        // Czech
        "popis zboží", "popis", "název zboží",
        // Arabic
        "وصف البضائع", "الوصف", "اسم البضاعة",
    ]),
    ("hs_code", &[
        // English
        "hs code", "h s code", "hs-code", "hscode", "hts code", "hs tariff",
        "tariff code", "commodity code", "hs no", "hs number", "tariff no",
        // Korean
        "세번", "세번부호", "hs부호",
        // Chinese (Simplified)
        "海关编码", "税则号", "商品编码", "hs编码",
        // Chinese (Traditional)
        "海關編碼", "稅則號", "商品編碼",
        // Japanese
        "税番", "関税番号", "統計番号", "hs番号",
        // Spanish
        "código arancelario", "partida arancelaria",
        // German
        "zolltarifnummer", "warennummer", "tarifnummer",
        // French
        "code tarifaire", "position tarifaire", "nomenclature",
        // Portuguese
        "código tarifário", "posição tarifária",
        // Italian
        "codice doganale", "numero tariffario",
        // Dutch
        "tariefnummer", "douanecode",
        // Czech
        "celní kód", "kód zboží",
        // Arabic
        "الرمز الجمركي", "رقم التعريفة",
    ]),
    ("country_of_manufacture", &[
        // English
        "country of manufacture", "country of origin", "made in", "origin",
        "manufacturing country", "country", "coo",
        // Korean
        "원산지", "생산국", "제조국",
        // Chinese (Simplified)
        "原产国", "制造国", "产地", "原产地",
        // Chinese (Traditional)
        "原產國", "製造國", "產地",
        // Japanese
        "原産国", "製造国", "生産国",
        // Spanish
        "país de origen", "país de fabricación", "origen",
        // German
        "herkunftsland", "ursprungsland", "herstellungsland",
        // French
        "pays d'origine", "pays de fabrication", "origine",
        // Portuguese
        "país de origem", "país de fabricação",
        // Italian
        "paese di origine", "paese di fabbricazione",
        // Dutch
        "land van oorsprong", "land van herkomst",
        // Czech
        "země původu", "země výroby",
        // Arabic
        "بلد المنشأ", "بلد الصنع",
    ]),
    ("unit", &[
        // English
        "unit of measure", "unit of measurement", "uom", "measure", "unit",
        "packing unit", "measurement unit",
        // Korean
        "단위", "거래단위",
        // Chinese (Simplified)
        "计量单位", "单位", "计量",
        // Chinese (Traditional)
        "計量單位", "單位",
        // Japanese
        "単位", "計量単位", "梱包単位",
        // Spanish
        "unidad de medida", "unidad",
        // German
        "maßeinheit", "einheit", "mengeneinheit",
        // French
        "unité de mesure", "unité",
        // Portuguese
        "unidade de medida", "unidade",
        // Italian
        "unità di misura", "unità",
        // Dutch
        "maateenheid", "eenheid",
        // Czech
        "měrná jednotka", "jednotka",
        // Arabic
        "وحدة القياس", "الوحدة",
    ]),
    ("quantity", &[
        // English
        "qty", "quantity", "q ty", "pieces", "pcs", "no of pcs", "number of units",
        "number of pieces", "shipped qty",
        // Korean
        "수량", "주문수량",
        // Chinese (Simplified)
        "数量", "件数", "个数", "数目",
        // Chinese (Traditional)
        "數量", "件數", "個數",
        // Japanese
        "数量", "個数", "点数",
        // Spanish
        "cantidad", "unidades",
        // German
        "menge", "anzahl", "stückzahl",
        // French
        "quantité", "nombre",
        // Portuguese
        "quantidade",
        // Italian
        "quantità", "numero di pezzi",
        // Dutch
        "aantal", "hoeveelheid",
        // Czech
        "množství", "počet",
        // Arabic
        "الكمية", "عدد القطع",
    ]),
    ("item_net_weight", &[
        // English
        "unit weight", "net weight", "n w", "nw", "net wt", "unit net weight",
        "weight per unit", "net weight kg",
        // Korean
        "순중량", "단위중량", "개당중량",
        // Chinese (Simplified)
        "净重", "单位净重", "净重量",
        // Chinese (Traditional)
        "淨重", "單位淨重",
        // Japanese
        "正味重量", "正味", "単位重量",
        // Spanish
        "peso neto", "peso neto unitario",
        // German
        "nettogewicht", "rein gewicht",
        // French
        "poids net",
        // Portuguese
        "peso líquido", "peso líquido unitário",
        // Italian
        "peso netto", "peso netto unitario",
        // Dutch
        "nettogewicht", "netto gewicht",
        // Czech
        "čistá hmotnost", "netto hmotnost",
        // Arabic
        "الوزن الصافي", "وزن الوحدة الصافي",
    ]),
    ("item_gross_weight", &[
        // English
        "gross weight", "g w", "gw", "gross wt", "total weight", "gross weight kg",
        // Korean
        "총중량", "총 중량",
        // Chinese (Simplified)
        "毛重", "总重量", "总重",
        // Chinese (Traditional)
        "毛重", "總重量",
        // Japanese
        "総重量", "総量", "全重量",
        // Spanish
        "peso bruto",
        // German
        "bruttogewicht", "rohgewicht",
        // French
        "poids brut",
        // Portuguese
        "peso bruto",
        // Italian
        "peso lordo",
        // Dutch
        "brutogewicht", "totaalgewicht",
        // Czech
        "hrubá hmotnost", "celková hmotnost",
        // Arabic
        "الوزن الإجمالي", "الوزن القائم",
    ]),
    ("unit_price", &[
        // English
        "unit value", "unit price", "price per unit", "u price", "unit cost",
        "rate", "price",
        // Korean
        "단가",
        // Chinese (Simplified)
        "单价", "单位价格", "每个价格",
        // Chinese (Traditional)
        "單價", "單位價格",
        // Japanese
        "単価", "単位価格",
        // Spanish
        "precio unitario", "precio por unidad",
        // German
        "einzelpreis", "stückpreis", "preis je einheit",
        // French
        "prix unitaire",
        // Portuguese
        "preço unitário",
        // Italian
        "prezzo unitario", "prezzo per unità",
        // Dutch
        "eenheidsprijs", "prijs per eenheid",
        // Czech
        "jednotková cena", "cena za kus",
        // Arabic
        "سعر الوحدة", "ثمن الوحدة",
    ]),
    ("total_price", &[
        // English
        "total value", "total price", "line total", "extended price", "extended value",
        "total amount", "amount", "value",
        // Korean
        "금액", "합계", "공급가액",
        // Chinese (Simplified)
        "总价", "总金额", "金额", "合计", "总值",
        // Chinese (Traditional)
        "總價", "總金額", "金額", "合計",
        // Japanese
        "総額", "合計金額", "金額", "合計",
        // Spanish
        "importe total", "valor total", "total", "importe",
        // German
        "gesamtbetrag", "gesamtsumme", "betrag", "gesamtwert",
        // French
        "montant total", "valeur totale", "total",
        // Portuguese
        "valor total", "montante total",
        // Italian
        "importo totale", "valore totale", "totale",
        // Dutch
        "totaalbedrag", "totale waarde", "totaal",
        // Czech
        "celková částka", "celková hodnota",
        // Arabic
        "المبلغ الإجمالي", "القيمة الإجمالية",
    ]),
    ("item_code", &[
        // English
        "item code", "item no", "item number", "sku", "part number", "part no",
        "model", "model no", "article no", "product code", "style no",
        // Korean
        "품번", "모델", "품목코드",
        // Chinese (Simplified)
        "货号", "型号", "产品编号", "物料编号", "款号",
        // Chinese (Traditional)
        "貨號", "型號", "產品編號",
        // Japanese
        "品番", "型番", "製品番号", "品目コード",
        // Spanish
        "código de artículo", "número de artículo", "referencia",
        // German
        "artikelnummer", "artikelnr", "teilenummer",
        // French
        "numéro d'article", "référence article",
        // Portuguese
        "código do artigo", "número do artigo",
        // Italian
        "codice articolo", "numero articolo",
        // Dutch
        "artikelnummer", "productcode",
        // Czech
        "kód položky", "číslo položky",
        // Arabic
        "رمز الصنف", "رقم القطعة",
    ]),
    ("item_package_count", &[
        // English
        "packages", "no of packages", "package count", "cartons", "ctns",
        "no of cartons", "number of packages", "case",
        // Korean
        "포장수", "박스수", "포장개수",
        // Chinese (Simplified)
        "包装数", "箱数", "包装件数", "件数",
        // Chinese (Traditional)
        "包裝數", "箱數", "包裝件數",
        // Japanese
        "梱包数", "箱数", "個数", "パッケージ数",
        // Spanish
        "número de bultos", "bultos", "cajas",
        // German
        "anzahl der packstücke", "packstücke", "kartons",
        // French
        "nombre de colis", "colis", "cartons",
        // Portuguese
        "número de volumes", "volumes", "caixas",
        // Italian
        "numero di colli", "colli", "cartoni",
        // Dutch
        "aantal pakketten", "pakketten", "dozen",
        // Czech
        "počet balení", "balení", "krabice",
        // Arabic
        "عدد الطرود", "الطرود", "الصناديق",
    ]),
    ("item_package_type", &[
        // English
        "package type", "kind of package", "packing", "type of package",
        "packing type", "kind of packages",
        // Korean
        "포장형태", "포장종류",
        // Chinese (Simplified)
        "包装类型", "包装方式", "包装形式",
        // Chinese (Traditional)
        "包裝類型", "包裝方式",
        // Japanese
        "梱包形態", "梱包種類", "包装形態",
        // Spanish
        "tipo de embalaje", "tipo de envase",
        // German
        "verpackungsart", "verpackungstyp",
        // French
        "type d'emballage", "nature de l'emballage",
        // Portuguese
        "tipo de embalagem",
        // Italian
        "tipo di imballaggio",
        // Dutch
        "verpakkingstype", "type verpakking",
        // Czech
        "typ balení", "druh balení",
        // Arabic
        "نوع التعبئة", "طريقة التغليف",
    ]),
];
/// 🌟 [LABEL ECHO] 서식의 '박스 라벨' 목록.
///  실측 로그에서 reference_invoice 가 "CONSIGNEE VAT/EORI" 를,
///  party_name 이 "SIGNATORY COMPANY" 를 값으로 받았습니다.
///  기존 '스키마 에코' 필터는 스키마 필드명(reference_invoice 등)만 잡기 때문에
///  인쇄 라벨이 값 자리로 들어오는 이 경로를 막지 못합니다.
/// 🌟 [LABEL ECHO] 서식의 '박스 라벨' 목록.
///  실측 로그에서 reference_invoice 가 "CONSIGNEE VAT/EORI" 를,
///  party_name 이 "SIGNATORY COMPANY" 를 값으로 받았습니다.
///  기존 '스키마 에코' 필터는 스키마 필드명(reference_invoice 등)만 잡기 때문에
///  인쇄 라벨이 값 자리로 들어오는 이 경로를 막지 못합니다.
///
///  🌟 [다국어 확장 근거]
///   무역 문서는 발행국 언어로 라벨이 인쇄됩니다.
///   중국어 문서의 "发票号码", 일본어 문서의 "荷送人",
///   독일어 문서의 "Rechnungsnummer" 등이 값으로 잘못 추출되는 것을
///   이 목록이 걸러냅니다.
///   is_printed_label_echo() 는 fold_column_label() 로 정규화 후
///   완전일치 판정을 하므로, 라벨 원문을 그대로 나열하면 됩니다.
pub static TRADE_PRINTED_LABELS: &[&str] = &[
    // ── English ──
    "invoice number", "invoice no", "invoice total", "airwaybill bill of lading",
    "date of exportation", "export reference", "exporter", "consignee",
    "exporter vat eori", "consignee vat eori", "vat eori",
    "country of export", "buyer if not consignee", "reason for export",
    "country of ultimate destination", "total number of packages", "total weight",
    "incoterm", "incoterms", "currency", "signature of exporter",
    "signatory name", "signatory company", "date", "shipper", "notify party",
    "description of goods", "hs code", "country of manufacture", "unit of measure",
    "qty", "unit weight", "unit value", "total value",
    "bill of lading number", "b/l number", "awb number", "air waybill number",
    "packing list number", "purchase order number", "letter of credit number",
    "booking number", "contract number", "customs declaration number",
    "certificate number", "seal number", "container number",
    "port of loading", "port of discharge", "place of receipt", "place of delivery",
    "vessel name", "flight number", "voyage number",
    "estimated time of departure", "estimated time of arrival",
    "freight prepaid", "freight collect", "payment terms",
    "net weight", "gross weight", "measurement", "volume",
    "marks and numbers", "shipping marks",
    "total amount", "grand total", "subtotal",
    "unit price", "total price", "amount",
    // ── Korean ──
    "품명", "수량", "단가", "금액", "원산지", "세번부호",
    "송장번호", "인보이스번호", "선하증권번호", "항공운송장번호",
    "포장명세서번호", "구매주문번호", "신용장번호", "부킹번호",
    "계약번호", "수출신고번호", "수입신고번호", "증명서번호",
    "컨테이너번호", "씰번호", "선적항", "양륙항",
    "선박명", "항차", "출항일", "입항일",
    "총중량", "순중량", "용적", "포장수",
    "발행일", "만료일", "결제조건", "통화",
    "합계", "소계", "총액",
    // ── Chinese (Simplified) ──
    "发票号码", "发票号", "发票总额", "提单号码", "空运单号",
    "装箱单号", "采购订单号", "信用证号", "订舱号", "合同号",
    "报关单号", "证书号", "集装箱号", "铅封号",
    "装货港", "卸货港", "收货地", "交货地",
    "船名", "航次", "航班号",
    "预计开航日", "预计到港日",
    "运费预付", "运费到付", "付款条件",
    "毛重", "净重", "体积", "包装数量",
    "唛头", "运输标志",
    "总金额", "合计", "小计",
    "单价", "总价", "金额",
    "品名", "数量", "原产地", "海关编码",
    "出口日期", "出口参考号", "出口商", "收货人",
    "通知方", "发货人", "签字", "日期",
    "贸易术语", "币种", "计量单位",
    // ── Chinese (Traditional) ──
    "發票號碼", "提單號碼", "裝箱單號", "採購訂單號",
    "信用狀號", "訂艙號", "合約號",
    "裝貨港", "卸貨港", "收貨地", "交貨地",
    "船名", "航次",
    "毛重", "淨重", "體積", "包裝數量",
    "總金額", "合計", "單價", "總價",
    "品名", "數量", "原產地",
    // ── Japanese ──
    "インボイス番号", "請求書番号", "船荷証券番号", "航空運送状番号",
    "パッキングリスト番号", "発注番号", "信用状番号", "ブッキング番号",
    "契約番号", "通関申告番号", "証明書番号",
    "コンテナ番号", "シール番号",
    "積込港", "揚地港", "受取地", "引渡地",
    "船名", "航海番号", "便名",
    "総重量", "正味重量", "容積", "梱包数",
    "発行日", "有効期限", "支払条件", "通貨",
    "合計金額", "小計", "単価", "総額",
    "品名", "数量", "原産国", "税番",
    "荷送人", "荷受人", "通知先", "署名", "日付",
    "貿易条件", "計量単位",
    // ── German ──
    "rechnungsnummer", "rechnungsnummer", "rechnungsbetrag",
    "frachtbriefnummer", "luftfrachtbriefnummer",
    "packlistennummer", "bestellnummer", "akkreditivnummer",
    "buchungsnummer", "vertragsnummer", "zollanmeldungsnummer",
    "zertifikatsnummer", "containernummer", "plombennummer",
    "ladehafen", "löschhafen", "übernahmeort", "lieferungsort",
    "schiffsname", "reisenummer", "flugnummer",
    "bruttogewicht", "nettogewicht", "volumen", "anzahl packstücke",
    "ausstellungsdatum", "gültigkeitsdatum", "zahlungsbedingungen", "währung",
    "gesamtbetrag", "zwischensumme", "einzelpreis", "gesamtpreis",
    "warenbezeichnung", "menge", "ursprungsland", "zolltarifnummer",
    "exporteur", "empfänger", "benachrichtigungspartei", "unterschrift", "datum",
    "handelsklausel", "mengeneinheit",
    // ── Spanish ──
    "número de factura", "importe total de factura",
    "número de conocimiento de embarque", "número de guía aérea",
    "número de lista de empaque", "número de orden de compra",
    "número de carta de crédito", "número de reserva",
    "número de contrato", "número de declaración aduanera",
    "número de certificado", "número de contenedor", "número de precinto",
    "puerto de carga", "puerto de descarga", "lugar de recepción", "lugar de entrega",
    "nombre del buque", "número de viaje", "número de vuelo",
    "peso bruto", "peso neto", "volumen", "número de bultos",
    "fecha de emisión", "fecha de vencimiento", "condiciones de pago", "moneda",
    "importe total", "subtotal", "precio unitario", "precio total",
    "descripción de mercancías", "cantidad", "país de origen", "código arancelario",
    "exportador", "consignatario", "parte a notificar", "firma", "fecha",
    "términos comerciales", "unidad de medida",
    // ── French ──
    "numéro de facture", "montant total de facture",
    "numéro de connaissement", "numéro de lettre de transport aérien",
    "numéro de liste de colisage", "numéro de bon de commande",
    "numéro de crédit documentaire", "numéro de réservation",
    "numéro de contrat", "numéro de déclaration en douane",
    "numéro de certificat", "numéro de conteneur", "numéro de scellé",
    "port de chargement", "port de déchargement", "lieu de réception", "lieu de livraison",
    "nom du navire", "numéro de voyage", "numéro de vol",
    "poids brut", "poids net", "volume", "nombre de colis",
    "date d'émission", "date d'expiration", "conditions de paiement", "devise",
    "montant total", "sous-total", "prix unitaire", "prix total",
    "désignation des marchandises", "quantité", "pays d'origine", "code tarifaire",
    "exportateur", "destinataire", "partie à notifier", "signature", "date",
    "termes commerciaux", "unité de mesure",
    // ── Portuguese ──
    "número da fatura", "valor total da fatura",
    "número do conhecimento de embarque", "número do conhecimento aéreo",
    "número da lista de embalagem", "número do pedido de compra",
    "número da carta de crédito", "número da reserva",
    "número do contrato", "número da declaração aduaneira",
    "número do certificado", "número do contêiner", "número do lacre",
    "porto de carga", "porto de descarga", "local de recebimento", "local de entrega",
    "nome do navio", "número da viagem", "número do voo",
    "peso bruto", "peso líquido", "volume", "número de volumes",
    "data de emissão", "data de validade", "condições de pagamento", "moeda",
    "valor total", "subtotal", "preço unitário", "preço total",
    "descrição das mercadorias", "quantidade", "país de origem", "código tarifário",
    "exportador", "consignatário", "parte a notificar", "assinatura", "data",
    "termos comerciais", "unidade de medida",
    // ── Italian ──
    "numero di fattura", "importo totale fattura",
    "numero di polizza di carico", "numero di lettera di vettura aerea",
    "numero di lista di imballaggio", "numero di ordine di acquisto",
    "numero di lettera di credito", "numero di prenotazione",
    "numero di contratto", "numero di dichiarazione doganale",
    "numero di certificato", "numero di container", "numero di sigillo",
    "porto di carico", "porto di scarico", "luogo di ricevimento", "luogo di consegna",
    "nome della nave", "numero di viaggio", "numero di volo",
    "peso lordo", "peso netto", "volume", "numero di colli",
    "data di emissione", "data di scadenza", "condizioni di pagamento", "valuta",
    "importo totale", "subtotale", "prezzo unitario", "prezzo totale",
    "descrizione delle merci", "quantità", "paese di origine", "codice tariffario",
    "esportatore", "destinatario", "parte da notificare", "firma", "data",
    "termini commerciali", "unità di misura",
    // ── Dutch ──
    "factuurnummer", "factuurbedrag",
    "vrachtbriefnummer", "luchtvrachtbriefnummer",
    "paklijstnummer", "bestelnummer", "documentair kredietnummer",
    "boekingsnummer", "contractnummer", "douaneaangiftenummer",
    "certificaatnummer", "containernummer", "zegelnummer",
    "laadhaven", "loshaven", "plaats van ontvangst", "plaats van levering",
    "scheepsnaam", "reisnummer", "vluchtnummer",
    "brutogewicht", "nettogewicht", "volume", "aantal pakketten",
    "datum van afgifte", "vervaldatum", "betalingsvoorwaarden", "valuta",
    "totaalbedrag", "subtotaal", "eenheidsprijs", "totaalprijs",
    "omschrijving van goederen", "hoeveelheid", "land van oorsprong", "tariefnummer",
    "exporteur", "geadresseerde", "te notificeren partij", "handtekening", "datum",
    "handelsvoorwaarden", "maateenheid",
    // ── Arabic ──
    "رقم الفاتورة", "إجمالي الفاتورة",
    "رقم بوليصة الشحن", "رقم بوليصة الشحن الجوي",
    "رقم قائمة التعبئة", "رقم أمر الشراء", "رقم الاعتماد المستندي",
    "رقم الحجز", "رقم العقد", "رقم البيان الجمركي",
    "رقم الشهادة", "رقم الحاوية", "رقم الختم",
    "ميناء الشحن", "ميناء التفريغ", "مكان الاستلام", "مكان التسليم",
    "اسم السفينة", "رقم الرحلة البحرية", "رقم الرحلة الجوية",
    "الوزن الإجمالي", "الوزن الصافي", "الحجم", "عدد الطرود",
    "تاريخ الإصدار", "تاريخ الانتهاء", "شروط الدفع", "العملة",
    "المبلغ الإجمالي", "المجموع الفرعي", "سعر الوحدة", "السعر الإجمالي",
    "وصف البضائع", "الكمية", "بلد المنشأ", "الرمز الجمركي",
    "المصدّر", "المرسل إليه", "الطرف المطلوب إخطاره", "التوقيع", "التاريخ",
    "الشروط التجارية", "وحدة القياس",
    // ── Czech ──
    "číslo faktury", "celková částka faktury",
    "číslo konosamentu", "číslo leteckého nákladního listu",
    "číslo balicího listu", "číslo nákupní objednávky", "číslo akreditivu",
    "číslo rezervace", "číslo smlouvy", "číslo celního prohlášení",
    "číslo certifikátu", "číslo kontejneru", "číslo plomby",
    "přístav nakládky", "přístav vykládky", "místo převzetí", "místo dodání",
    "název plavidla", "číslo plavby", "číslo letu",
    "hrubá hmotnost", "čistá hmotnost", "objem", "počet balení",
    "datum vystavení", "datum platnosti", "platební podmínky", "měna",
    "celková částka", "mezisoučet", "jednotková cena", "celková cena",
    "popis zboží", "množství", "země původu", "celní kód",
    "vývozce", "příjemce", "strana k oznámení", "podpis", "datum",
    "obchodní podmínky", "měrná jednotka",
];

/// 🌟 [LABEL SCRIPT DETECT] 별칭/라벨 문자열의 유니코드 스크립트를 판정합니다.
///  한글 > 가나 > 아랍 > 한자 > 라틴 순서로 우선 검사하여
///  한자+가나 혼재 시 일본어로 판정합니다.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LabelScript {
    Latin,
    Korean,
    Japanese,
    Chinese,
    Arabic,
}

fn detect_label_script(text: &str) -> LabelScript {
    let mut korean = 0usize;
    let mut kana   = 0usize;
    let mut cjk    = 0usize;
    let mut arabic = 0usize;
    let mut latin  = 0usize;
    for c in text.chars() {
        if c.is_ascii_alphabetic()                          { latin  += 1; }
        else if ('\u{AC00}'..='\u{D7AF}').contains(&c)     { korean += 1; }
        else if ('\u{3040}'..='\u{30FF}').contains(&c)     { kana   += 1; }
        else if ('\u{4E00}'..='\u{9FFF}').contains(&c)     { cjk    += 1; }
        else if ('\u{0600}'..='\u{06FF}').contains(&c)     { arabic += 1; }
    }
    if korean > 0 { return LabelScript::Korean; }
    if kana   > 0 { return LabelScript::Japanese; }
    if arabic > 0 { return LabelScript::Arabic; }
    if cjk    > 0 { return LabelScript::Chinese; }
    if latin  > 0 { return LabelScript::Latin; }
    LabelScript::Latin
}

/// 🌟 [ALIAS LANG MATCH] 별칭의 스크립트가 문서 언어와 매칭되는지 판정합니다.
///  라틴(영어/독일어/스페인어/프랑스어 등)은 국제 표준이므로 항상 통과시킵니다.
///  doc_lang 이 빈 문자열이면(미확정) 전체 매칭으로 폴백합니다.
fn alias_lang_matches(alias: &str, doc_lang: &str) -> bool {
    if doc_lang.is_empty() { return true; }
    let script = detect_label_script(alias);
    match script {
        LabelScript::Latin   => true,
        LabelScript::Korean  => doc_lang == "ko",
        LabelScript::Japanese=> doc_lang == "ja",
        LabelScript::Chinese => doc_lang.starts_with("zh"),
        LabelScript::Arabic  => doc_lang == "ar",
    }
}

/// 라벨 문자열을 비교 가능한 형태로 접습니다. 영숫자 외는 공백으로 바꾸고 소문자화합니다.
fn fold_column_label(raw: &str) -> String {
    raw.chars()
        .map(|c| if c.is_alphanumeric() { c.to_lowercase().next().unwrap_or(c) } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// 인쇄된 컬럼명을 스키마 필드명으로 확정합니다. 매칭 실패 시 빈 문자열을 돌려줍니다.
/// 🌟 [LANG FILTER] doc_lang 이 확정되어 있으면 해당 언어의 별칭만 대조합니다.
///    영어(라틴 스크립트)는 국제 표준이므로 항상 포함됩니다.
///    매칭 대상이 약 480개 → 약 30~50개로 축소되어 속도가 오히려 빨라집니다.
pub fn canonicalize_trade_column(raw_header: &str, doc_lang: &str) -> String {
    let norm = fold_column_label(raw_header);
    if norm.is_empty() {
        return String::new();
    }

    // 1. 완전 일치 우선 (언어 필터링 적용)
    for (field, aliases) in TRADE_COLUMN_ALIASES.iter() {
        for a in aliases.iter() {
            if !alias_lang_matches(a, doc_lang) {
                continue;
            }
            if fold_column_label(a) == norm {
                return field.to_string();
            }
        }
    }

    // 2. 부분 일치 — 가장 긴 별칭이 이깁니다.
    //    'unit weight'(11자) 가 'unit'(4자) 보다 먼저 잡혀야
    //    item_net_weight 로 가고 unit 으로 오배정되지 않습니다.
    let mut best_len = 0usize;
    let mut best_field = "";
    for (field, aliases) in TRADE_COLUMN_ALIASES.iter() {
        for a in aliases.iter() {
            if !alias_lang_matches(a, doc_lang) {
                continue;
            }
            let a_norm = fold_column_label(a);
            if a_norm.is_empty() {
                continue;
            }
            if norm.contains(&a_norm) && a_norm.len() > best_len {
                best_len = a_norm.len();
                best_field = field;
            }
        }
    }

    best_field.to_string()
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

pub fn is_printed_label_echo(value: &str, doc_lang: &str) -> bool {
    let norm = fold_column_label(value);
    if norm.is_empty() {
        return false;
    }
    if TRADE_PRINTED_LABELS.iter().any(|l| {
        alias_lang_matches(l, doc_lang) && fold_column_label(l) == norm
    }) {
        return true;
    }
    TRADE_COLUMN_ALIASES.iter().any(|(_, aliases)| {
        aliases.iter().any(|a| {
            alias_lang_matches(a, doc_lang) && fold_column_label(a) == norm
        })
    })
}

/// 🌟 [ROW CONTRACT] 표 타일 프롬프트에 실을 계약 문자열을 만듭니다.
///  실측 로그에서 타일 2 는 T-Shirt 행과 Shorts 행을 둘 다 담고 있었는데
///  응답이 객체 1개였고, 파이프라인이 그것을 배열 1원소로 승격했습니다.
///  ("ARRAY COERCE 단일 객체 응답을 원소 1개 배열로 승격합니다")
///  그 결과 Shorts 행이 영구 소멸했습니다. 행 수 = 타일 수가 되어 버립니다.
///  타일 스키마를 배열로 고정하고, 인쇄 라벨과 필드명의 대응을 명시해
///  '보이는 데이터 행 수만큼' 원소를 만들도록 강제합니다.
pub fn build_table_row_contract(headers: &[Vec<String>], doc_lang: &str) -> String {
    if headers.is_empty() {
        return String::new();
    }
    let width = headers.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut mapping = Vec::new();
    let mut fields = Vec::new();
    for col in 0..width {
        let mut parts: Vec<&str> = Vec::new();
        for h_row in headers.iter() {
            if let Some(seg) = h_row.get(col) {
                let seg = seg.trim();
                if !seg.is_empty() && !parts.contains(&seg) {
                    parts.push(seg);
                }
            }
        }
        let printed = parts.join(" ");
        if printed.is_empty() {
            continue;
        }
        let field = canonicalize_trade_column(&printed, doc_lang);
        if field.is_empty() {
            continue;
        }
        mapping.push(format!("   column {} \"{}\" -> \"{}\"", col, printed, field));
        if !fields.contains(&field) {
            fields.push(field);
        }
    }
    if mapping.is_empty() {
        return String::new();
    }
    let obj = fields
        .iter()
        .map(|f| format!("\"{}\": null", f))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "COLUMN MAP (printed header -> output field):
{}
\
RULES:
\
- Return a JSON ARRAY. One object per printed data row. Never collapse rows.
\
- If the image shows 2 data rows, the array MUST have 2 elements.
\
- Do not output header rows, subtotal rows, or total rows as elements.
\
- Use null for a column that is not printed on that row.
\
SCHEMA:
[ {{ {} }} ]",
        mapping.join("\n"),
        obj
    )
}

/// 🌟 [HEADER BAND v2 — SYNC] DOM 순회를 동기로 수행하여 소유 데이터를 반환합니다.
///    `&Html` / `ElementRef` 가 비동기 경계를 넘지 않도록 이 함수는 순수 동기입니다.
///    반환값: (직사각 헤더 격자, 임베딩 폴백 대기 목록)
pub fn extract_doc_table_headers_sync(
    document: &Html,
    table_selector: &str,
    doc_lang: &str,
) -> (Vec<Vec<String>>, Vec<(usize, String)>) {
    let empty: Vec<Vec<String>> = Vec::new();
    let sel = match Selector::parse(table_selector) { Ok(s) => s, Err(_) => return (empty, Vec::new()) };
    let first_match = match document.select(&sel).next() { Some(m) => m, None => return (empty, Vec::new()) };

    // 1. 선택자에서 위로 올라가 감싸는 <table> 을 찾습니다.
    let mut table_ref = None;
    let mut current = first_match.parent();
    while let Some(parent) = current {
        if let Some(el) = parent.value().as_element() {
            if el.name() == "table" {
                table_ref = scraper::ElementRef::wrap(parent);
                break;
            }
        }
        current = parent.parent();
    }
    let table_ref = match table_ref { Some(t) => t, None => return (empty, Vec::new()) };

    let tr_sel = match Selector::parse("tr") { Ok(s) => s, Err(_) => return (empty, Vec::new()) };
    let cell_sel = match Selector::parse("th, td") { Ok(s) => s, Err(_) => return (empty, Vec::new()) };

    // 2. 헤더 행 후보 수집
    let mut header_rows: Vec<scraper::ElementRef> = Vec::new();
    if let Ok(thead_sel) = Selector::parse("thead") {
        if let Some(thead) = table_ref.select(&thead_sel).next() {
            header_rows.extend(thead.select(&tr_sel));
        }
    }
    if header_rows.is_empty() {
        for tr in table_ref.select(&tr_sel) {
            let cells: Vec<_> = tr.select(&cell_sel).collect();
            if cells.is_empty() { continue; }
            let th_count = cells.iter()
                .filter(|c| c.value().name().eq_ignore_ascii_case("th"))
                .count();
            let scope_col = cells.iter()
                .any(|c| c.value().attr("scope").map_or(false, |s| s.eq_ignore_ascii_case("col")));
            if th_count * 2 >= cells.len() || scope_col {
                header_rows.push(tr);
            } else {
                break;
            }
        }
    }
    if header_rows.is_empty() { return (empty, Vec::new()); }

    let band = header_rows.len();

    // 3. colspan / rowspan 을 펼쳐 직사각 격자로 만듭니다.
    let mut grid: Vec<Vec<String>> = vec![Vec::new(); band];
    let mut all_pending: Vec<(usize, String)> = Vec::new();
    let mut global_cell_idx: usize = 0;

    for (r, tr) in header_rows.iter().enumerate() {
        let mut c = 0usize;

        for cell in tr.select(&cell_sel) {
            while grid[r].len() > c && !grid[r][c].is_empty() { c += 1; }

            let colspan = cell
                .value()
                .attr("colspan")
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(1)
                .clamp(1, 64);
            let rowspan = cell
                .value()
                .attr("rowspan")
                .and_then(|v| v.trim().parse::<usize>().ok())
                .unwrap_or(1)
                .clamp(1, 64)
                .min(band - r);

            let raw = cell
                .value()
                .attr("abbr")
                .map(|s| s.to_string())
                .unwrap_or_else(|| cell.text().collect::<Vec<_>>().join(" "));
            let title = raw.split_whitespace().collect::<Vec<_>>().join(" ");

            let canonical = canonicalize_trade_column(&title, doc_lang);
            if canonical.is_empty() && !title.is_empty() {
                all_pending.push((global_cell_idx, title.clone()));
            }
            global_cell_idx += 1;

            for dr in 0..rowspan {
                let rr = r + dr;
                for dc in 0..colspan {
                    let cc = c + dc;
                    if grid[rr].len() <= cc {
                        grid[rr].resize(cc + 1, String::new());
                    }
                    grid[rr][cc] = title.clone();
                }
            }
            c += colspan;
        }
    }

    let width = grid.iter().map(|r| r.len()).max().unwrap_or(0);
    if width == 0 { return (empty, Vec::new()); }
    for row in grid.iter_mut() { row.resize(width, String::new()); }
    (grid, all_pending)
}

/// 🌟 [HEADER BAND v2 — ASYNC] 임베딩 폴백만 수행합니다.
///    `&Html` / `ElementRef` 가 이 함수에 전혀 등장하지 않으므로
///    `tokio::spawn` 내부에서 안전하게 `.await` 할 수 있습니다.
pub async fn extract_doc_table_headers_async(
    mut grid: Vec<Vec<String>>,
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

pub fn split_html_to_pug_list(html: &str, selector_str: &str, mode: PugMode) -> Vec<String> {
    let document = Html::parse_document(html);
    split_doc_to_pug_list(&document, selector_str, mode)
}

// =====================================================================
// 🌟 [DEPRECATED] 무역 서식별 고정 크롭 좌표표
// ---------------------------------------------------------------------
//  ── 왜 폐기하는가 ──
//   이 표는 '문서 세로 비율' 을 서식마다 손으로 적어 둔 것입니다.
//   실제 문서 레이아웃과 어긋나는 경우가 구조적으로 발생합니다.
//
//   ① 가로 2단 배치 붕괴
//      B/L 은 좌측에 Shipper, 우측에 Consignee 를 나란히 인쇄합니다.
//      ("parties", 0.00, 0.60) 은 세로 60% 를 통째로 자르므로
//      두 당사자 + 문서번호 + 선박명이 한 조각에 뭉개져 들어가고,
//      LLM 은 어느 값이 어느 필드인지 구분할 근거를 잃습니다.
//
//   ② 표 위치 불일치
//      ("items", 0.30, 0.70) 은 품목표가 중단에 있다고 가정합니다.
//      표가 하단에 몰린 서식(대부분의 Packing List)에서는
//      이 슬라이스가 빈 여백만 잡아 items 가 항상 빈 배열이 됩니다.
//
//   ③ 카테고리 영역 중복
//      ("header", 0.00, 0.25) 와 ("parties", 0.00, 0.40) 은 0.00~0.25 가 겹칩니다.
//      같은 픽셀을 두 번 크롭해 LLM 을 두 번 호출하고,
//      두 호출이 같은 값을 서로 다른 필드로 뱉으면 조건이 오염됩니다.
//
//   ④ 서식 확장 비용
//      45종 데이터셋 중 이 표가 다루는 것은 27종뿐이며,
//      HBL / SWB / FCR / POD / SOA / TI 등은 폴백 4슬라이스로 떨어져
//      cargo / financials / logistics 가 아예 추출되지 않았습니다.
//
//  ── 무엇으로 대체되었는가 ──
//   models/siglip2/vision_encoder.rs :: build_column_heatmaps
//     bias_schema 의 필드 semantic/bias 구를 SigLIP2 텍스트 공간에 올리고
//     이미지 패치와 코사인을 재서 '실제로 인쇄된 위치' 를 찾습니다.
//   models/siglip2/vision_crop.rs :: plan_crops
//     히트맵 → 연결 성분 → 인접 병합 → IoU dedup → 배타 배정 → 픽셀 박스.
//     한 카테고리는 한 영역만, 한 영역은 한 카테고리만 가져갑니다.
//
//   좌표를 코드에 적지 않으므로 서식이 늘어도 이 파일은 수정 대상이 아닙니다.
//   새 필드가 필요하면 bias.json 의 trade_schema 에만 추가하면 됩니다.
//
//  ── 왜 삭제하지 않고 남기는가 ──
//   호출부가 남아 있으면 컴파일 경고로 즉시 드러나야 하고,
//   나중에 누군가 '고정 좌표가 필요하다' 며 다시 작성하는 것을 막기 위해
//   폐기 사유를 코드에 남깁니다.
// =====================================================================
#[deprecated(
    since = "vision-nms",
    note = "고정 비율 크롭은 폐기되었습니다. \
            siglip2::vision_encoder::build_column_heatmaps + \
            siglip2::vision_crop::plan_crops 를 사용하십시오."
)]
#[allow(dead_code)]
pub fn get_trade_doc_slice_config(_doc_type: &str) -> Vec<(&'static str, f32, f32)> {
    // 🌟 어떤 좌표도 돌려주지 않습니다.
    //    실수로 호출되더라도 잘못된 영역을 크롭하는 대신
    //    호출부가 '크롭 계획 없음' 을 인지하고 전체 페이지 폴백으로 가도록 만듭니다.
    Vec::new()
}

/// 🌟 [VISION CROP CATEGORIES] 비전 크롭 + LLM 추출이 순회하는 카테고리 목록.
///
///  ── get_trade_doc_slice_config 의 유일한 유효 계승분 ──
///   기존 함수가 실제로 제공하던 정보 중 좌표를 뺀 나머지,
///   즉 '이 서식에서 어떤 카테고리를 뽑아야 하는가' 만 남깁니다.
///   좌표는 히트맵이 결정하고, 카테고리 집합은 스키마가 결정합니다.
///
///  ── 왜 스키마에서 유도하는가 ──
///   get_trade_category_schema 가 특정 카테고리에 대해 빈 스키마를 돌려주면
///   그 카테고리는 이 서식에 존재하지 않는 것입니다.
///   그 사실을 좌표표에 다시 적을 이유가 없습니다.
pub fn get_trade_doc_categories(doc_type: &str) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for cat in crate::logic::TRADE_EXTRACTION_CATEGORIES.iter() {
        let schema = get_trade_category_schema(cat, doc_type);
        // get_trade_category_schema 는 필드가 없으면 "SCHEMA:\n{}" 또는
        // "SCHEMA:\n[ {} ]" 를 돌려줍니다. 그 서식에 없는 카테고리입니다.
        if schema.contains("SCHEMA:\n{}") || schema.contains("SCHEMA:\n[ {} ]") {
            continue;
        }
        out.push(cat);
    }

    // 🌟 [RELAY FLOOR] 릴레이 키를 담는 카테고리는 스키마 판정과 무관하게 반드시 남깁니다.
    //    header      : doc_number / reference_bl / reference_po
    //    logistics   : awb_number / flight_number / vessel
    //    containers  : container_number
    //    이 카테고리가 순회 대상에서 빠지면 크롭 계획이 만들어지지 않고,
    //    "TRADE RELAY STARVED — 빈 키" 가 스키마 단계에서 이미 확정됩니다.
    for must in ["header", "logistics", "containers"] {
        if crate::logic::TRADE_EXTRACTION_CATEGORIES.iter().any(|c| *c == must)
            && !out.contains(&must)
        {
            out.push(must);
        }
    }

    if out.is_empty() {
        out.push("header");
    }
    out
}

/// 릴레이 키의 역할. 같은 역할끼리만 서로 연결됩니다.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TradeRelayKey {
    pub role: &'static str,   // "self" | "transport" | "booking" | "order" | "contract" | "credit" | "container"
    pub source_field: String, // 내 문서에서 값을 가져온 필드 (진단용)
    pub search_field: String,
    pub raw: String,          // 인쇄된 원문
    pub normalized: String,   // normalize_identifier 통과값
    pub index: u32,           // crc32(normalized) — commerce 의 tracking index 와 같은 축
    pub id: String,           // hash_id(normalized)
}

/// 역할별로 훑을 필드 이름을 우선순위 순으로 선언합니다.
/// 앞쪽 필드가 먼저 채택되고, 같은 역할 안에서 중복 index 는 제거됩니다.
pub static TRADE_RELAY_FIELDS: &[(&str, &[&str])] = &[
    // 이 문서 자신을 가리키는 번호. CI 의 INVOICE NUMBER 가 여기 옵니다.
    ("self",      &["doc_number", "reference_number", "reference_invoice"]),
    // 운송증권 번호. CI 에도 AWB / B/L 번호가 인쇄되므로 CI↔BL / CI↔AWB 가 여기서 성립합니다.
    ("transport", &["reference_bl", "reference_master_bl", "bl_number",
                    "awb_number", "airway_bill_number", "tracking_number"]),
    ("booking",   &["reference_booking", "booking_number"]),
    ("order",     &["reference_po", "po_number", "order_number"]),
    ("contract",  &["reference_contract", "reference_sr", "contract_number"]),
    ("credit",    &["reference_lc", "lc_number"]),
    ("container", &["container_number"]),
];

/// JSON 트리에서 문자열/숫자 값을 얕게 찾아 옵니다.
/// 루트 → 카테고리 객체(header/logistics/containers …) → 배열 원소 순으로 훑습니다.
fn find_relay_value(data: &Value, key: &str) -> Option<String> {
    fn as_text(v: &Value) -> Option<String> {
        match v {
            Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
            Value::Number(n) => Some(n.to_string()),
            _ => None,
        }
    }
    if let Some(v) = data.get(key) {
        if let Some(t) = as_text(v) { return Some(t); }
    }
    if let Some(map) = data.as_object() {
        for (_, sub) in map.iter() {
            match sub {
                Value::Object(_) => {
                    if let Some(v) = sub.get(key) {
                        if let Some(t) = as_text(v) { return Some(t); }
                    }
                }
                Value::Array(items) => {
                    for it in items.iter() {
                        if let Some(v) = it.get(key) {
                            if let Some(t) = as_text(v) { return Some(t); }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    None
}

pub fn extract_trade_relay_keys(data: &Value, doc_lang: &str) -> Vec<TradeRelayKey> {
    // 🌟 [BACK-COMPAT] doc_type 을 모르는 레거시 호출부용. 기존 동작을 그대로 유지합니다.
    extract_trade_relay_keys_for(data, doc_lang, "")
}
/// 🌟 [REVERSE ROLE v4] 자기 문서번호를 '남이 나를 부르는 이름' 으로 등록합니다.
///
///  ── 무엇이 문제였나 ──
///   기존 v3 는 문서 종류와 무관하게 role/search_field 를 "reference_invoice" 로
///   고정했습니다. PO 문서가 저장되면
///     "reference_invoice == PO-99281A 인 문서를 찾아라"
///   라는 성립 불가능한 릴레이가 나갑니다. 정답은 reference_po 입니다.
///   실측 데이터셋 45종 중 CI 를 제외한 44종이 전부 이 경로로 빗나갑니다.
///
///  ── 수정 근거 ──
///   '이 서식을 남이 어떤 참조 축으로 부르는가' 는 logic.rs 의
///   trade_reference_field_of 가 이미 소유한 사실입니다. 코드에 사전을
///   다시 적지 않고 그 함수 하나에 위임합니다.
pub fn extract_trade_relay_keys_for(data: &Value, doc_lang: &str, doc_type: &str) -> Vec<TradeRelayKey> {
    let mut out: Vec<TradeRelayKey> = Vec::new();
    let reverse_field: &'static str =
        crate::logic::trade_reference_field_of(doc_type).unwrap_or("reference_invoice");
    if let Some(doc_num_raw) = data.get("doc_number")
        .or_else(|| data.get("document_number"))
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty() && s.as_str() != "N/A")
    {
        if !is_printed_label_echo(&doc_num_raw, doc_lang) && crate::utils::hash::is_valid_relay_key(&doc_num_raw) {
            let normalized = crate::utils::hash::normalize_identifier(&doc_num_raw);
            let index = crate::utils::hash::relay_index(&normalized);
            if index > 0 {
                println!(
                    "[RELAY] 🔁 [REVERSE ROLE] doc_type='{}' 의 역방향 참조 축을 '{}' 로 확정했습니다. (값 '{}')",
                    doc_type, reverse_field, doc_num_raw
                );
                out.push(TradeRelayKey {
                    role: reverse_field,
                    source_field: "doc_number".to_string(),
                    search_field: reverse_field.to_string(),
                    raw: doc_num_raw.clone(),
                    normalized: normalized.clone(),
                    index,
                    id: crate::utils::hash::relay_id(&normalized, reverse_field),
                });
            }
        }
    }
    // 🌟 [v3] reference_po도 역할로 추가합니다.
    for ref_field in ["reference_po", "reference_lc", "reference_bl", "reference_booking", "reference_contract"] {
        let raw = match find_relay_value(data, ref_field) { Some(r) => r, None => continue };
        if is_printed_label_echo(&raw, doc_lang) { continue; }
        if !crate::utils::hash::is_valid_relay_key(&raw) { continue; }
        let normalized = crate::utils::hash::normalize_identifier(&raw);
        let index = crate::utils::hash::relay_index(&normalized);
        if index == 0 { continue; }
        // role은 참조 필드명에서 "reference_" 접두를 제거한 것으로 매핑하지 않고,
        // 참조 필드명을 그대로 역할로 사용합니다.
        // 이렇게 하면 plan_trade_relays에서 역할별 타겟을 정확히 매핑할 수 있습니다.
        let role = match ref_field {
            "reference_po" => "order",
            "reference_lc" => "credit",
            "reference_bl" => "transport",
            "reference_booking" => "booking",
            "reference_contract" => "contract",
            _ => continue,
        };
        if out.iter().any(|k| k.role == role && k.index == index) { continue; }
        // 🌟 [SEARCH FIELD FIX] 역할별 검색 필드 결정:
        //    - "내가 참조하는 값"을 찾는 역할(transport/booking/order/contract/credit):
        //      상대 문서의 고유 식별 필드 "doc_number"에서 검색합니다.
        //    - "나를 참조하는 값"을 찾는 역할은 이 블록에 없습니다.
        let search_field = match role {
            "transport" | "booking" | "order" | "contract" | "credit" => "doc_number".to_string(),
            "container" => "container_number".to_string(),
            _ => "doc_number".to_string(),
        };
        out.push(TradeRelayKey {
            role,
            source_field: ref_field.to_string(),
            search_field,
            raw: raw.clone(),
            normalized: normalized.clone(),
            index,
            id: crate::utils::hash::relay_id(&normalized, role),
        });
    }
    // 🌟 [v3] 기존 TRADE_RELAY_FIELDS 순회도 유지하되,
    //    "self" 역할이 이미 제거되었으므로 중복 생성되지 않습니다.
    //    단, container_number 등 기존 역할은 그대로 유지합니다.
    for (role, fields) in TRADE_RELAY_FIELDS.iter() {
        // "self" 역할은 더 이상 사용하지 않습니다 (위에서 "reference_invoice"으로 대체)
        if *role == "self" { continue; }
        for f in fields.iter() {
            let raw = match find_relay_value(data, f) { Some(r) => r, None => continue };
            if is_printed_label_echo(&raw, doc_lang) { continue; }
            if !crate::utils::hash::is_valid_relay_key(&raw) { continue; }
            let normalized = crate::utils::hash::normalize_identifier(&raw);
            let index = crate::utils::hash::relay_index(&normalized);
            if index == 0 { continue; }
            if out.iter().any(|k| k.role == *role && k.index == index) { continue; }
            // 🌟 [SEARCH FIELD FIX] 역할별 검색 필드 결정:
            //    - "내가 참조하는 값"을 찾는 역할: 상대 문서의 고유 식별 필드에서 검색
            //    - "나를 참조하는 값"을 찾는 역할: 상대 문서의 참조 필드에서 검색
            let search_field = match *role {
                "transport" | "booking" | "order" | "contract" | "credit" => "doc_number".to_string(),
                "container" => "container_number".to_string(),
                "reference_invoice" => "reference_invoice".to_string(),
                "reference_po" => "reference_po".to_string(),
                "reference_lc" => "reference_lc".to_string(),
                _ => "doc_number".to_string(),
            };
            out.push(TradeRelayKey {
                role,
                source_field: f.to_string(),
                search_field,
                raw: raw.clone(),
                normalized: normalized.clone(),
                index,
                id: crate::utils::hash::relay_id(&normalized, role),
            });
        }
    }
    out
}

pub fn resolve_trade_doc_identity(doc_type: &str, data: &Value, doc_lang: &str) -> (String, u32, bool) {
    let keys = extract_trade_relay_keys(data, doc_lang);

    // 🌟 [DOC NUMBER DIRECT] extracted_data에서 doc_number를 직접 확인합니다.
    //    extract_trade_relay_keys의 "self" 역할 매핑이 유효성 검사에서
    //    떨어지는 경우(예: "26" 같은 부분 캡처)를 방지합니다.
    //    2자 이상이면 문서번호로 인정합니다.
    let direct_doc_number = data
        .get("doc_number")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| s.chars().count() >= 2 && s != "null" && s != "N/A");

    if let Some(dn) = &direct_doc_number {
        let idx = crate::utils::hash::relay_index(dn);
        if idx > 0 {
            return (dn.clone(), idx, false);
        }
    }

    // 1순위: 이 문서 자신의 번호 (기존 경로 유지)
    if let Some(k) = keys.iter().find(|k| k.role == "self") {
        let idx = crate::utils::hash::relay_index(&k.normalized);
        return (k.normalized.clone(), idx, false);
    }

    // 2순위: 운송증권 번호
    // 🌟 [TRANSPORT FALLBACK GUARD] transport 번호를 문서 식별자로 쓸 때
    //    반드시 "서식코드:번호" 형식으로 합성합니다.
    //    이렇게 해야 서로 다른 서식이 같은 운송번호를 공유해도
    //    각각 다른 식별자를 갖게 됩니다.
    if let Some(k) = keys.iter().find(|k| k.role == "transport") {
        let composed = format!("{}:{}", doc_type, k.normalized);
        let idx = crate::utils::hash::relay_index(&composed);
        return (composed, idx, false);
    }
    
    // 3순위: 발주 / 계약 / 신용장 번호
    for role in ["order", "contract", "credit", "booking", "container"] {
        if let Some(k) = keys.iter().find(|k| k.role == role) {
            let composed = format!("{}:{}", doc_type, k.normalized);
            let idx = crate::utils::hash::relay_index(&composed);
            return (composed, idx, false);
        }
    }
    
    // 4순위: 내용 지문.
    // 🌟 [FINGERPRINT v4] 기존은 `crc32` 만 사용했는데,
    //    내용 지문에도 전각 영숫자가 포함될 수 있으므로
    //    `normalize_identifier` 를 먼저 적용합니다.
    let mut parts: Vec<String> = vec![doc_type.to_string()];
    for f in ["issue_date", "grand_total_amount", "amount", "currency",
              "sender_name", "recipient_name", "weight_gross", "package_count"] {
        if let Some(v) = find_relay_value(data, f) { parts.push(format!("{}={}", f, v)); }
    }
    
    let fingerprint = parts.join("|");
    let normalized_fp = crate::utils::hash::normalize_identifier(&fingerprint);
    let index = crate::utils::hash::crc32(&normalized_fp);
    (fingerprint, index, true)
}
pub fn plan_trade_relays(doc_type: &str, data: &Value, doc_lang: &str) -> Vec<(&'static str, TradeRelayKey)> {
    let keys = extract_trade_relay_keys_for(data, doc_lang, doc_type);
    let reverse_field: &'static str =
        crate::logic::trade_reference_field_of(doc_type).unwrap_or("reference_invoice");
    let targets: Vec<&'static str> = crate::logic::related_trading(doc_type);
    let mut plan: Vec<(&'static str, TradeRelayKey)> = Vec::new();
    let mut forward_targets: Vec<&'static str> = Vec::new();

    let push = |plan: &mut Vec<(&'static str, TradeRelayKey)>,
                t: &'static str,
                k: &TradeRelayKey,
                search: &str| {
        if t == doc_type { return; }
        if plan.iter().any(|(tt, kk)| *tt == t && kk.index == k.index) { return; }
        let mut kk = k.clone();
        kk.search_field = search.to_string();
        plan.push((t, kk));
    };

    // ── ① forward : 내가 참조하는 번호로 '그 번호를 자기 문서번호로 갖는' 상대를 찾습니다 ──
    for k in keys.iter() {
        if k.role == reverse_field { continue; }
        // 🌟 [CONTAINER] 컨테이너 번호는 어느 서식의 doc_number 도 아닙니다.
        //    trading_relay_pair 는 reference_* 축만 다루므로 이 역할만 예외로 둡니다.
        if k.role == "container" {
            for t in ["PL", "BL", "DO", "CM"] {
                push(&mut plan, t, k, "container_number");
                if !forward_targets.contains(&t) { forward_targets.push(t); }
            }
            continue;
        }
        for t in targets.iter() {
            let mine = match crate::logic::trading_relay_pair(doc_type, *t) {
                Some((m, _)) => m,
                None => continue,
            };
            // 이 상대를 가리키는 내 필드가 바로 이 키의 출처일 때만 성립합니다.
            if mine != k.source_field { continue; }
            if !forward_targets.contains(t) { forward_targets.push(*t); }
            push(&mut plan, *t, k, "doc_number");
        }
    }

    // ── ② reverse : 내 문서번호를 참조하고 있을 상대를 찾습니다 ──
    for k in keys.iter() {
        if k.role != reverse_field { continue; }
        for t in targets.iter() {
            if forward_targets.contains(t) { continue; }
            let foreign = match crate::logic::trading_relay_pair(doc_type, *t) {
                Some((_, f)) => f,
                None => continue,
            };
            push(&mut plan, *t, k, foreign);
        }
    }
    plan
}

// =====================================================================
// 🌟 [TRADING SCOPE] 무역 트랙의 cc / ref 축
// ---------------------------------------------------------------------
//  ── 왜 별도 축이 필요한가 ──
//   commerce 의 cc 는 '어느 쇼핑몰에서 왔는가' 이고 ref 는 '어느 페이지인가' 입니다.
//   무역 서식(B/L PDF · 스캔 이미지)에는 출처 사이트도 페이지도 없습니다.
//   브라우저 URL 로 두 축을 만들면 '그때 열려 있던 탭' 이 스코프가 되어,
//   같은 선적 서류가 탭에 따라 다른 버킷으로 흩어집니다.
//
//  ── 대체 정의 ──
//   cc  = hash_id(TRADING_HOST)                  고정 합성 도메인
//   ref = hash_id(team + cc + '#' + hub_key)     '거래 건' 단위
//   commerce 의 ref 가 '페이지' 였던 자리에 '선적/거래 건' 이 들어갑니다.
//
//  ⚠️ trading Worker(src/index.ts)의 tradingRef / resolveHubKey 와
//     '문자 단위로 동일' 해야 합니다. 한쪽만 바꾸면 같은 문서에
//     서버 ref / 로컬 ref 두 벌이 생겨 스코프가 갈립니다.
// =====================================================================

/// 무역 트랙의 합성 도메인. commerce Worker 가 홈 화면에 'logis.center' 를
/// 쓰는 것과 같은 계보이며, 실제 사이트가 없는 트랙 자체를 하나의 도메인으로 봅니다.
pub const TRADING_HOST: &str = "trading.logis.center";

/// 🌟 무역 트랙의 cc. 요청/URL 로 바뀌지 않는 상수입니다.
pub fn trading_cc() -> String {
    crate::utils::hash::hash_id(TRADING_HOST)
}

/// 🌟 허브 우선순위. logic.rs 의 TRADE_HUB_TYPES 와 같은 순서입니다.
///    PO(거래 시작) → CI(물품 명세) → BL(운송) → LC(결제 보증)
const TRADE_HUB_FIELDS: &[(&str, &str)] = &[
    ("PO", "reference_po"),
    ("CI", "reference_invoice"),
    ("BL", "reference_bl"),
    ("LC", "reference_lc"),
];

/// 🌟 [HUB KEY] 이 문서가 속한 '거래 건' 의 씨앗을 고릅니다.
///
///  ── 순서와 근거 ──
///   ① 허브 참조(PO → CI → BL → LC)가 있으면 그것이 이 거래의 뿌리입니다.
///   ② 없고 자기 자신이 허브 타입이면 자기 doc_number 가 뿌리입니다.
///      (PO 문서에는 reference_po 가 없습니다. 자기가 시작점이기 때문입니다)
///   ③ 그래도 없으면 TRADE_RELAY_FIELDS 의 아무 참조라도 잡아 최소 묶음을 만듭니다.
///   ④ 전부 실패하면 자기 doc_number, 그것도 없으면 id 로 고립시킵니다.
///
///  ④ 로 떨어져도 손해가 없습니다. ref 가 자기 자신만 담는 1건짜리 거래가 될 뿐,
///  다른 거래에 섞이지 않습니다. '틀린 묶음' 보다 '고립' 이 항상 안전합니다.
///
///  ⚠️ 값 탐색은 find_relay_value(루트 → 카테고리 객체 → 배열 원소)를 그대로 씁니다.
///     루트만 보면 header/logistics/containers 안의 참조를 통째로 놓칩니다.
pub fn resolve_trade_hub_key(doc_type: &str, data: &Value, fallback_id: &str) -> String {
    let norm = |s: &str| crate::utils::hash::normalize_identifier(s);
    // ① 허브 참조
    for (_, field) in TRADE_HUB_FIELDS.iter() {
        if let Some(v) = find_relay_value(data, field) {
            let n = norm(&v);
            if !n.is_empty() { return n; }
        }
    }
    // ② 자기 자신이 허브 타입
    let code = doc_type.trim().to_uppercase();
    let self_no = find_relay_value(data, "doc_number")
        .map(|v| norm(&v))
        .unwrap_or_default();
    if TRADE_HUB_FIELDS.iter().any(|(c, _)| *c == code) && !self_no.is_empty() {
        return self_no;
    }
    // ③ 아무 참조라도
    for (_, fields) in TRADE_RELAY_FIELDS.iter() {
        for f in fields.iter() {
            if *f == "doc_number" { continue; }
            if let Some(v) = find_relay_value(data, f) {
                let n = norm(&v);
                if !n.is_empty() { return n; }
            }
        }
    }
    // ④ 최후
    if !self_no.is_empty() { return self_no; }
    norm(fallback_id)
}

/// 🌟 [TRADING REF] commerce 의 hash_id(team + cc + link) 자리에 hub_key 가 들어갑니다.
///    이 한 값으로 그 거래의 전 서류(PI / CI / BL / SOA ...)가 한 스코프에 묶입니다.
pub fn trading_ref(team: &str, cc: &str, hub_key: &str) -> String {
    crate::utils::hash::hash_id(&format!("{}{}#{}", team, cc, hub_key))
}

/// 🌟 [TRADING ENVELOPE] doc_type + data 로 (cc, bcc, ref) 를 한 번에 확정합니다.
///    호출부가 세 축을 따로 계산하면 그중 하나만 어긋나는 사고가 반복됩니다.
///    bcc 는 entity_bcc 와 동일한 hash_id(type + cc) 이며 평문 cc 를 씁니다.
pub fn trading_envelope(team: &str, doc_type: &str, data: &Value, fallback_id: &str)
    -> (String, String, String)
{
    let cc = trading_cc();
    let bcc = crate::utils::hash::hash_id(&format!("{}{}", doc_type, cc));
    let hub = resolve_trade_hub_key(doc_type, data, fallback_id);
    let r = trading_ref(team, &cc, &hub);
    (cc, bcc, r)
}
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
    // 🌟 [ALLOW-LIST 폐기] 기존 구현은 "ASCII + 한글만 통과" 화이트리스트였습니다.
    //    무역 문서는 다국어가 원칙이므로 이 필터는 값을 조용히 파괴합니다.
    //      Gaisbergstraße → Gaisbergstrae   (독일 수하인 주소, ß = U+00DF 소멸)
    //      Köln → Kln,  中国 / 深圳 → 전소  (country_of_manufacture 의 표준 표기)
    //      € ¥ £ ° № → 전소               (currency / 단위)
    //    파괴된 문자열은 STAGE-6 접지 검증에서 '이미지에 없는 값' 으로 폐기되거나,
    //    normalize_identifier 를 통과해도 원본과 다른 index 로 떨어져 릴레이가 끊깁니다.
    //    따라서 화이트리스트를 버리고, LLM 파이프라인을 실제로 깨뜨리는 문자만
    //    블랙리스트로 제거합니다.
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
    let re_attr = Regex::new(r#"(?i)\b(data-[a-z0-9\-]+|placeholder|rowspan|colspan|disabled|readonly|selected|summary|headers|checked|class|scope|title|value|abbr|href|type|name|rows|cols|alt|for|src|id)(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s>]+))?(?=[\s/>]|$)"#).unwrap();
    
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
                            let canonical = canonicalize_trade_column(&title);
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

// =====================================================================
// 🌟 [TRADE COLUMN DICTIONARY] 인쇄된 컬럼명 ↔ 스키마 필드명 사전
// ---------------------------------------------------------------------
//  이 사전이 없어서 실측 로그에서 'UNIT WEIGHT' 열(값 0.1)이 통째로 버려졌습니다.
//  스키마 필드는 item_net_weight 인데 인쇄 라벨은 'UNIT WEIGHT' 라서
//  모델이 둘을 잇지 못하고 item_net_weight / item_gross_weight 를 모두 null 로 반환했습니다.
//  bias.json 의 의미 구(semantic phrase)는 '위치를 찾는' 용도이고,
//  이 사전은 '찾은 열의 이름을 확정하는' 용도입니다. 역할이 다르므로 분리합니다.
// =====================================================================
pub static TRADE_COLUMN_ALIASES: &[(&str, &[&str])] = &[
    ("description", &[
        "description of goods", "description", "goods description", "commodity",
        "commodity description", "description of merchandise", "article", "articles",
        "item description", "product", "product name", "name of goods", "nature of goods",
        "품명", "상품명", "품목", "화물명", "물품명",
    ]),
    ("hs_code", &[
        "hs code", "h s code", "hs-code", "hscode", "hts code", "hs tariff",
        "tariff code", "commodity code", "hs no", "hs number", "tariff no",
        "세번", "세번부호", "hs부호",
    ]),
    ("country_of_manufacture", &[
        "country of manufacture", "country of origin", "made in", "origin",
        "manufacturing country", "country", "coo",
        "원산지", "생산국", "제조국",
    ]),
    ("unit", &[
        "unit of measure", "unit of measurement", "uom", "measure", "unit",
        "packing unit", "measurement unit",
        "단위", "거래단위",
    ]),
    ("quantity", &[
        "qty", "quantity", "q ty", "pieces", "pcs", "no of pcs", "number of units",
        "number of pieces", "shipped qty",
        "수량", "주문수량",
    ]),
    ("item_net_weight", &[
        "unit weight", "net weight", "n w", "nw", "net wt", "unit net weight",
        "weight per unit", "net weight kg",
        "순중량", "단위중량", "개당중량",
    ]),
    ("item_gross_weight", &[
        "gross weight", "g w", "gw", "gross wt", "total weight", "gross weight kg",
        "총중량", "총 중량",
    ]),
    ("unit_price", &[
        "unit value", "unit price", "price per unit", "u price", "unit cost",
        "rate", "price",
        "단가",
    ]),
    ("total_price", &[
        "total value", "total price", "line total", "extended price", "extended value",
        "total amount", "amount", "value",
        "금액", "합계", "공급가액",
    ]),
    ("item_code", &[
        "item code", "item no", "item number", "sku", "part number", "part no",
        "model", "model no", "article no", "product code", "style no",
        "품번", "모델", "품목코드",
    ]),
    ("item_package_count", &[
        "packages", "no of packages", "package count", "cartons", "ctns",
        "no of cartons", "number of packages", "case",
        "포장수", "박스수", "포장개수",
    ]),
    ("item_package_type", &[
        "package type", "kind of package", "packing", "type of package",
        "packing type", "kind of packages",
        "포장형태", "포장종류",
    ]),
];

/// 🌟 [LABEL ECHO] 서식의 '박스 라벨' 목록.
///  실측 로그에서 reference_invoice 가 "CONSIGNEE VAT/EORI" 를,
///  party_name 이 "SIGNATORY COMPANY" 를 값으로 받았습니다.
///  기존 '스키마 에코' 필터는 스키마 필드명(reference_invoice 등)만 잡기 때문에
///  인쇄 라벨이 값 자리로 들어오는 이 경로를 막지 못합니다.
pub static TRADE_PRINTED_LABELS: &[&str] = &[
    "invoice number", "invoice no", "invoice total", "airwaybill bill of lading",
    "date of exportation", "export reference", "exporter", "consignee",
    "exporter vat eori", "consignee vat eori", "vat eori",
    "country of export", "buyer if not consignee", "reason for export",
    "country of ultimate destination", "total number of packages", "total weight",
    "incoterm", "incoterms", "currency", "signature of exporter",
    "signatory name", "signatory company", "date", "shipper", "notify party",
    "description of goods", "hs code", "country of manufacture", "unit of measure",
    "qty", "unit weight", "unit value", "total value",
    "품명", "수량", "단가", "금액", "원산지", "세번부호",
];

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
pub fn canonicalize_trade_column(raw_header: &str) -> String {
    let norm = fold_column_label(raw_header);
    if norm.is_empty() { return String::new(); }

    // 1. 완전 일치 우선
    for (field, aliases) in TRADE_COLUMN_ALIASES.iter() {
        for a in aliases.iter() {
            if fold_column_label(a) == norm { return field.to_string(); }
        }
    }

    // 2. 부분 일치 — 가장 긴 별칭이 이깁니다.
    //    'unit weight'(11자) 가 'unit'(4자) 보다 먼저 잡혀야
    //    item_net_weight 로 가고 unit 으로 오배정되지 않습니다.
    let mut best_len = 0usize;
    let mut best_field = "";
    for (field, aliases) in TRADE_COLUMN_ALIASES.iter() {
        for a in aliases.iter() {
            let a_norm = fold_column_label(a);
            if a_norm.is_empty() { continue; }
            if norm.contains(&a_norm) && a_norm.len() > best_len {
                best_len = a_norm.len();
                best_field = field;
            }
        }
    }
    best_field.to_string()
}

/// 추출된 '값' 이 사실은 인쇄 라벨인지 판정합니다. true 면 폐기해야 합니다.
pub fn is_printed_label_echo(value: &str) -> bool {
    let norm = fold_column_label(value);
    if norm.is_empty() { return false; }
    if TRADE_PRINTED_LABELS.iter().any(|l| fold_column_label(l) == norm) { return true; }
    TRADE_COLUMN_ALIASES.iter().any(|(_, aliases)| {
        aliases.iter().any(|a| fold_column_label(a) == norm)
    })
}

/// 🌟 [ROW CONTRACT] 표 타일 프롬프트에 실을 계약 문자열을 만듭니다.
///  실측 로그에서 타일 2 는 T-Shirt 행과 Shorts 행을 둘 다 담고 있었는데
///  응답이 객체 1개였고, 파이프라인이 그것을 배열 1원소로 승격했습니다.
///  ("ARRAY COERCE 단일 객체 응답을 원소 1개 배열로 승격합니다")
///  그 결과 Shorts 행이 영구 소멸했습니다. 행 수 = 타일 수가 되어 버립니다.
///  타일 스키마를 배열로 고정하고, 인쇄 라벨과 필드명의 대응을 명시해
///  '보이는 데이터 행 수만큼' 원소를 만들도록 강제합니다.
pub fn build_table_row_contract(headers: &[Vec<String>]) -> String {
    if headers.is_empty() { return String::new(); }

    let width = headers.iter().map(|r| r.len()).max().unwrap_or(0);
    let mut mapping = Vec::new();
    let mut fields = Vec::new();

    for col in 0..width {
        let mut parts: Vec<&str> = Vec::new();
        for h_row in headers.iter() {
            if let Some(seg) = h_row.get(col) {
                let seg = seg.trim();
                if !seg.is_empty() && !parts.contains(&seg) { parts.push(seg); }
            }
        }
        let printed = parts.join(" ");
        if printed.is_empty() { continue; }
        let field = canonicalize_trade_column(&printed);
        if field.is_empty() { continue; }
        mapping.push(format!("  column {} \"{}\" -> \"{}\"", col, printed, field));
        if !fields.contains(&field) { fields.push(field); }
    }

    if mapping.is_empty() { return String::new(); }

    let obj = fields.iter()
        .map(|f| format!("\"{}\": null", f))
        .collect::<Vec<_>>()
        .join(", ");

    format!(
"COLUMN MAP (printed header -> output field):\n{}\n\
RULES:\n\
  - Return a JSON ARRAY. One object per printed data row. Never collapse rows.\n\
  - If the image shows 2 data rows, the array MUST have 2 elements.\n\
  - Do not output header rows, subtotal rows, or total rows as elements.\n\
  - Use null for a column that is not printed on that row.\n\
SCHEMA:\n[ {{ {} }} ]",
        mapping.join("\n"), obj
    )
}

/// 🌟 [HEADER BAND v2] 기존 구현이 못 잡던 3가지를 잡습니다.
///   ① thead 부재 : 무역 서식 다수는 thead 없이 표의 첫 tr 을 th 로 씁니다.
///      기존 코드는 이때 빈 배열을 돌려주고, 빈 배열이면 generate_pug_lines 의
///      alt= 주입 자체가 일어나지 않아 컬럼명이 LLM 에 도달하지 못했습니다.
///   ② colspan : 'UNIT' 이 2열을 덮는 2단 헤더에서 셀을 1열로 세면
///      그 뒤 모든 열의 라벨이 한 칸씩 밀립니다.
///   ③ rowspan : 'DESCRIPTION' 이 2행을 덮으면 아래 행의 그 열이 빈칸이 되어
///      본문 셀이 라벨을 못 받습니다.
///   결과는 항상 직사각 격자(모든 행의 길이가 같음)로 반환합니다.
pub fn extract_doc_table_headers(document: &Html, table_selector: &str) -> Vec<Vec<String>> {
    let empty: Vec<Vec<String>> = Vec::new();

    let sel = match Selector::parse(table_selector) { Ok(s) => s, Err(_) => return empty };
    let first_match = match document.select(&sel).next() { Some(m) => m, None => return empty };

    // 1. 선택자에서 위로 올라가 감싸는 <table> 을 찾습니다. (기존 동작 유지)
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
    let table_ref = match table_ref { Some(t) => t, None => return empty };

    let tr_sel = match Selector::parse("tr") { Ok(s) => s, Err(_) => return empty };
    let cell_sel = match Selector::parse("th, td") { Ok(s) => s, Err(_) => return empty };

    // 2. 헤더 행 후보 수집
    let mut header_rows: Vec<scraper::ElementRef> = Vec::new();
    if let Ok(thead_sel) = Selector::parse("thead") {
        if let Some(thead) = table_ref.select(&thead_sel).next() {
            header_rows.extend(thead.select(&tr_sel));
        }
    }
    if header_rows.is_empty() {
        // thead 가 없으면 표 선두에서 'th 우세 행' 또는 'scope=col 보유 행' 이
        // 이어지는 동안을 헤더 밴드로 봅니다. 데이터 행을 만나면 즉시 중단합니다.
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
    if header_rows.is_empty() { return empty; }

    let band = header_rows.len();

    // 3. colspan / rowspan 을 펼쳐 직사각 격자로 만듭니다.
    let mut grid: Vec<Vec<String>> = vec![Vec::new(); band];
    for (r, tr) in header_rows.iter().enumerate() {
        let mut c = 0usize;
        for cell in tr.select(&cell_sel) {
            // 위 행의 rowspan 이 이미 점유한 칸을 건너뜁니다.
            while grid[r].len() > c && !grid[r][c].is_empty() { c += 1; }

            let colspan = cell.value().attr("colspan")
                .and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(1).clamp(1, 64);
            let rowspan = cell.value().attr("rowspan")
                .and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(1).clamp(1, 64)
                .min(band - r); // 헤더 밴드 밖으로 흘러넘치지 않게 잘라냅니다.

            // abbr 가 있으면 그것이 정식 컬럼명입니다. 없으면 인쇄 텍스트를 씁니다.
            let raw = cell.value().attr("abbr")
                .map(|s| s.to_string())
                .unwrap_or_else(|| cell.text().collect::<Vec<_>>().join(" "));
            let title = raw.split_whitespace().collect::<Vec<_>>().join(" ");

            for dr in 0..rowspan {
                let rr = r + dr;
                for dc in 0..colspan {
                    let cc = c + dc;
                    if grid[rr].len() <= cc { grid[rr].resize(cc + 1, String::new()); }
                    grid[rr][cc] = title.clone();
                }
            }
            c += colspan;
        }
    }

    let width = grid.iter().map(|r| r.len()).max().unwrap_or(0);
    if width == 0 { return empty; }
    for row in grid.iter_mut() { row.resize(width, String::new()); }
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

// =====================================================================
// 🌟 [TRADE RELAY POINT] mode="trading" 의 문서 연결 진입점
// ---------------------------------------------------------------------
//  ── 왜 필요한가 ──
//   mode="commerce" 는 type="tracking" 과 type="order" 가
//     tracking_number → hash::normalize_identifier → crc32 → index
//   라는 단일 경로로 만나기 때문에 릴레이가 성립합니다.
//   반면 trading 에는 이 경로에 해당하는 함수가 코드에 존재하지 않았습니다.
//   parsing.rs 는 '어떤 카테고리를 뽑을지'(get_trade_doc_categories)만 정하고
//   '무엇으로 문서를 이을지'는 아무 것도 정하지 않았습니다.
//
//  ── 실측 실패 ──
//   ["PL←doc_number(빈 키)", "BL←doc_number(빈 키)",
//    "ED←doc_number(빈 키)", "AWB←flight_number(빈 키)"]
//   ① 규칙 3개가 doc_number 하나에만 매달려 있습니다.
//      CI↔PL 은 인보이스번호, CI↔BL 은 B/L 번호, CI↔PO 는 발주번호로 이어지는데
//      단일 키로는 셋 중 하나만 맞아도 나머지 둘이 끊깁니다.
//   ② AWB←flight_number 는 의미가 틀렸습니다. flight_number 는 편명(KE083)이고,
//      AWB 를 특정하는 것은 항공운송장 번호입니다. 이 서식은 그 번호를
//      AIRWAYBILL / BILL OF LADING 박스에 93763111837 로 인쇄하고 있습니다.
//      즉 추출이 완벽했더라도 이 규칙표로는 AWB 릴레이가 성립하지 않습니다.
//   ③ 키가 하나도 없을 때 task_id 로 폴백했습니다. task_id 는 실행마다 달라지므로
//      같은 문서가 재실행될 때마다 새 id 를 받습니다. 실측 로그에서
//      0xef44… (이전 실행) 과 0xcf1c… (이번 실행) 로 갈렸습니다.
//
//  좌표를 코드에 적지 않는 vision_crop 의 원칙과 같은 이유로,
//  키 이름도 문서 종류마다 하드코딩하지 않고 '역할' 로 선언합니다.
// =====================================================================

/// 릴레이 키의 역할. 같은 역할끼리만 서로 연결됩니다.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TradeRelayKey {
    pub role: &'static str,   // "self" | "transport" | "booking" | "order" | "contract" | "credit" | "container"
    pub source_field: String, // 어느 필드에서 나왔는지 (진단용)
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

/// 🌟 추출 JSON 에서 릴레이 키를 전부 뽑습니다.
///  commerce 의 tracking_number 한 개에 대응하는, trading 쪽의 다중 키 버전입니다.
/// 🌟 [RELAY KEYS v4]
///  ── 무엇이 바뀌었나 ──
///   기존은 `find_relay_value` 로 값을 찾고 `is_printed_label_echo` 로
///   플레이스홀더를 차단한 뒤 `is_valid_relay_key` 로 유효성을 검증했는데,
///   이 순서에서 `is_valid_relay_key` 가 `normalize_identifier` 를
///   내부에서 다시 호출하여 전각 접기가 두 번 수행되었습니다.
///   또한 `relay_index` 와 `relay_id` 에 원본 `raw` 를 전달했는데,
///   이 값은 이미 `normalize_identifier` 를 거친 `normalized` 와 다를 수 있습니다.
///   일관성을 위해 `normalized` 를 기준으로 인덱스와 id 를 계산합니다.
pub fn extract_trade_relay_keys(data: &Value) -> Vec<TradeRelayKey> {
    let mut out: Vec<TradeRelayKey> = Vec::new();
    for (role, fields) in TRADE_RELAY_FIELDS.iter() {
        for f in fields.iter() {
            let raw = match find_relay_value(data, f) { Some(r) => r, None => continue };
            
            // 🌟 인쇄 라벨이 값 자리로 들어온 경우를 여기서 차단합니다.
            if is_printed_label_echo(&raw) { continue; }
            
            // 🌟 유효성 검증을 먼저 수행합니다.
            //    is_valid_relay_key 가 내부에서 전각 접기를 수행하므로
            //    여기서 한 번만 수행됩니다.
            if !crate::utils::hash::is_valid_relay_key(&raw) { continue; }
            
            let normalized = crate::utils::hash::normalize_identifier(&raw);
            
            // 🌟 normalized 기준으로 인덱스와 id 를 계산하여 일관성 보장
            let index = crate::utils::hash::relay_index(&normalized);
            if index == 0 { continue; }
            if out.iter().any(|k| k.role == *role && k.index == index) { continue; }
            
            out.push(TradeRelayKey {
                role,
                source_field: f.to_string(),
                raw: raw.clone(),
                normalized: normalized.clone(),
                index,
                id: crate::utils::hash::relay_id(&normalized),
            });
        }
    }
    out
}

/// 🌟 문서 식별자를 확정합니다. task_id 폴백을 제거하는 것이 목적입니다.
///  반환: (정규화된 키, index, 폴백 여부)
///  폴백이 필요하면 task_id 대신 '내용 지문' 을 씁니다.
///  내용 지문은 같은 문서를 다시 태워도 같은 값이 나오므로
///  0xef44… / 0xcf1c… 처럼 id 가 갈리는 일이 생기지 않습니다.
/// 🌟 [DOC IDENTITY v4]
///  ── 무엇이 바뀌었나 ──
///   기존은 4순위까지 순차 탐색 후 내용 지문으로 폴백했는데,
///   내용 지문은 `crc32` 만 사용하고 `relay_index` 를 사용하지 않아
///   전각 영숫자가 포함된 경우 다른 인덱스가 생성되었습니다.
///   또한 1순위에서 `k.normalized` 을 반환했는데,
///   이 값은 `normalize_identifier` 통과값이므로 이미 정규화되어 있습니다.
///   하지만 `k.index` 는 `extract_trade_relay_keys` 내부에서 계산된 값이라
///   `relay_index` 와 동일한 경로를 거치지 않을 수 있었습니다.
///   여기서 `relay_index` 를 다시 호출하여 일관성을 보장합니다.
pub fn resolve_trade_doc_identity(doc_type: &str, data: &Value) -> (String, u32, bool) {
    let keys = extract_trade_relay_keys(data);
    
    // 1순위: 이 문서 자신의 번호
    if let Some(k) = keys.iter().find(|k| k.role == "self") {
        // 🌟 relay_index 를 다시 호출하여 일관성 보장
        let idx = crate::utils::hash::relay_index(&k.normalized);
        return (k.normalized.clone(), idx, false);
    }
    
    // 2순위: 운송증권 번호 (CI 에도 인쇄됩니다 — 이 서식의 93763111837)
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

/// 🌟 릴레이 규칙 평가. 어떤 문서 종류로 이어질 수 있는지를 돌려줍니다.
///  기존 규칙표의 "AWB←flight_number" 의미 오류를 여기서 바로잡습니다.
/// 🌟 [RELAY PLAN v4]
///  ── 무엇이 바뀌었나 ──
///   기존은 역할별로 하드코딩된 타겟 목록을 사용했는데,
///   이 목록이 `logic.rs` 의 `related_trading` 과 어긋나는 경우가 있었습니다.
///   또한 `k.clone()` 으로 키를 복제했는데, 이 복제가 불필요한 메모리 할당을 유발했습니다.
///   역할별 타겟을 `related_trading` 의 허브 목록과 일치시키고,
///   불필요한 복제를 제거합니다.
pub fn plan_trade_relays(doc_type: &str, data: &Value) -> Vec<(&'static str, TradeRelayKey)> {
    let keys = extract_trade_relay_keys(data);
    let mut plan = Vec::new();
    
    for k in keys.into_iter() {
        // 🌟 역할별 타겟을 `logic.rs` 의 `related_trading` 허브와 일치시킵니다.
        let targets: &[&'static str] = match k.role {
            // 같은 인보이스 번호를 인쇄하는 문서들
            "self"      => &["PL", "CINV", "TI", "SOA", "DN", "CN"],
            // 운송증권 번호를 공유하는 문서들.
            // 🌟 [MISSING] CSI, BK, SR, WR, BE, IP, DN, CN, FC 추가
            "transport" => &["BL", "HBL", "SWB", "AWB", "DO", "AN", "POD", "FCR", "ED", "ID", "CSI", "BK", "SR", "WR", "BE", "IP", "DN", "CN", "FC"],
            "booking"   => &["BK", "SR", "BC"],
            "order"     => &["PO", "PI", "SC"],
            "contract"  => &["SC", "PI", "CP"],
            "credit"    => &["LC", "LLC", "BE"],
            "container" => &["PL", "BL", "DO", "CM"],
            _ => &[],
        };
        
        for t in targets.iter() {
            if *t == doc_type { continue; }
            plan.push((*t, k.clone()));
        }
    }
    plan
}
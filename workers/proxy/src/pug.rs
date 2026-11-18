//! `convertHtmlToCleanPug(body)` · `generatePugLines` · `parseBody` (원본 1020–1158 · 1984–2032).
//!
//! `$Contains`(1003–1018)는 Workers 에 없는 전역 `document` 를 쓰고 어디서도 부르지 않아 옮기지 않았다.

use logis_worker_common::jsv::{js_trim, to_lower};
use logis_worker_common::{console_log, Throw, R, V};

use crate::dom::{Data, Document, NodeId};

/// 늘 넣는 속성
const ALWAYS_INCLUDE: &[&str] =
    &["src", "href", "type", "name", "value", "placeholder", "checked", "selected", "disabled", "readonly", "rows", "cols"];
/// 값 없이 이름만 쓰는 불리언 속성
const BOOLEAN_ATTRS: &[&str] = &["checked", "selected", "disabled", "readonly"];
/// 건너뛰는 태그
const SKIP_TAGS: &[&str] = &["script", "style", "link", "noscript", "iframe"];

/// `convertHtmlToCleanPug(body)` — 실패하면 ''
pub fn convert_html_to_clean_pug(body: &V) -> V {
    let document = Document::parse(&format!("<html><body>{}</body></html>", body.to_str()));
    let result = document.body().and_then(|b| match b {
        Some(b) => generate_pug_lines(&document, document.children(b)),
        None => Ok(Vec::new()),
    });
    match result {
        Ok(lines) => V::from(lines.join("\n")),
        Err(error) => {
            console_log!("변환 중 오류가 발생했습니다:", error);
            V::from("")
        }
    }
}

/// `generatePugLines(nodes, 0)` — 재귀 대신 명시적 스택(깊은 문서에서 wasm 스택이 넘치지 않게). 줄 순서는 원본과 같다
fn generate_pug_lines(doc: &Document, nodes: &[NodeId]) -> R<Vec<String>> {
    let mut lines: Vec<String> = Vec::new();
    // (형제 목록, 다음 위치, 들여쓰기 단계)
    let mut stack: Vec<(&[NodeId], usize, usize)> = vec![(nodes, 0, 0)];
    while let Some(frame) = stack.last_mut() {
        let (list, pos, level) = (frame.0, frame.1, frame.2);
        if pos >= list.len() {
            stack.pop();
            continue;
        }
        frame.1 += 1;
        let node = list[pos];
        let indent = "    ".repeat(level);
        match &doc.node(node).data {
            Data::Element(element) => {
                let tag_name = to_lower(&element.name.to_uppercase());

                // base64 이미지를 포함하는 img 태그 제외
                let src = doc.get_attribute(node, "src");
                if tag_name == "img" && src.as_deref().is_some_and(|s| !s.is_empty() && s.contains("base64")) {
                    continue;
                }
                if SKIP_TAGS.contains(&tag_name.as_str()) {
                    continue;
                }

                let mut attributes_string = String::new();
                let mut other_attributes: Vec<String> = Vec::new();

                // node.id = getAttribute('id') || ''
                let id = doc.get_attribute(node, "id").unwrap_or_default();
                if !id.is_empty() {
                    attributes_string.push('#');
                    attributes_string.push_str(&id);
                }
                if !element.class_list.is_empty() {
                    attributes_string.push('.');
                    attributes_string.push_str(&element.class_list.join("."));
                }
                for (attr_name, attr_value) in &element.attrs {
                    if attr_name == "id" || attr_name == "class" {
                        continue;
                    }
                    if attr_name.starts_with("data-") || ALWAYS_INCLUDE.contains(&attr_name.as_str()) {
                        if BOOLEAN_ATTRS.contains(&attr_name.as_str()) && (attr_value.is_empty() || attr_value == attr_name) {
                            other_attributes.push(attr_name.clone());
                        } else if !attr_value.is_empty() {
                            let safe_value = attr_value.replace('"', "'");
                            other_attributes.push(format!("{attr_name}=\"{safe_value}\""));
                        }
                    }
                }
                if !other_attributes.is_empty() {
                    attributes_string.push('(');
                    attributes_string.push_str(&other_attributes.join(" "));
                    attributes_string.push(')');
                }

                // div 축약: 자식(요소 또는 공백이 아닌 글 · 주석)이 하나뿐이고 그것이 DIV 면 안으로 내려간다
                let mut current = node;
                while doc.tag_name(current).as_deref() == Some("DIV")
                    && doc
                        .children(current)
                        .iter()
                        .filter(|c| doc.element(**c).is_some() || doc.node_value(**c).is_some_and(|v| !js_trim(v).is_empty()))
                        .count()
                        == 1
                    && doc.first_element_child(current).and_then(|c| doc.tag_name(c)).as_deref() == Some("DIV")
                {
                    current = doc.first_element_child(current).unwrap_or(current);
                }

                lines.push(format!("{indent}{tag_name}{attributes_string}"));

                if tag_name == "textarea" {
                    // node.value — SVG 안의 textarea(SVGElement)는 value 가 undefined 라 .trim() 에서 던진다
                    if element.svg {
                        return Err(Throw::type_error("Cannot read properties of undefined (reading 'trim')"));
                    }
                    let value = doc.text_content(node);
                    if !js_trim(&value).is_empty() {
                        for line in value.split('\n') {
                            lines.push(format!("{indent}    | {line}"));
                        }
                    }
                } else if !doc.children(current).is_empty() {
                    stack.push((doc.children(current), 0, level + 1));
                }
            }
            Data::Text { text, .. } => {
                let text_content = js_trim(text);
                if !text_content.is_empty() {
                    lines.push(format!("{indent}| {text_content}"));
                }
            }
            _ => {}
        }
    }
    Ok(lines)
}

/// `parseBody(body, page)` — 원본에서 부르는 곳은 없다. 원본 그대로 옮긴다:
/// - `var body = ''` 가 인자를 가려 늘 빈 문서를 읽는다
/// - 반복 안의 `selectors.hasOwnProperty(s)` 는 정의되지 않은 전역 `selectors` 를 읽어, `page.selectors` 에 열거되는 키가
///   하나라도 있으면 첫 바퀴에서 ReferenceError 를 던진다. 그래서 그 뒤(querySelector · type/checked/selected/value 분기)는
///   닿지 않고, 던지지 않을 때의 결과는 늘 `[]` 다
pub fn parse_body(_body: &V, page: &V) -> R<V> {
    // 원본 그대로: var body = ''
    let body = "";
    let _document = Document::parse(&format!("<html><body>{body}</body></html>"));
    let results = V::new_arr();
    // for (const s in page.selectors) — page 가 undefined · null 이면 TypeError, selectors 가 undefined · null 이면 반복 없음
    let selectors = page.g("selectors")?;
    if !selectors.keys().is_empty() {
        // 원본 그대로: selectors 는 parseBody 의 범위에 없다
        return Err(Throw { name: "ReferenceError", message: "selectors is not defined".into() });
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use logis_worker_common::obj;

    #[test]
    fn pug_basics() {
        let html = r#"<div class="a"><div><p id="x" data-k='v"w'>hi &amp; bye</p></div></div><img src="data:image/png;base64,AA"><input type="checkbox" checked><script>x</script>"#;
        let pug = convert_html_to_clean_pug(&V::from(html));
        assert_eq!(pug.to_str(), "div.a\n    p#x(data-k=\"v'w\")\n        | hi\n        | &\n        | bye\ninput(type=\"checkbox\" checked)");
    }

    #[test]
    fn parse_body_quirks() {
        assert_eq!(parse_body(&V::from("<p>x</p>"), &obj! { "selectors" => obj! {} }).unwrap().len(), 0);
        assert_eq!(parse_body(&V::Undef, &obj! {}).unwrap().len(), 0);
        let err = parse_body(&V::Undef, &obj! { "selectors" => obj! { "title" => "h1" } }).unwrap_err();
        assert_eq!(err.name, "ReferenceError");
        assert_eq!(parse_body(&V::Undef, &V::Undef).unwrap_err().name, "TypeError");
    }
}

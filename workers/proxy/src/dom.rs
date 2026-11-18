//! linkedom 0.18.11 (`parseHTML`, 안의 htmlparser2 10.1.0) 과 같은 트리 + CSS 선택자.
//!
//! htmlparser2 는 HTML5 규격 파서가 아니다: `<html>`·`<body>` 를 보정하지 않고, 닫히지 않은 태그는 `openImpliesClose` 표대로만
//! 닫으며, 글을 엔터티 앞뒤 · `<` 앞에서 여러 Text 노드로 가른다. 원본이 만든 Pug 글(LLM 프롬프트)과 선택자 일치 수가 같아야 해서
//! 토크나이저 · 파서 · 엔터티 해독을 그대로 옮겼다([`parser`] · [`entities`]). 선택자는 scraper 의 선택자 문법으로 읽어
//! 이 트리에 직접 맞춘다([`select`]).
//!
//! linkedom 이 노드에 보이는 값:
//! - `tagName` = localName 의 대문자(SVG 안의 요소도 문서가 HTML 이라 대문자), localName 은 htmlparser2 가 소문자로 바꾼 이름
//! - 속성 이름은 쓴 그대로(대소문자 보존), 같은 이름은 먼저 나온 것, 순서는 JS 객체 키 순서(첨자 꼴 이름이 먼저)
//! - `class` 속성값은 classList(JS `\s+` 로 가르고 중복 제거)를 공백 하나로 다시 이은 값, `getAttribute('class')` 는 없어도 ''
//! - Text 노드는 `ontext` 콜백 하나에 하나(이웃해도 합치지 않는다)

mod entities;
mod entity_tree;
mod parser;
mod select;

use logis_worker_common::{Throw, R};

/// 노드 번호 (0 = 문서). 만들어진 순서 = 문서 순서
pub type NodeId = usize;

/// 요소
#[derive(Debug, Clone)]
pub struct Element {
    /// localName (htmlparser2 가 소문자로 바꾼 태그 이름)
    pub name: String,
    /// (이름, 값) — 원본 순서, 이름은 대소문자 그대로
    pub attrs: Vec<(String, String)>,
    /// `classList` (순서대로, 중복 없음)
    pub class_list: Vec<String>,
    /// linkedom 이 SVGElement 로 만든 요소 (`<svg>` 와 그 자손)
    pub svg: bool,
}

#[derive(Debug, Clone)]
pub enum Data {
    Document,
    Element(Element),
    /// `unit`: 이름 엔터티가 낸 외톨이 서로게이트 한 단위(JS 문자열에선 그대로, 여기선 `text` 가 U+FFFD)
    Text { text: String, unit: Option<u16> },
    Comment(String),
}

#[derive(Debug, Clone)]
pub struct Node {
    pub parent: Option<NodeId>,
    /// 부모의 `children` 안에서의 위치 (형제 찾기)
    pub index: usize,
    pub children: Vec<NodeId>,
    pub data: Data,
}

/// `parseHTML(html).document`
pub struct Document {
    nodes: Vec<Node>,
}

impl Document {
    pub fn parse(html: &str) -> Document {
        Document { nodes: parser::parse(html) }
    }

    /// `document.querySelectorAll(selector)` — 잘못된 선택자는 SyntaxError
    pub fn query_selector_all(&self, selector: &str) -> R<Vec<NodeId>> {
        Ok(match select::parse(selector)? {
            Some(list) => select::select(self, &list),
            None => Vec::new(),
        })
    }

    /// `document.querySelector(selector)`
    pub fn query_selector(&self, selector: &str) -> R<Option<NodeId>> {
        Ok(self.query_selector_all(selector)?.into_iter().next())
    }

    /// `node.textContent` — 요소는 자손 Text 노드를 문서 순서로 이은 것(주석 제외), Text · 주석은 자기 글
    pub fn text_content(&self, id: NodeId) -> String {
        let Some(node) = self.nodes.get(id) else { return String::new() };
        match &node.data {
            Data::Text { text, .. } => text.clone(),
            Data::Comment(text) => text.clone(),
            Data::Document => String::new(),
            Data::Element(_) => {
                let mut out = String::new();
                let mut pending: Option<u16> = None;
                let mut stack: Vec<NodeId> = node.children.iter().rev().copied().collect();
                while let Some(cur) = stack.pop() {
                    let n = &self.nodes[cur];
                    match &n.data {
                        Data::Text { text, unit } => match unit {
                            // 갈라진 서로게이트 쌍은 이으면 다시 한 글자
                            Some(u) if (0xD800..=0xDBFF).contains(u) => {
                                if pending.replace(*u).is_some() {
                                    out.push('\u{FFFD}');
                                }
                            }
                            Some(u) => match pending.take() {
                                Some(hi) => {
                                    let c = 0x10000 + ((hi as u32 - 0xD800) << 10) + (*u as u32 - 0xDC00);
                                    out.push(char::from_u32(c).unwrap_or('\u{FFFD}'));
                                }
                                None => out.push('\u{FFFD}'),
                            },
                            None => {
                                if pending.take().is_some() {
                                    out.push('\u{FFFD}');
                                }
                                out.push_str(text);
                            }
                        },
                        Data::Element(_) => stack.extend(n.children.iter().rev().copied()),
                        _ => {}
                    }
                }
                if pending.is_some() {
                    out.push('\u{FFFD}');
                }
                out
            }
        }
    }

    /// `node.getAttribute(name)` — 이름은 정확히 같아야 한다. `class` 는 classList 값(속성이 없어도 '')
    pub fn get_attribute(&self, id: NodeId, name: &str) -> Option<String> {
        let e = self.element(id)?;
        if name == "class" {
            return Some(e.class_list.join(" "));
        }
        e.attrs.iter().find(|(n, _)| n == name).map(|(_, v)| v.clone())
    }

    // ───── Pug 변환 · 시험이 쓰는 읽기 ─────

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id]
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.len() <= 1
    }

    pub fn children(&self, id: NodeId) -> &[NodeId] {
        &self.nodes[id].children
    }

    pub fn element(&self, id: NodeId) -> Option<&Element> {
        match &self.nodes.get(id)?.data {
            Data::Element(e) => Some(e),
            _ => None,
        }
    }

    /// `node.tagName` (HTML 문서: localName 의 대문자)
    pub fn tag_name(&self, id: NodeId) -> Option<String> {
        self.element(id).map(|e| e.name.to_uppercase())
    }

    /// `node.nodeValue` (Text · 주석만, 요소는 null)
    pub fn node_value(&self, id: NodeId) -> Option<&str> {
        match &self.nodes[id].data {
            Data::Text { text, .. } | Data::Comment(text) => Some(text),
            _ => None,
        }
    }

    /// `node.firstElementChild`
    pub fn first_element_child(&self, id: NodeId) -> Option<NodeId> {
        self.nodes[id].children.iter().copied().find(|c| self.element(*c).is_some())
    }

    fn next_element_sibling(&self, id: NodeId) -> Option<NodeId> {
        let parent = self.nodes[id].parent?;
        let siblings = &self.nodes[parent].children;
        siblings[self.nodes[id].index + 1..].iter().copied().find(|c| self.element(*c).is_some())
    }

    /// `document.documentElement` (문서의 첫 요소)
    pub fn document_element(&self) -> Option<NodeId> {
        self.first_element_child(0)
    }

    /// `document.body` — documentElement 의 첫 요소가 HEAD 면 그 다음 요소, 아니면 첫 요소가 BODY 일 때 그것.
    /// linkedom 은 없으면 빈 head · body 를 새로 끼워 넣는다 → `Ok(None)` (빈 body). documentElement 가 없으면 TypeError
    pub fn body(&self) -> R<Option<NodeId>> {
        let root = self
            .document_element()
            .ok_or_else(|| Throw::type_error("Cannot read properties of null (reading 'firstElementChild')"))?;
        let first = self.first_element_child(root);
        let candidate = match first {
            Some(head) if self.tag_name(head).as_deref() == Some("HEAD") => self.next_element_sibling(head),
            other => other,
        };
        Ok(candidate.filter(|b| self.tag_name(*b).as_deref() == Some("BODY")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(doc: &Document, id: NodeId) -> Vec<String> {
        doc.children(id).iter().filter_map(|c| doc.node_value(*c).map(str::to_string)).collect()
    }

    #[test]
    fn text_splits_like_htmlparser2() {
        let doc = Document::parse("<p>a &amp; b < c &copy 1</p>");
        let p = doc.query_selector("p").unwrap().unwrap();
        assert_eq!(texts(&doc, p), ["a ", "&", " b ", "< c ", "©", " 1"]);
        assert_eq!(doc.text_content(p), "a & b < c © 1");
    }

    #[test]
    fn implied_tags_and_attributes() {
        let doc = Document::parse("<ul><li>a<li class=' x  y x'>b</ul></p><div ID=1 id=2 Data-X='q'>");
        assert_eq!(doc.query_selector_all("ul > li").unwrap().len(), 2);
        assert_eq!(doc.query_selector_all("p").unwrap().len(), 1);
        let li = doc.query_selector_all("li").unwrap()[1];
        assert_eq!(doc.element(li).unwrap().class_list, ["x", "y"]);
        assert_eq!(doc.get_attribute(li, "class").as_deref(), Some("x y"));
        let div = doc.query_selector("div").unwrap().unwrap();
        assert_eq!(doc.get_attribute(div, "id").as_deref(), Some("2"));
        assert_eq!(doc.get_attribute(div, "ID").as_deref(), Some("1"));
        assert!(doc.query_selector_all("[data-x]").unwrap().is_empty());
        assert_eq!(doc.query_selector_all("div[").unwrap_err().name, "SyntaxError");
    }

    #[test]
    fn many_siblings_stay_fast() {
        let html = format!("<ul>{}</ul>", "<li class=i><b>x</b></li>".repeat(5000));
        let doc = Document::parse(&html);
        assert_eq!(doc.query_selector_all("li:nth-child(odd)").unwrap().len(), 2500);
        assert_eq!(doc.query_selector_all("li:nth-last-of-type(2) b").unwrap().len(), 1);
        assert_eq!(doc.query_selector_all("li + li ~ li").unwrap().len(), 4998);
    }
}

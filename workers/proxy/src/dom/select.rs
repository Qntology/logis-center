//! CSS 선택자 — scraper 0.25 의 선택자 문법(`scraper::selector::Parser` · `Simple`, 곧 `selectors` 0.33)으로 읽고,
//! 우리 트리에 `selectors::Element` 를 직접 입혀 맞춘다.
//!
//! linkedom 은 css-select 를 `xmlMode: false` 로 부른다. 그와 같게 하려고:
//! - 모든 요소를 "HTML 문서의 HTML 요소"로 본다 → 선택자의 태그 · 속성 이름은 소문자로 바뀌어 정확히(대소문자 구분) 견주고,
//!   HTML 규정의 일부 속성값(type · checked 등)은 대소문자를 가리지 않는다
//! - `class` 의 값 비교는 linkedom 어댑터처럼 classList 값으로 한다(속성이 없어도 '' — `[class=""]` 가 class 없는 요소에 맞는다).
//!   있는지(`[class]`)는 실제 속성으로
//! - `<template>` 의 자손은 건너뛴다(linkedom querySelector(All) 의 순회), `:scope` 는 문서라 어느 요소에도 맞지 않는다
//! - css-what 이 받아 주는 꼴 몇 가지: 빈 선택자는 일치 없음, 숫자로 시작하는 `#1a` · `.1a` 도 이름으로.
//!   이름공간(`|`) · 주석(`/`)은 css-select 처럼 던진다

use std::borrow::Cow;
use std::fmt;

use selectors::attr::{AttrSelectorOperation, CaseSensitivity, NamespaceConstraint};
use selectors::bloom::BloomFilter;
use selectors::context::{
    MatchingContext, MatchingForInvalidation, MatchingMode, NeedsSelectorFlags, QuirksMode, SelectorCaches,
};
use selectors::matching::ElementSelectorFlags;
use selectors::parser::{ParseRelative, SelectorImpl, SelectorList};
use selectors::{Element as SelElement, OpaqueElement};

use scraper::selector::{CssLocalName, CssString, NonTSPseudoClass, Parser, PseudoElement, Simple};

use super::{Data, Document, Element, NodeId};
use logis_worker_common::{Throw, R};

type Namespace = <Simple as SelectorImpl>::NamespaceUrl;

/// css-what 의 이름 글자 `[\w\-°-￿]` (+ 이스케이프 시작)
fn css_what_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '\\' || c >= '\u{b0}'
}

/// 따옴표 · `[]` 밖에서 css-what 이 받는 `#`·`.` 이름을 CSS 로 고쳐 쓴다. css-what 의 이름 정규식은
/// `/^[^\\#]?(?:\\…|[\w\-°-￿])+/` 라 첫 글자가 숫자(`#1a`)이거나 아무 글자(`a..b` = class ".b")여도 된다 —
/// 그 첫 글자를 이스케이프한다. `|`(이름공간) · `/` 는 css-select 처럼 거절
fn css_what_compat(sel: &str) -> R<Cow<'_, str>> {
    let invalid = || Throw::syntax_error(format!("'{sel}' is not a valid selector"));
    let chars: Vec<char> = sel.chars().collect();
    let mut out = String::with_capacity(sel.len() + 8);
    let mut changed = false;
    let mut quote: Option<char> = None;
    let mut depth = 0usize;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        out.push(c);
        i += 1;
        if c == '\\' {
            if let Some(&n) = chars.get(i) {
                out.push(n);
                i += 1;
            }
            continue;
        }
        if let Some(q) = quote {
            if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '"' | '\'' => quote = Some(c),
            '[' => depth += 1,
            ']' => depth = depth.saturating_sub(1),
            '|' | '/' if depth == 0 => return Err(invalid()),
            '#' | '.' if depth == 0 => {
                let Some(&n) = chars.get(i) else { continue };
                let ident_start = n.is_ascii_alphabetic() || n == '_' || n == '\\' || n as u32 >= 0x80;
                if n.is_ascii_digit() {
                    // `#1a` → `#\31 a`
                    out.push_str(&format!("\\{:x} ", n as u32));
                    i += 1;
                    changed = true;
                } else if n == '-' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit() || *d == '-') {
                    out.push_str("\\-");
                    i += 1;
                    changed = true;
                } else if !ident_start && n != '-' && n != '#' && chars.get(i + 1).copied().is_some_and(css_what_name_char) {
                    // `a..b` → `a.\.b`
                    out.push('\\');
                    out.push(n);
                    i += 1;
                    changed = true;
                }
            }
            _ => {}
        }
    }
    Ok(if changed { Cow::Owned(out) } else { Cow::Borrowed(sel) })
}

/// 선택자 읽기 — 빈 선택자는 None(일치 없음)
pub(super) fn parse(sel: &str) -> R<Option<SelectorList<Simple>>> {
    if sel.trim_matches(|c: char| c.is_ascii_whitespace()).is_empty() {
        return Ok(None);
    }
    let text = css_what_compat(sel)?;
    let mut input = cssparser::ParserInput::new(&text);
    let mut parser = cssparser::Parser::new(&mut input);
    SelectorList::parse(&Parser, &mut parser, ParseRelative::No)
        .map(Some)
        .map_err(|_| Throw::syntax_error(format!("'{sel}' is not a valid selector")))
}

/// 문서 순서로 일치하는 요소 (`<template>` 자손 제외)
pub(super) fn select(doc: &Document, list: &SelectorList<Simple>) -> Vec<NodeId> {
    let mut caches = SelectorCaches::default();
    let mut out = Vec::new();
    let mut in_template = vec![false; doc.nodes.len()];
    for (id, node) in doc.nodes.iter().enumerate() {
        let Some(parent) = node.parent else { continue };
        in_template[id] = in_template[parent] || matches!(&doc.nodes[parent].data, Data::Element(e) if e.name == "template");
        if in_template[id] || !matches!(node.data, Data::Element(_)) {
            continue;
        }
        let mut context = MatchingContext::new(
            MatchingMode::Normal,
            None,
            &mut caches,
            QuirksMode::NoQuirks,
            NeedsSelectorFlags::No,
            MatchingForInvalidation::No,
        );
        // :scope = querySelectorAll 을 부른 문서 → 어느 요소와도 같지 않다
        context.scope_element = Some(OpaqueElement::new(&doc.nodes[0]));
        if selectors::matching::matches_selector_list(list, &El { doc, id }, &mut context) {
            out.push(id);
        }
    }
    out
}

#[derive(Clone, Copy)]
struct El<'a> {
    doc: &'a Document,
    id: NodeId,
}

impl fmt::Debug for El<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "El({})", self.id)
    }
}

impl<'a> El<'a> {
    fn e(&self) -> &'a Element {
        match &self.doc.nodes[self.id].data {
            Data::Element(e) => e,
            _ => unreachable!("El 은 요소만"),
        }
    }

    fn wrap(&self, id: NodeId) -> Option<El<'a>> {
        self.doc.element(id).map(|_| El { doc: self.doc, id })
    }

    fn sibling_element(&self, step: isize) -> Option<El<'a>> {
        let parent = self.doc.nodes[self.id].parent?;
        let siblings = &self.doc.nodes[parent].children;
        let mut pos = self.doc.nodes[self.id].index as isize;
        loop {
            pos += step;
            if pos < 0 || pos as usize >= siblings.len() {
                return None;
            }
            if let Some(el) = self.wrap(siblings[pos as usize]) {
                return Some(el);
            }
        }
    }
}

impl<'a> SelElement for El<'a> {
    type Impl = Simple;

    fn opaque(&self) -> OpaqueElement {
        OpaqueElement::new(&self.doc.nodes[self.id])
    }

    fn parent_element(&self) -> Option<Self> {
        self.doc.nodes[self.id].parent.and_then(|p| self.wrap(p))
    }

    fn parent_node_is_shadow_root(&self) -> bool {
        false
    }

    fn containing_shadow_host(&self) -> Option<Self> {
        None
    }

    fn is_pseudo_element(&self) -> bool {
        false
    }

    fn prev_sibling_element(&self) -> Option<Self> {
        self.sibling_element(-1)
    }

    fn next_sibling_element(&self) -> Option<Self> {
        self.sibling_element(1)
    }

    fn first_element_child(&self) -> Option<Self> {
        self.doc.first_element_child(self.id).map(|id| El { doc: self.doc, id })
    }

    fn is_html_element_in_html_document(&self) -> bool {
        true
    }

    fn has_local_name(&self, name: &CssLocalName) -> bool {
        self.e().name == *name.0
    }

    fn has_namespace(&self, ns: &Namespace) -> bool {
        // 모든 요소를 HTML 로 본다 (이름공간 선택자는 읽기에서 이미 거절)
        &**ns == "http://www.w3.org/1999/xhtml"
    }

    fn is_same_type(&self, other: &Self) -> bool {
        self.e().name == other.e().name
    }

    fn attr_matches(
        &self,
        ns: &NamespaceConstraint<&Namespace>,
        local_name: &CssLocalName,
        operation: &AttrSelectorOperation<&CssString>,
    ) -> bool {
        if let NamespaceConstraint::Specific(url) = ns {
            if !url.is_empty() {
                return false;
            }
        }
        let e = self.e();
        let name: &str = &local_name.0;
        if name == "class" && !matches!(operation, AttrSelectorOperation::Exists) {
            return operation.eval_str(&e.class_list.join(" "));
        }
        e.attrs.iter().find(|(n, _)| n == name).is_some_and(|(_, v)| operation.eval_str(v))
    }

    fn match_non_ts_pseudo_class(&self, pc: &NonTSPseudoClass, _context: &mut MatchingContext<'_, Simple>) -> bool {
        match *pc {}
    }

    fn match_pseudo_element(&self, pe: &PseudoElement, _context: &mut MatchingContext<'_, Simple>) -> bool {
        match *pe {}
    }

    fn apply_selector_flags(&self, _flags: ElementSelectorFlags) {}

    /// css-select `:any-link` = `:is(a, area, link)[href]`
    fn is_link(&self) -> bool {
        let e = self.e();
        matches!(e.name.as_str(), "a" | "area" | "link") && e.attrs.iter().any(|(n, _)| n == "href")
    }

    fn is_html_slot_element(&self) -> bool {
        false
    }

    fn has_id(&self, id: &CssLocalName, case_sensitivity: CaseSensitivity) -> bool {
        let e = self.e();
        e.attrs.iter().find(|(n, _)| n == "id").is_some_and(|(_, v)| case_sensitivity.eq(id.0.as_bytes(), v.as_bytes()))
    }

    fn has_class(&self, name: &CssLocalName, case_sensitivity: CaseSensitivity) -> bool {
        self.e().class_list.iter().any(|c| case_sensitivity.eq(c.as_bytes(), name.0.as_bytes()))
    }

    fn has_custom_state(&self, _name: &CssLocalName) -> bool {
        false
    }

    fn imported_part(&self, _name: &CssLocalName) -> Option<CssLocalName> {
        None
    }

    fn is_part(&self, _name: &CssLocalName) -> bool {
        false
    }

    /// css-select: 요소나 빈 글이 아닌 글이 없으면 (주석은 셈하지 않는다)
    fn is_empty(&self) -> bool {
        !self.doc.nodes[self.id].children.iter().any(|c| match &self.doc.nodes[*c].data {
            Data::Element(_) => true,
            Data::Text { text, .. } => !text.is_empty(),
            _ => false,
        })
    }

    fn is_root(&self) -> bool {
        self.doc.nodes[self.id].parent.is_some_and(|p| matches!(self.doc.nodes[p].data, Data::Document))
    }

    fn add_element_unique_hashes(&self, _filter: &mut BloomFilter) -> bool {
        false
    }
}

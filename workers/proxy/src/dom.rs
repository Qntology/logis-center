//! (옮기는 중) linkedom(htmlparser2 10.1.0) 과 같은 트리 + CSS 선택자

use logis_worker_common::R;

/// 노드 번호
pub type NodeId = usize;

/// `parseHTML(html).document`
pub struct Document;

impl Document {
    pub fn parse(_html: &str) -> Document {
        Document
    }

    /// `document.querySelectorAll(selector)` — 잘못된 선택자는 SyntaxError
    pub fn query_selector_all(&self, _selector: &str) -> R<Vec<NodeId>> {
        Ok(Vec::new())
    }

    /// `document.querySelector(selector)`
    pub fn query_selector(&self, selector: &str) -> R<Option<NodeId>> {
        Ok(self.query_selector_all(selector)?.into_iter().next())
    }

    /// `node.textContent`
    pub fn text_content(&self, _id: NodeId) -> String {
        String::new()
    }

    /// `node.getAttribute(name)`
    pub fn get_attribute(&self, _id: NodeId, _name: &str) -> Option<String> {
        None
    }
}

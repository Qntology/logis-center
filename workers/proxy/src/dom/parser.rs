//! htmlparser2 10.1.0 `Tokenizer` · `Parser`(HTML 모드) 와 linkedom 0.18.11 `parseFromString` 의 콜백을 한 기계로 옮긴 것.
//!
//! 옵션은 linkedom 이 주는 그대로: `{ lowerCaseAttributeNames: false, decodeEntities: true, xmlMode: false }`
//! (태그 이름은 소문자로, 속성 이름은 그대로, `recognizeSelfClosing` 꺼짐, CDATA 는 주석).
//! 글은 원본처럼 여러 `ontext` 로 갈라져 나오고(엔터티 앞뒤 · `<` 앞 · 버퍼 끝), linkedom 은 콜백마다 Text 노드를 하나씩 붙인다.
//!
//! 위치는 UTF-8 바이트다. 원본은 UTF-16 단위로 세지만, 상태를 바꾸는 문자는 모두 ASCII 이고 ASCII 가 아닌 바이트는
//! 어떤 비교에도 걸리지 않아(`c | 0x20` 도 0x80 이상으로 남는다) 같은 곳에서 같은 일이 일어난다.

use super::entities::{DecodingMode, EntityDecoder};
use super::{Data, Element, Node, NodeId};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Text,
    BeforeTagName,
    InTagName,
    InSelfClosingTag,
    BeforeClosingTagName,
    InClosingTagName,
    AfterClosingTagName,
    BeforeAttributeName,
    InAttributeName,
    AfterAttributeName,
    BeforeAttributeValue,
    InAttributeValueDq,
    InAttributeValueSq,
    InAttributeValueNq,
    BeforeDeclaration,
    InDeclaration,
    InProcessingInstruction,
    BeforeComment,
    CdataSequence,
    InSpecialComment,
    InCommentLike,
    BeforeSpecialS,
    BeforeSpecialT,
    SpecialStartSequence,
    InSpecialTag,
    InEntity,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Seq {
    Cdata,
    CdataEnd,
    CommentEnd,
    ScriptEnd,
    StyleEnd,
    TitleEnd,
    TextareaEnd,
    XmpEnd,
}

impl Seq {
    fn bytes(self) -> &'static [u8] {
        match self {
            Seq::Cdata => b"CDATA[",
            Seq::CdataEnd => b"]]>",
            Seq::CommentEnd => b"-->",
            Seq::ScriptEnd => b"</script",
            Seq::StyleEnd => b"</style",
            Seq::TitleEnd => b"</title",
            Seq::TextareaEnd => b"</textarea",
            Seq::XmpEnd => b"</xmp",
        }
    }
}

fn is_ws(c: u8) -> bool {
    matches!(c, b' ' | b'\n' | b'\t' | 0x0C | b'\r')
}

fn is_end_of_tag_section(c: u8) -> bool {
    c == b'/' || c == b'>' || is_ws(c)
}

const FORM_TAGS: &[&str] = &["input", "option", "optgroup", "select", "button", "datalist", "textarea"];
const P_TAG: &[&str] = &["p"];

/// `openImpliesClose`
fn open_implies_close(name: &str) -> Option<&'static [&'static str]> {
    Some(match name {
        "tr" => &["tr", "th", "td"],
        "th" => &["th"],
        "td" => &["thead", "th", "td"],
        "body" => &["head", "link", "script"],
        "li" => &["li"],
        "p" | "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => P_TAG,
        "select" | "input" | "output" | "button" | "datalist" | "textarea" => FORM_TAGS,
        "option" => &["option"],
        "optgroup" => &["optgroup", "option"],
        "dd" | "dt" => &["dd", "dt"],
        "address" | "article" | "aside" | "blockquote" | "details" | "div" | "dl" | "fieldset" | "figcaption" | "figure"
        | "footer" | "form" | "header" | "hr" | "main" | "nav" | "ol" | "pre" | "section" | "table" | "ul" => P_TAG,
        "rt" | "rp" => &["rt", "rp"],
        "tbody" | "tfoot" => &["thead", "tbody"],
        _ => return None,
    })
}

fn is_void(name: &str) -> bool {
    matches!(
        name,
        "area" | "base" | "basefont" | "br" | "col" | "command" | "embed" | "frame" | "hr" | "img" | "input" | "isindex"
            | "keygen" | "link" | "meta" | "param" | "source" | "track" | "wbr"
    )
}

fn is_foreign_context(name: &str) -> bool {
    name == "math" || name == "svg"
}

fn is_html_integration(name: &str) -> bool {
    matches!(name, "mi" | "mo" | "mn" | "ms" | "mtext" | "annotation-xml" | "foreignobject" | "desc" | "title")
}

/// JS `/\s/` (WhiteSpace + LineTerminator)
fn is_js_space(c: char) -> bool {
    logis_worker_common::jsv::is_js_space(c)
}

/// JS 객체 키 순서: 배열 첨자 꼴 키가 오름차순으로 먼저 (htmlparser2 의 attribs 는 평범한 객체)
fn js_key_order(attrs: Vec<(String, String)>) -> Vec<(String, String)> {
    let index = |k: &str| logis_worker_common::jsv::array_index(k);
    let mut idx: Vec<(u32, (String, String))> = Vec::new();
    let mut rest = Vec::new();
    for a in attrs {
        match index(&a.0) {
            Some(i) => idx.push((i, a)),
            None => rest.push(a),
        }
    }
    if idx.is_empty() {
        return rest;
    }
    idx.sort_by_key(|(i, _)| *i);
    idx.into_iter().map(|(_, a)| a).chain(rest).collect()
}

/// 코드 포인트(또는 이름 엔터티가 내는 서로게이트 한 단위)를 문자열에 붙인다. 높은 서로게이트는 다음 낮은 것과 합친다
fn push_unit(out: &mut String, pending: &mut Option<u16>, cp: u32) {
    if (0xD800..=0xDBFF).contains(&cp) {
        if pending.is_some() {
            out.push('\u{FFFD}');
        }
        *pending = Some(cp as u16);
        return;
    }
    if (0xDC00..=0xDFFF).contains(&cp) {
        if let Some(hi) = pending.take() {
            let c = 0x10000 + ((hi as u32 - 0xD800) << 10) + (cp - 0xDC00);
            out.push(char::from_u32(c).unwrap_or('\u{FFFD}'));
        } else {
            out.push('\u{FFFD}');
        }
        return;
    }
    if pending.take().is_some() {
        out.push('\u{FFFD}');
    }
    out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
}

/// linkedom 쪽 트리 쌓기 (`parseFromString` 의 콜백)
struct Builder {
    nodes: Vec<Node>,
    /// `node` (현재 부모). 문서 위로 닫히면 None
    cur: Option<NodeId>,
    /// `ownerSVGElement`
    owner_svg: Option<NodeId>,
}

impl Builder {
    fn append(&mut self, data: Data) -> Option<NodeId> {
        let parent = self.cur?;
        let id = self.nodes.len();
        let index = self.nodes[parent].children.len();
        self.nodes.push(Node { parent: Some(parent), index, children: Vec::new(), data });
        self.nodes[parent].children.push(id);
        Some(id)
    }

    /// `onopentag(name, attributes)`
    fn open(&mut self, name: &str, attrs: Vec<(String, String)>) {
        let svg = self.owner_svg.is_some() || name == "svg" || name == "SVG";
        let mut attrs = js_key_order(attrs);
        // `element.className = value` — classList 로 갈라 중복을 빼고 class 속성값을 다시 이어 쓴다
        let mut class_list: Vec<String> = Vec::new();
        if let Some((_, value)) = attrs.iter_mut().find(|(n, _)| n == "class") {
            for token in value.split(is_js_space) {
                if !token.is_empty() && !class_list.iter().any(|t| t == token) {
                    class_list.push(token.to_string());
                }
            }
            *value = class_list.join(" ");
        }
        let element = Element { name: name.to_string(), attrs, class_list, svg };
        let Some(id) = self.append(Data::Element(element)) else { return };
        if svg && self.owner_svg.is_none() {
            self.owner_svg = Some(id);
        }
        self.cur = Some(id);
    }

    /// `onclosetag()` — 이름은 보지 않고 한 단계 위로
    fn close(&mut self) {
        let Some(cur) = self.cur else { return };
        if self.owner_svg == Some(cur) {
            self.owner_svg = None;
        }
        self.cur = self.nodes[cur].parent;
    }

    fn text(&mut self, text: String, unit: Option<u16>) {
        self.append(Data::Text { text, unit });
    }

    fn comment(&mut self, text: String) {
        self.append(Data::Comment(text));
    }
}

struct Machine<'a> {
    buf: &'a [u8],
    len: i64,
    // Tokenizer
    state: State,
    section_start: i64,
    index: i64,
    entity_start: i64,
    base_state: State,
    is_special: bool,
    seq: Seq,
    sequence_index: usize,
    decoder: EntityDecoder,
    // Parser (stack · foreignContext 는 원본의 [0] 이 Vec 의 끝)
    tagname: String,
    attribname: String,
    attribvalue: String,
    attrib_pending: Option<u16>,
    attribs: Option<Vec<(String, String)>>,
    stack: Vec<String>,
    foreign: Vec<bool>,
    b: Builder,
}

pub(super) fn parse(html: &str) -> Vec<Node> {
    let buf = html.as_bytes();
    let mut m = Machine {
        buf,
        len: buf.len() as i64,
        state: State::Text,
        section_start: 0,
        index: 0,
        entity_start: 0,
        base_state: State::Text,
        is_special: false,
        seq: Seq::CommentEnd,
        sequence_index: 0,
        decoder: EntityDecoder::new(),
        tagname: String::new(),
        attribname: String::new(),
        attribvalue: String::new(),
        attrib_pending: None,
        attribs: None,
        stack: Vec::new(),
        foreign: vec![false],
        b: Builder {
            nodes: vec![Node { parent: None, index: 0, children: Vec::new(), data: Data::Document }],
            cur: Some(0),
            owner_svg: None,
        },
    };
    m.run();
    m.b.nodes
}

impl Machine<'_> {
    fn at(&self, i: i64) -> u8 {
        self.buf[i as usize]
    }

    /// `getSlice(start, end)` — JS `String#slice` 처럼 음수는 끝에서, 거꾸로면 빈 글
    fn slice(&self, start: i64, end: i64) -> String {
        let norm = |x: i64| if x < 0 { (self.len + x).max(0) } else { x.min(self.len) };
        let (a, b) = (norm(start), norm(end));
        if b <= a {
            return String::new();
        }
        let bytes = &self.buf[a as usize..b as usize];
        match std::str::from_utf8(bytes) {
            Ok(s) => s.to_string(),
            Err(_) => String::from_utf8_lossy(bytes).into_owned(),
        }
    }

    fn run(&mut self) {
        // write(chunk) → parse()
        while self.index < self.len {
            let c = self.at(self.index);
            match self.state {
                State::Text => self.state_text(c),
                State::SpecialStartSequence => self.state_special_start_sequence(c),
                State::InSpecialTag => self.state_in_special_tag(c),
                State::CdataSequence => self.state_cdata_sequence(c),
                State::InAttributeValueDq => self.handle_in_attribute_value(c, b'"'),
                State::InAttributeName => self.state_in_attribute_name(c),
                State::InCommentLike => self.state_in_comment_like(c),
                State::InSpecialComment => self.state_in_special_comment(c),
                State::BeforeAttributeName => self.state_before_attribute_name(c),
                State::InTagName => self.state_in_tag_name(c),
                State::InClosingTagName => self.state_in_closing_tag_name(c),
                State::BeforeTagName => self.state_before_tag_name(c),
                State::AfterAttributeName => self.state_after_attribute_name(c),
                State::InAttributeValueSq => self.handle_in_attribute_value(c, b'\''),
                State::BeforeAttributeValue => self.state_before_attribute_value(c),
                State::BeforeClosingTagName => self.state_before_closing_tag_name(c),
                State::AfterClosingTagName => self.state_after_closing_tag_name(c),
                State::BeforeSpecialS => self.state_before_special_s(c),
                State::BeforeSpecialT => self.state_before_special_t(c),
                State::InAttributeValueNq => self.state_in_attribute_value_no_quotes(c),
                State::InSelfClosingTag => self.state_in_self_closing_tag(c),
                State::InDeclaration => self.state_in_declaration(c),
                State::BeforeDeclaration => self.state_before_declaration(c),
                State::BeforeComment => self.state_before_comment(c),
                State::InProcessingInstruction => self.state_in_processing_instruction(c),
                State::InEntity => self.state_in_entity(),
            }
            self.index += 1;
        }
        self.cleanup();
        // end() → finish()
        if self.state == State::InEntity {
            self.decoder.end();
            self.flush_entity_output();
            self.state = self.base_state;
        }
        self.handle_trailing_data();
        self.p_onend();
    }

    // ───────────── Tokenizer ─────────────

    fn state_text(&mut self, c: u8) {
        if c == b'<' {
            if self.index > self.section_start {
                self.p_ontext(self.section_start, self.index);
            }
            self.state = State::BeforeTagName;
            self.section_start = self.index;
        } else if c == b'&' {
            self.start_entity();
        }
    }

    fn state_special_start_sequence(&mut self, c: u8) {
        let seq = self.seq.bytes();
        let is_end = self.sequence_index == seq.len();
        let is_match = if is_end { is_end_of_tag_section(c) } else { (c | 0x20) == seq[self.sequence_index] };
        if !is_match {
            self.is_special = false;
        } else if !is_end {
            self.sequence_index += 1;
            return;
        }
        self.sequence_index = 0;
        self.state = State::InTagName;
        self.state_in_tag_name(c);
    }

    /// 끝 태그를 찾는다. `<title>` 은 엔터티도 푼다
    fn state_in_special_tag(&mut self, c: u8) {
        let seq = self.seq.bytes();
        if self.sequence_index == seq.len() {
            if c == b'>' || is_ws(c) {
                let end_of_text = self.index - seq.len() as i64;
                if self.section_start < end_of_text {
                    self.p_ontext(self.section_start, end_of_text);
                }
                self.is_special = false;
                self.section_start = end_of_text + 2;
                self.state_in_closing_tag_name(c);
                return;
            }
            self.sequence_index = 0;
        }
        if (c | 0x20) == seq[self.sequence_index] {
            self.sequence_index += 1;
        } else if self.sequence_index == 0 {
            if self.seq == Seq::TitleEnd {
                if c == b'&' {
                    self.start_entity();
                }
            } else if self.fast_forward_to(b'<') {
                self.sequence_index = 1;
            }
        } else {
            self.sequence_index = (c == b'<') as usize;
        }
    }

    fn state_cdata_sequence(&mut self, c: u8) {
        let seq = Seq::Cdata.bytes();
        if c == seq[self.sequence_index] {
            self.sequence_index += 1;
            if self.sequence_index == seq.len() {
                self.state = State::InCommentLike;
                self.seq = Seq::CdataEnd;
                self.sequence_index = 0;
                self.section_start = self.index + 1;
            }
        } else {
            self.sequence_index = 0;
            self.state = State::InDeclaration;
            self.state_in_declaration(c);
        }
    }

    fn fast_forward_to(&mut self, c: u8) -> bool {
        loop {
            self.index += 1;
            if self.index >= self.len {
                break;
            }
            if self.at(self.index) == c {
                return true;
            }
        }
        self.index = self.len - 1;
        false
    }

    /// 주석 · CDATA 는 `-->` · `]]>` 로 끝난다
    fn state_in_comment_like(&mut self, c: u8) {
        let seq = self.seq.bytes();
        if c == seq[self.sequence_index] {
            self.sequence_index += 1;
            if self.sequence_index == seq.len() {
                if self.seq == Seq::CdataEnd {
                    self.p_oncdata(self.section_start, self.index, 2);
                } else {
                    self.p_oncomment(self.section_start, self.index, 2);
                }
                self.sequence_index = 0;
                self.section_start = self.index + 1;
                self.state = State::Text;
            }
        } else if self.sequence_index == 0 {
            if self.fast_forward_to(seq[0]) {
                self.sequence_index = 1;
            }
        } else if c != seq[self.sequence_index - 1] {
            self.sequence_index = 0;
        }
    }

    fn start_special(&mut self, seq: Seq, offset: usize) {
        self.is_special = true;
        self.seq = seq;
        self.sequence_index = offset;
        self.state = State::SpecialStartSequence;
    }

    fn state_before_tag_name(&mut self, c: u8) {
        if c == b'!' {
            self.state = State::BeforeDeclaration;
            self.section_start = self.index + 1;
        } else if c == b'?' {
            self.state = State::InProcessingInstruction;
            self.section_start = self.index + 1;
        } else if c.is_ascii_alphabetic() {
            let lower = c | 0x20;
            self.section_start = self.index;
            self.state = if lower == b's' {
                State::BeforeSpecialS
            } else if lower == b't' || lower == b'x' {
                State::BeforeSpecialT
            } else {
                State::InTagName
            };
        } else if c == b'/' {
            self.state = State::BeforeClosingTagName;
        } else {
            self.state = State::Text;
            self.state_text(c);
        }
    }

    fn state_in_tag_name(&mut self, c: u8) {
        if is_end_of_tag_section(c) {
            self.p_onopentagname(self.section_start, self.index);
            self.section_start = -1;
            self.state = State::BeforeAttributeName;
            self.state_before_attribute_name(c);
        }
    }

    fn state_before_closing_tag_name(&mut self, c: u8) {
        if is_ws(c) {
            // 무시
        } else if c == b'>' {
            self.state = State::Text;
        } else {
            self.state = if c.is_ascii_alphabetic() { State::InClosingTagName } else { State::InSpecialComment };
            self.section_start = self.index;
        }
    }

    fn state_in_closing_tag_name(&mut self, c: u8) {
        if c == b'>' || is_ws(c) {
            self.p_onclosetag(self.section_start, self.index);
            self.section_start = -1;
            self.state = State::AfterClosingTagName;
            self.state_after_closing_tag_name(c);
        }
    }

    fn state_after_closing_tag_name(&mut self, c: u8) {
        if c == b'>' || self.fast_forward_to(b'>') {
            self.state = State::Text;
            self.section_start = self.index + 1;
        }
    }

    fn state_before_attribute_name(&mut self, c: u8) {
        if c == b'>' {
            self.p_onopentagend();
            if self.is_special {
                self.state = State::InSpecialTag;
                self.sequence_index = 0;
            } else {
                self.state = State::Text;
            }
            self.section_start = self.index + 1;
        } else if c == b'/' {
            self.state = State::InSelfClosingTag;
        } else if !is_ws(c) {
            self.state = State::InAttributeName;
            self.section_start = self.index;
        }
    }

    fn state_in_self_closing_tag(&mut self, c: u8) {
        if c == b'>' {
            self.p_onselfclosingtag();
            self.state = State::Text;
            self.section_start = self.index + 1;
            self.is_special = false;
        } else if !is_ws(c) {
            self.state = State::BeforeAttributeName;
            self.state_before_attribute_name(c);
        }
    }

    fn state_in_attribute_name(&mut self, c: u8) {
        if c == b'=' || is_end_of_tag_section(c) {
            self.p_onattribname(self.section_start, self.index);
            self.section_start = self.index;
            self.state = State::AfterAttributeName;
            self.state_after_attribute_name(c);
        }
    }

    fn state_after_attribute_name(&mut self, c: u8) {
        if c == b'=' {
            self.state = State::BeforeAttributeValue;
        } else if c == b'/' || c == b'>' {
            self.p_onattribend();
            self.section_start = -1;
            self.state = State::BeforeAttributeName;
            self.state_before_attribute_name(c);
        } else if !is_ws(c) {
            self.p_onattribend();
            self.state = State::InAttributeName;
            self.section_start = self.index;
        }
    }

    fn state_before_attribute_value(&mut self, c: u8) {
        if c == b'"' {
            self.state = State::InAttributeValueDq;
            self.section_start = self.index + 1;
        } else if c == b'\'' {
            self.state = State::InAttributeValueSq;
            self.section_start = self.index + 1;
        } else if !is_ws(c) {
            self.section_start = self.index;
            self.state = State::InAttributeValueNq;
            self.state_in_attribute_value_no_quotes(c);
        }
    }

    fn handle_in_attribute_value(&mut self, c: u8, quote: u8) {
        if c == quote {
            self.p_onattribdata(self.section_start, self.index);
            self.section_start = -1;
            self.p_onattribend();
            self.state = State::BeforeAttributeName;
        } else if c == b'&' {
            self.start_entity();
        }
    }

    fn state_in_attribute_value_no_quotes(&mut self, c: u8) {
        if is_ws(c) || c == b'>' {
            self.p_onattribdata(self.section_start, self.index);
            self.section_start = -1;
            self.p_onattribend();
            self.state = State::BeforeAttributeName;
            self.state_before_attribute_name(c);
        } else if c == b'&' {
            self.start_entity();
        }
    }

    fn state_before_declaration(&mut self, c: u8) {
        if c == b'[' {
            self.state = State::CdataSequence;
            self.sequence_index = 0;
        } else {
            self.state = if c == b'-' { State::BeforeComment } else { State::InDeclaration };
        }
    }

    /// `<!DOCTYPE …>` 등 — linkedom 은 document.doctype 만 고친다(트리 노드 없음)
    fn state_in_declaration(&mut self, c: u8) {
        if c == b'>' || self.fast_forward_to(b'>') {
            self.state = State::Text;
            self.section_start = self.index + 1;
        }
    }

    /// `<?…>` — 노드 없음
    fn state_in_processing_instruction(&mut self, c: u8) {
        if c == b'>' || self.fast_forward_to(b'>') {
            self.state = State::Text;
            self.section_start = self.index + 1;
        }
    }

    fn state_before_comment(&mut self, c: u8) {
        if c == b'-' {
            self.state = State::InCommentLike;
            self.seq = Seq::CommentEnd;
            // 짧은 주석(`<!-->`)도 받는다
            self.sequence_index = 2;
            self.section_start = self.index + 1;
        } else {
            self.state = State::InDeclaration;
        }
    }

    fn state_in_special_comment(&mut self, c: u8) {
        if c == b'>' || self.fast_forward_to(b'>') {
            self.p_oncomment(self.section_start, self.index, 0);
            self.state = State::Text;
            self.section_start = self.index + 1;
        }
    }

    fn state_before_special_s(&mut self, c: u8) {
        let lower = c | 0x20;
        if lower == Seq::ScriptEnd.bytes()[3] {
            self.start_special(Seq::ScriptEnd, 4);
        } else if lower == Seq::StyleEnd.bytes()[3] {
            self.start_special(Seq::StyleEnd, 4);
        } else {
            self.state = State::InTagName;
            self.state_in_tag_name(c);
        }
    }

    fn state_before_special_t(&mut self, c: u8) {
        let lower = c | 0x20;
        if lower == Seq::TitleEnd.bytes()[3] {
            self.start_special(Seq::TitleEnd, 4);
        } else if lower == Seq::TextareaEnd.bytes()[3] {
            self.start_special(Seq::TextareaEnd, 4);
        } else if lower == Seq::XmpEnd.bytes()[3] {
            self.start_special(Seq::XmpEnd, 4);
        } else {
            self.state = State::InTagName;
            self.state_in_tag_name(c);
        }
    }

    fn start_entity(&mut self) {
        self.base_state = self.state;
        self.state = State::InEntity;
        self.entity_start = self.index;
        let mode = if self.base_state == State::Text || self.base_state == State::InSpecialTag {
            DecodingMode::Legacy
        } else {
            DecodingMode::Attribute
        };
        self.decoder.start_entity(mode);
    }

    fn state_in_entity(&mut self) {
        let length = self.decoder.write(self.buf, self.index as usize);
        self.flush_entity_output();
        if length >= 0 {
            self.state = self.base_state;
            if length == 0 {
                self.index -= 1;
            }
        } else {
            if self.index < self.len && self.at(self.index) == b'&' {
                self.state = self.base_state;
                self.index -= 1;
                return;
            }
            self.index = self.len - 1;
        }
    }

    /// `emitCodePoint(cp, consumed)` — 디코더가 내보낸 순서대로
    fn flush_entity_output(&mut self) {
        let out = std::mem::take(&mut self.decoder.out);
        for (cp, consumed) in out {
            let in_text = self.base_state == State::Text || self.base_state == State::InSpecialTag;
            if self.section_start < self.entity_start {
                if in_text {
                    self.p_ontext(self.section_start, self.entity_start);
                } else {
                    self.p_onattribdata(self.section_start, self.entity_start);
                }
            }
            self.section_start = self.entity_start + consumed as i64;
            self.index = self.section_start - 1;
            if in_text {
                self.p_ontextentity(cp);
            } else {
                self.p_onattribentity(cp);
            }
        }
    }

    /// 버퍼 끝: 글 · 속성값이면 지금까지를 내보낸다
    fn cleanup(&mut self) {
        if self.section_start != self.index {
            if self.state == State::Text || (self.state == State::InSpecialTag && self.sequence_index == 0) {
                self.p_ontext(self.section_start, self.index);
                self.section_start = self.index;
            } else if matches!(self.state, State::InAttributeValueDq | State::InAttributeValueSq | State::InAttributeValueNq) {
                self.p_onattribdata(self.section_start, self.index);
                self.section_start = self.index;
            }
        }
    }

    fn handle_trailing_data(&mut self) {
        let end_index = self.len;
        if self.section_start >= end_index {
            return;
        }
        match self.state {
            State::InCommentLike => {
                if self.seq == Seq::CdataEnd {
                    self.p_oncdata(self.section_start, end_index, 0);
                } else {
                    self.p_oncomment(self.section_start, end_index, 0);
                }
            }
            // 열거나 닫는 태그 도중이면 그 태그는 버린다
            State::InTagName
            | State::BeforeAttributeName
            | State::BeforeAttributeValue
            | State::AfterAttributeName
            | State::InAttributeName
            | State::InAttributeValueSq
            | State::InAttributeValueDq
            | State::InAttributeValueNq
            | State::InClosingTagName => {}
            _ => self.p_ontext(self.section_start, end_index),
        }
    }

    // ───────────── Parser ─────────────

    fn p_ontext(&mut self, start: i64, end: i64) {
        let data = self.slice(start, end);
        self.b.text(data, None);
    }

    /// 엔터티 하나 = Text 노드 하나. 이름 엔터티의 BMP 밖 값은 서로게이트 단위마다 노드가 갈린다(JS 에선 외톨이 서로게이트)
    fn p_ontextentity(&mut self, cp: u32) {
        if (0xD800..=0xDFFF).contains(&cp) {
            self.b.text("\u{FFFD}".to_string(), Some(cp as u16));
        } else {
            self.b.text(char::from_u32(cp).unwrap_or('\u{FFFD}').to_string(), None);
        }
    }

    fn p_onopentagname(&mut self, start: i64, end: i64) {
        let name = self.slice(start, end).to_lowercase();
        self.emit_open_tag(name);
    }

    fn emit_open_tag(&mut self, name: String) {
        if let Some(closes) = open_implies_close(&name) {
            while let Some(top) = self.stack.last() {
                if !closes.contains(&top.as_str()) {
                    break;
                }
                self.stack.pop();
                self.b.close();
            }
        }
        if !is_void(&name) {
            self.stack.push(name.clone());
            if is_foreign_context(&name) {
                self.foreign.push(true);
            } else if is_html_integration(&name) {
                self.foreign.push(false);
            }
        }
        self.tagname = name;
        self.attribs = Some(Vec::new());
    }

    fn end_open_tag(&mut self) {
        if let Some(attrs) = self.attribs.take() {
            let name = self.tagname.clone();
            self.b.open(&name, attrs);
        }
        if is_void(&self.tagname) {
            self.b.close();
        }
        self.tagname.clear();
    }

    fn p_onopentagend(&mut self) {
        self.end_open_tag();
    }

    fn p_onclosetag(&mut self, start: i64, end: i64) {
        let name = self.slice(start, end).to_lowercase();
        if is_foreign_context(&name) || is_html_integration(&name) {
            self.foreign.pop();
        }
        if !is_void(&name) {
            if let Some(pos) = self.stack.iter().rev().position(|n| *n == name) {
                for _ in 0..=pos {
                    self.stack.pop();
                    self.b.close();
                }
            } else if name == "p" {
                // 여는 태그 없는 </p> 는 빈 <p></p>
                self.emit_open_tag("p".to_string());
                self.close_current_tag();
            }
        } else if name == "br" {
            // </br> 는 <br>
            self.b.open("br", Vec::new());
            self.b.close();
        }
    }

    fn p_onselfclosingtag(&mut self) {
        if self.foreign.last().copied().unwrap_or(false) {
            self.close_current_tag();
        } else {
            // HTML 에선 `/>` 를 무시한다
            self.p_onopentagend();
        }
    }

    fn close_current_tag(&mut self) {
        let name = self.tagname.clone();
        self.end_open_tag();
        if self.stack.last() == Some(&name) {
            self.b.close();
            self.stack.pop();
        }
    }

    fn p_onattribname(&mut self, start: i64, end: i64) {
        self.attribname = self.slice(start, end);
    }

    fn p_onattribdata(&mut self, start: i64, end: i64) {
        let s = self.slice(start, end);
        if self.attrib_pending.take().is_some() {
            self.attribvalue.push('\u{FFFD}');
        }
        self.attribvalue.push_str(&s);
    }

    fn p_onattribentity(&mut self, cp: u32) {
        push_unit(&mut self.attribvalue, &mut self.attrib_pending, cp);
    }

    fn p_onattribend(&mut self) {
        if self.attrib_pending.take().is_some() {
            self.attribvalue.push('\u{FFFD}');
        }
        let value = std::mem::take(&mut self.attribvalue);
        if let Some(attrs) = &mut self.attribs {
            // 먼저 나온 이름이 이긴다. `__proto__` 는 평범한 객체에 문자열로 넣으면 사라진다
            if self.attribname != "__proto__" && !attrs.iter().any(|(n, _)| *n == self.attribname) {
                attrs.push((self.attribname.clone(), value));
            }
        }
    }

    fn p_oncomment(&mut self, start: i64, end: i64, offset: i64) {
        let data = self.slice(start, end - offset);
        self.b.comment(data);
    }

    /// HTML 모드의 CDATA 는 주석 `[CDATA[…]]`
    fn p_oncdata(&mut self, start: i64, end: i64, offset: i64) {
        let data = self.slice(start, end - offset);
        self.b.comment(format!("[CDATA[{data}]]"));
    }

    /// 남은 태그를 모두 닫는다
    fn p_onend(&mut self) {
        for _ in 0..self.stack.len() {
            self.b.close();
        }
    }
}

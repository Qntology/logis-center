//! entities 7.0.1 `EntityDecoder` (dist/esm/decode.js) 를 그대로 옮긴 것 — htmlparser2 Tokenizer 가 쓰는 부분만.
//!
//! 입력은 바이트(UTF-8)다. 엔터티 이름 · 숫자는 모두 ASCII 라 UTF-16 단위로 센 원본과 `consumed` 가 같다.
//! `emitCodePoint` 콜백 대신 내보낸 (코드 포인트, consumed) 를 `out` 에 쌓고, 부른 쪽이 `write` · `end` 직후에 처리한다
//! (원본도 값을 내보낸 뒤 곧바로 돌아온다).

use super::entity_tree::HTML_DECODE_TREE;

const VALUE_LENGTH: u16 = 49152;
const FLAG13: u16 = 8192;
const BRANCH_LENGTH: u16 = 8064;
const JUMP_TABLE: u16 = 127;

const NUM: u8 = b'#';
const SEMI: u8 = b';';
const TO_LOWER_BIT: u8 = 32;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum DecodingMode {
    /// 글: `;` 없는 옛 엔터티도 받음
    Legacy,
    /// 속성값: `;` 없는 엔터티 뒤에 `=` · 영숫자가 오면 풀지 않음
    Attribute,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum St {
    EntityStart,
    NumericStart,
    NumericDecimal,
    NumericHex,
    NamedEntity,
}

/// `isEntityInAttributeInvalidEnd`
fn attr_invalid_end(c: u8) -> bool {
    c == b'=' || c.is_ascii_alphanumeric()
}

/// `replaceCodePoint` (서로게이트 · 범위 밖은 U+FFFD, C1 제어 문자는 windows-1252 표)
fn replace_code_point(cp: i64) -> u32 {
    if (0xD800..=0xDFFF).contains(&cp) || cp > 0x10FFFF {
        return 0xFFFD;
    }
    match cp {
        0 => 65533,
        128 => 8364,
        130 => 8218,
        131 => 402,
        132 => 8222,
        133 => 8230,
        134 => 8224,
        135 => 8225,
        136 => 710,
        137 => 8240,
        138 => 352,
        139 => 8249,
        140 => 338,
        142 => 381,
        145 => 8216,
        146 => 8217,
        147 => 8220,
        148 => 8221,
        149 => 8226,
        150 => 8211,
        151 => 8212,
        152 => 732,
        153 => 8482,
        154 => 353,
        155 => 8250,
        156 => 339,
        158 => 382,
        159 => 376,
        _ => cp as u32,
    }
}

fn tree(i: i64) -> u16 {
    HTML_DECODE_TREE.get(i as usize).copied().unwrap_or(0)
}

/// `determineBranch`
fn determine_branch(current: u16, node_index: i64, c: u8) -> i64 {
    let branch_count = ((current & BRANCH_LENGTH) >> 7) as i64;
    let jump_offset = (current & JUMP_TABLE) as i64;
    let c = c as i64;
    if branch_count == 0 {
        return if jump_offset != 0 && c == jump_offset { node_index } else { -1 };
    }
    if jump_offset != 0 {
        let value = c - jump_offset;
        return if value < 0 || value >= branch_count { -1 } else { tree(node_index + value) as i64 - 1 };
    }
    let packed_key_slots = (branch_count + 1) >> 1;
    let (mut lo, mut hi) = (0i64, branch_count - 1);
    while lo <= hi {
        let mid = (lo + hi) >> 1;
        let packed = tree(node_index + (mid >> 1)) as i64;
        let mid_key = (packed >> ((mid & 1) * 8)) & 0xff;
        if mid_key < c {
            lo = mid + 1;
        } else if mid_key > c {
            hi = mid - 1;
        } else {
            return tree(node_index + packed_key_slots + mid) as i64;
        }
    }
    -1
}

pub(super) struct EntityDecoder {
    state: St,
    consumed: i64,
    result: i64,
    tree_index: i64,
    excess: i64,
    mode: DecodingMode,
    run_consumed: i64,
    /// 내보낸 (코드 포인트 — 이름 엔터티의 BMP 밖 값은 서로게이트 단위 하나씩, consumed)
    pub(super) out: Vec<(u32, usize)>,
}

impl EntityDecoder {
    pub(super) fn new() -> Self {
        EntityDecoder {
            state: St::EntityStart,
            consumed: 1,
            result: 0,
            tree_index: 0,
            excess: 1,
            mode: DecodingMode::Legacy,
            run_consumed: 0,
            out: Vec::new(),
        }
    }

    pub(super) fn start_entity(&mut self, mode: DecodingMode) {
        self.mode = mode;
        self.state = St::EntityStart;
        self.result = 0;
        self.tree_index = 0;
        self.excess = 1;
        self.consumed = 1;
        self.run_consumed = 0;
    }

    fn emit(&mut self, cp: u32, consumed: i64) {
        self.out.push((cp, consumed as usize));
    }

    /// 끝나면 consumed(`&` 포함), 엔터티가 아니면 0, 입력이 모자라면 -1
    pub(super) fn write(&mut self, input: &[u8], offset: usize) -> i64 {
        match self.state {
            St::EntityStart => {
                if input.get(offset) == Some(&NUM) {
                    self.state = St::NumericStart;
                    self.consumed += 1;
                    return self.numeric_start(input, offset + 1);
                }
                self.state = St::NamedEntity;
                self.named_entity(input, offset)
            }
            St::NumericStart => self.numeric_start(input, offset),
            St::NumericDecimal => self.numeric_decimal(input, offset),
            St::NumericHex => self.numeric_hex(input, offset),
            St::NamedEntity => self.named_entity(input, offset),
        }
    }

    fn numeric_start(&mut self, input: &[u8], offset: usize) -> i64 {
        if offset >= input.len() {
            return -1;
        }
        if (input[offset] | TO_LOWER_BIT) == b'x' {
            self.state = St::NumericHex;
            self.consumed += 1;
            return self.numeric_hex(input, offset + 1);
        }
        self.state = St::NumericDecimal;
        self.numeric_decimal(input, offset)
    }

    fn numeric_hex(&mut self, input: &[u8], mut offset: usize) -> i64 {
        while offset < input.len() {
            let c = input[offset];
            if c.is_ascii_hexdigit() {
                let digit = if c <= b'9' { (c - b'0') as i64 } else { ((c | TO_LOWER_BIT) - b'a' + 10) as i64 };
                // 원본은 f64 라 끝없이 커진다 — 0x10FFFF 를 넘으면 어차피 U+FFFD
                self.result = self.result.saturating_mul(16).saturating_add(digit);
                self.consumed += 1;
                offset += 1;
            } else {
                return self.emit_numeric(c as i64, 3);
            }
        }
        -1
    }

    fn numeric_decimal(&mut self, input: &[u8], mut offset: usize) -> i64 {
        while offset < input.len() {
            let c = input[offset];
            if c.is_ascii_digit() {
                self.result = self.result.saturating_mul(10).saturating_add((c - b'0') as i64);
                self.consumed += 1;
                offset += 1;
            } else {
                return self.emit_numeric(c as i64, 2);
            }
        }
        -1
    }

    /// `emitNumericEntity` (last: 마지막 문자, 입력 끝이면 0)
    fn emit_numeric(&mut self, last: i64, expected_length: i64) -> i64 {
        if self.consumed <= expected_length {
            return 0;
        }
        if last == SEMI as i64 {
            self.consumed += 1;
        }
        self.emit(replace_code_point(self.result), self.consumed);
        self.consumed
    }

    fn named_entity(&mut self, input: &[u8], mut offset: usize) -> i64 {
        let mut current = tree(self.tree_index);
        let mut value_length = ((current & VALUE_LENGTH) >> 14) as i64;
        while offset < input.len() {
            // 압축된 연속 문자(run): 값 없음 + FLAG13
            if value_length == 0 && (current & FLAG13) != 0 {
                let run_length = ((current & BRANCH_LENGTH) >> 7) as i64;
                if self.run_consumed == 0 {
                    let first = (current & JUMP_TABLE) as u8;
                    if input[offset] != first {
                        return if self.result == 0 { 0 } else { self.emit_not_terminated() };
                    }
                    offset += 1;
                    self.excess += 1;
                    self.run_consumed += 1;
                }
                while self.run_consumed < run_length {
                    if offset >= input.len() {
                        return -1;
                    }
                    let idx = self.run_consumed - 1;
                    let packed = tree(self.tree_index + 1 + (idx >> 1));
                    let expected = if idx % 2 == 0 { packed & 0xff } else { (packed >> 8) & 0xff };
                    if input[offset] as u16 != expected {
                        self.run_consumed = 0;
                        return if self.result == 0 { 0 } else { self.emit_not_terminated() };
                    }
                    offset += 1;
                    self.excess += 1;
                    self.run_consumed += 1;
                }
                self.run_consumed = 0;
                self.tree_index += 1 + (run_length >> 1);
                current = tree(self.tree_index);
                value_length = ((current & VALUE_LENGTH) >> 14) as i64;
            }
            if offset >= input.len() {
                break;
            }
            let c = input[offset];
            // ';' 가지가 따로 없는(세미콜론 필수) 값 노드
            if c == SEMI && value_length != 0 && (current & FLAG13) != 0 {
                let consumed = self.consumed + self.excess;
                return self.emit_named_data(self.tree_index, value_length, consumed);
            }
            self.tree_index = determine_branch(current, self.tree_index + value_length.max(1), c);
            if self.tree_index < 0 {
                return if self.result == 0
                    || (self.mode == DecodingMode::Attribute && (value_length == 0 || attr_invalid_end(c)))
                {
                    0
                } else {
                    self.emit_not_terminated()
                };
            }
            current = tree(self.tree_index);
            value_length = ((current & VALUE_LENGTH) >> 14) as i64;
            if value_length != 0 {
                if c == SEMI {
                    let consumed = self.consumed + self.excess;
                    return self.emit_named_data(self.tree_index, value_length, consumed);
                }
                // `;` 없이 끝날 수 있는 옛 엔터티 (Strict 는 이 모형에 없다)
                if (current & FLAG13) == 0 {
                    self.result = self.tree_index;
                    self.consumed += self.excess;
                    self.excess = 0;
                }
            }
            offset += 1;
            self.excess += 1;
        }
        -1
    }

    fn emit_not_terminated(&mut self) -> i64 {
        let result = self.result;
        let value_length = ((tree(result) & VALUE_LENGTH) >> 14) as i64;
        let consumed = self.consumed;
        self.emit_named_data(result, value_length, consumed);
        self.consumed
    }

    fn emit_named_data(&mut self, result: i64, value_length: i64, consumed: i64) -> i64 {
        let first = if value_length == 1 { tree(result) & !(VALUE_LENGTH | FLAG13) } else { tree(result + 1) };
        self.emit(first as u32, consumed);
        if value_length == 3 {
            // 두 번째 값 (BMP 밖 문자는 서로게이트 두 단위로 저장돼 있다)
            self.emit(tree(result + 2) as u32, consumed);
        }
        consumed
    }

    /// 입력이 끝났을 때 (`end()`)
    pub(super) fn end(&mut self) -> i64 {
        match self.state {
            St::NamedEntity => {
                if self.result != 0 && (self.mode != DecodingMode::Attribute || self.result == self.tree_index) {
                    self.emit_not_terminated()
                } else {
                    0
                }
            }
            St::NumericDecimal => self.emit_numeric(0, 2),
            St::NumericHex => self.emit_numeric(0, 3),
            St::NumericStart | St::EntityStart => 0,
        }
    }
}

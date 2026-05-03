//! JSON Schema-driven structured output constraint.
//!
//! Parses a subset of JSON Schema and generates a state machine that
//! enforces valid JSON matching the schema at the token level.
//!
//! Supported schema features:
//! - type: "object", "array", "string", "number", "integer", "boolean", "null"
//! - properties (required/optional)
//! - required fields list
//! - enum values (string enums)
//! - minLength / maxLength for strings
//! - minimum / maximum for numbers
//! - items (array element type)
//! - maxItems / minItems for arrays

use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors arising from schema parsing, validation, or enforcement.
#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    /// The schema JSON could not be parsed.
    #[error("invalid schema JSON: {0}")]
    InvalidJson(String),

    /// A schema feature that is not supported was encountered.
    #[error("unsupported schema feature: {0}")]
    UnsupportedFeature(String),

    /// The schema object is missing a required `"type"` field.
    #[error("missing 'type' field in schema")]
    MissingType,

    /// The `"type"` field contained an unrecognized value.
    #[error("unknown type: '{0}'")]
    UnknownType(String),

    /// Validation of a value against the schema failed.
    #[error("validation error: {0}")]
    ValidationError(String),

    /// A constraint violation detected at a specific character position.
    #[error("schema violation at position {pos}: {msg}")]
    SchemaViolation {
        /// Character offset in the generated text.
        pos: usize,
        /// Description of the violation.
        msg: String,
    },
}

// ─────────────────────────────────────────────────────────────────────────────
// SchemaType — parsed schema representation
// ─────────────────────────────────────────────────────────────────────────────

/// A parsed JSON schema (subset).
#[derive(Debug, Clone)]
pub enum SchemaType {
    /// An object with named properties and a list of required keys.
    Object {
        /// Property name → sub-schema.
        properties: HashMap<String, SchemaType>,
        /// Keys that must be present.
        required: Vec<String>,
    },
    /// An array, optionally with a uniform item schema and length bounds.
    Array {
        /// Schema for each element.
        items: Option<Box<SchemaType>>,
        /// Minimum number of elements.
        min_items: Option<usize>,
        /// Maximum number of elements.
        max_items: Option<usize>,
    },
    /// A string, optionally constrained by enum values or length.
    String {
        /// If present, the string must be one of these values.
        enum_values: Option<Vec<String>>,
        /// Minimum string length (in characters).
        min_length: Option<usize>,
        /// Maximum string length (in characters).
        max_length: Option<usize>,
    },
    /// A floating-point number with optional bounds.
    Number {
        /// Inclusive minimum.
        minimum: Option<f64>,
        /// Inclusive maximum.
        maximum: Option<f64>,
    },
    /// An integer with optional bounds.
    Integer {
        /// Inclusive minimum.
        minimum: Option<i64>,
        /// Inclusive maximum.
        maximum: Option<i64>,
    },
    /// A JSON boolean (`true` or `false`).
    Boolean,
    /// The JSON `null` literal.
    Null,
    /// A union of schemas (`anyOf`).
    AnyOf(Vec<SchemaType>),
}

impl SchemaType {
    /// Returns a human-readable name for the schema variant.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Object { .. } => "object",
            Self::Array { .. } => "array",
            Self::String { .. } => "string",
            Self::Number { .. } => "number",
            Self::Integer { .. } => "integer",
            Self::Boolean => "boolean",
            Self::Null => "null",
            Self::AnyOf(_) => "anyOf",
        }
    }

    /// Returns `true` if `key` is in the `required` list of an object schema.
    pub fn is_required_property(&self, key: &str) -> bool {
        match self {
            Self::Object { required, .. } => required.iter().any(|k| k == key),
            _ => false,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal JSON value — minimal hand-rolled representation
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal JSON value for internal schema parsing (no serde dependency).
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    /// Try to interpret this value as a string.
    fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Try to interpret this value as an array.
    fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            Self::Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Try to interpret this value as an object (ordered key-value pairs).
    fn as_object(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            Self::Object(kv) => Some(kv.as_slice()),
            _ => None,
        }
    }

    /// Try to interpret this value as an f64.
    fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Look up a key in an object.
    fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_object()
            .and_then(|pairs| pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal JSON parser
// ─────────────────────────────────────────────────────────────────────────────

/// Skip leading ASCII whitespace.
fn skip_ws(input: &str) -> &str {
    input.trim_start()
}

/// Parse a JSON value from the beginning of `input`.
/// Returns `(value, remaining_input)`.
fn parse_json_value(input: &str) -> Result<(JsonValue, &str), SchemaError> {
    let s = skip_ws(input);
    if s.is_empty() {
        return Err(SchemaError::InvalidJson("unexpected end of input".into()));
    }

    let first = s.as_bytes()[0];
    match first {
        b'"' => parse_json_string(s),
        b'{' => parse_json_object(s),
        b'[' => parse_json_array(s),
        b't' | b'f' => parse_json_bool(s),
        b'n' => parse_json_null(s),
        b'-' | b'0'..=b'9' => parse_json_number(s),
        _ => Err(SchemaError::InvalidJson(format!(
            "unexpected character '{}'",
            s.chars().next().unwrap_or('?')
        ))),
    }
}

/// Parse a JSON string (including the surrounding quotes).
fn parse_json_string(input: &str) -> Result<(JsonValue, &str), SchemaError> {
    debug_assert!(input.starts_with('"'));
    let mut chars = input[1..].char_indices();
    let mut result = String::new();
    loop {
        match chars.next() {
            None => return Err(SchemaError::InvalidJson("unterminated string".into())),
            Some((_, '\\')) => match chars.next() {
                Some((_, '"')) => result.push('"'),
                Some((_, '\\')) => result.push('\\'),
                Some((_, '/')) => result.push('/'),
                Some((_, 'n')) => result.push('\n'),
                Some((_, 'r')) => result.push('\r'),
                Some((_, 't')) => result.push('\t'),
                Some((_, 'b')) => result.push('\u{0008}'),
                Some((_, 'f')) => result.push('\u{000C}'),
                Some((_, 'u')) => {
                    let hex = collect_n_chars(&mut chars, 4)?;
                    let cp = u32::from_str_radix(&hex, 16).map_err(|_| {
                        SchemaError::InvalidJson(format!("invalid unicode escape: \\u{hex}"))
                    })?;
                    let c = char::from_u32(cp).ok_or_else(|| {
                        SchemaError::InvalidJson(format!("invalid codepoint: U+{cp:04X}"))
                    })?;
                    result.push(c);
                }
                Some((_, c)) => {
                    return Err(SchemaError::InvalidJson(format!("unknown escape: \\{c}")))
                }
                None => return Err(SchemaError::InvalidJson("unterminated escape".into())),
            },
            Some((i, '"')) => {
                // i is the index *within* input[1..], the closing quote
                let rest = &input[1 + i + 1..];
                return Ok((JsonValue::Str(result), rest));
            }
            Some((_, c)) => result.push(c),
        }
    }
}

/// Helper: collect `n` chars from an iterator into a String.
fn collect_n_chars(iter: &mut std::str::CharIndices<'_>, n: usize) -> Result<String, SchemaError> {
    let mut s = String::with_capacity(n);
    for _ in 0..n {
        match iter.next() {
            Some((_, c)) => s.push(c),
            None => return Err(SchemaError::InvalidJson("unexpected end in escape".into())),
        }
    }
    Ok(s)
}

/// Parse a JSON number (integer or floating-point).
fn parse_json_number(input: &str) -> Result<(JsonValue, &str), SchemaError> {
    let mut end = 0;
    let bytes = input.as_bytes();
    // optional leading minus
    if end < bytes.len() && bytes[end] == b'-' {
        end += 1;
    }
    // integer part
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    // fractional part
    if end < bytes.len() && bytes[end] == b'.' {
        end += 1;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
    }
    // exponent
    if end < bytes.len() && (bytes[end] == b'e' || bytes[end] == b'E') {
        end += 1;
        if end < bytes.len() && (bytes[end] == b'+' || bytes[end] == b'-') {
            end += 1;
        }
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
    }
    if end == 0 || (end == 1 && bytes[0] == b'-') {
        return Err(SchemaError::InvalidJson("expected number".into()));
    }
    let num_str = &input[..end];
    let val: f64 = num_str
        .parse()
        .map_err(|_| SchemaError::InvalidJson(format!("invalid number: {num_str}")))?;
    Ok((JsonValue::Number(val), &input[end..]))
}

/// Parse a JSON boolean.
fn parse_json_bool(input: &str) -> Result<(JsonValue, &str), SchemaError> {
    if let Some(rest) = input.strip_prefix("true") {
        Ok((JsonValue::Bool(true), rest))
    } else if let Some(rest) = input.strip_prefix("false") {
        Ok((JsonValue::Bool(false), rest))
    } else {
        Err(SchemaError::InvalidJson("expected boolean".into()))
    }
}

/// Parse the JSON `null` literal.
fn parse_json_null(input: &str) -> Result<(JsonValue, &str), SchemaError> {
    if let Some(rest) = input.strip_prefix("null") {
        Ok((JsonValue::Null, rest))
    } else {
        Err(SchemaError::InvalidJson("expected null".into()))
    }
}

/// Parse a JSON object.
fn parse_json_object(input: &str) -> Result<(JsonValue, &str), SchemaError> {
    debug_assert!(input.starts_with('{'));
    let mut rest = skip_ws(&input[1..]);
    let mut pairs = Vec::new();
    if let Some(after_brace) = rest.strip_prefix('}') {
        return Ok((JsonValue::Object(pairs), after_brace));
    }
    loop {
        // key
        if !rest.starts_with('"') {
            return Err(SchemaError::InvalidJson("expected string key".into()));
        }
        let (key_val, after_key) = parse_json_string(rest)?;
        let key = match key_val {
            JsonValue::Str(s) => s,
            _ => return Err(SchemaError::InvalidJson("key must be string".into())),
        };
        let after_colon = skip_ws(after_key);
        if !after_colon.starts_with(':') {
            return Err(SchemaError::InvalidJson("expected ':' after key".into()));
        }
        let after_colon = skip_ws(&after_colon[1..]);
        let (val, after_val) = parse_json_value(after_colon)?;
        pairs.push((key, val));
        rest = skip_ws(after_val);
        if let Some(after_brace) = rest.strip_prefix('}') {
            return Ok((JsonValue::Object(pairs), after_brace));
        }
        if rest.starts_with(',') {
            rest = skip_ws(&rest[1..]);
        } else {
            return Err(SchemaError::InvalidJson(
                "expected ',' or '}' in object".into(),
            ));
        }
    }
}

/// Parse a JSON array.
fn parse_json_array(input: &str) -> Result<(JsonValue, &str), SchemaError> {
    debug_assert!(input.starts_with('['));
    let mut rest = skip_ws(&input[1..]);
    let mut items = Vec::new();
    if let Some(after_bracket) = rest.strip_prefix(']') {
        return Ok((JsonValue::Array(items), after_bracket));
    }
    loop {
        let (val, after_val) = parse_json_value(rest)?;
        items.push(val);
        rest = skip_ws(after_val);
        if let Some(after_bracket) = rest.strip_prefix(']') {
            return Ok((JsonValue::Array(items), after_bracket));
        }
        if rest.starts_with(',') {
            rest = skip_ws(&rest[1..]);
        } else {
            return Err(SchemaError::InvalidJson(
                "expected ',' or ']' in array".into(),
            ));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Schema conversion
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a JSON schema from its JSON string representation.
///
/// Supports the subset described in [`SchemaType`].
pub fn parse_schema(schema_json: &str) -> Result<SchemaType, SchemaError> {
    let (value, rest) = parse_json_value(schema_json)?;
    let rest_trimmed = skip_ws(rest);
    if !rest_trimmed.is_empty() {
        return Err(SchemaError::InvalidJson(format!(
            "trailing characters after schema: {rest_trimmed}"
        )));
    }
    json_value_to_schema(&value)
}

/// Convert a parsed [`JsonValue`] into a [`SchemaType`].
fn json_value_to_schema(value: &JsonValue) -> Result<SchemaType, SchemaError> {
    let _obj = value
        .as_object()
        .ok_or_else(|| SchemaError::InvalidJson("schema must be an object".into()))?;

    // Check for anyOf
    if let Some(any_of_val) = value.get("anyOf") {
        let arr = any_of_val
            .as_array()
            .ok_or_else(|| SchemaError::InvalidJson("anyOf must be an array".into()))?;
        let schemas: Result<Vec<SchemaType>, _> = arr.iter().map(json_value_to_schema).collect();
        return Ok(SchemaType::AnyOf(schemas?));
    }

    let type_val = value.get("type").ok_or(SchemaError::MissingType)?;
    let type_str = type_val
        .as_str()
        .ok_or_else(|| SchemaError::InvalidJson("'type' must be a string".into()))?;

    match type_str {
        "object" => {
            let mut properties = HashMap::new();
            if let Some(props_val) = value.get("properties") {
                if let Some(props_obj) = props_val.as_object() {
                    for (k, v) in props_obj {
                        let sub = json_value_to_schema(v)?;
                        properties.insert(k.clone(), sub);
                    }
                }
            }
            let mut required = Vec::new();
            if let Some(req_val) = value.get("required") {
                if let Some(arr) = req_val.as_array() {
                    for item in arr {
                        if let Some(s) = item.as_str() {
                            required.push(s.to_string());
                        }
                    }
                }
            }
            Ok(SchemaType::Object {
                properties,
                required,
            })
        }
        "array" => {
            let items = match value.get("items") {
                Some(v) => Some(Box::new(json_value_to_schema(v)?)),
                None => None,
            };
            let min_items = value
                .get("minItems")
                .and_then(|v| v.as_f64())
                .map(|n| n as usize);
            let max_items = value
                .get("maxItems")
                .and_then(|v| v.as_f64())
                .map(|n| n as usize);
            Ok(SchemaType::Array {
                items,
                min_items,
                max_items,
            })
        }
        "string" => {
            let enum_values = value.get("enum").and_then(|v| {
                v.as_array().map(|arr| {
                    arr.iter()
                        .filter_map(|item| item.as_str().map(|s| s.to_string()))
                        .collect()
                })
            });
            let min_length = value
                .get("minLength")
                .and_then(|v| v.as_f64())
                .map(|n| n as usize);
            let max_length = value
                .get("maxLength")
                .and_then(|v| v.as_f64())
                .map(|n| n as usize);
            Ok(SchemaType::String {
                enum_values,
                min_length,
                max_length,
            })
        }
        "number" => {
            let minimum = value.get("minimum").and_then(|v| v.as_f64());
            let maximum = value.get("maximum").and_then(|v| v.as_f64());
            Ok(SchemaType::Number { minimum, maximum })
        }
        "integer" => {
            let minimum = value
                .get("minimum")
                .and_then(|v| v.as_f64())
                .map(|n| n as i64);
            let maximum = value
                .get("maximum")
                .and_then(|v| v.as_f64())
                .map(|n| n as i64);
            Ok(SchemaType::Integer { minimum, maximum })
        }
        "boolean" => Ok(SchemaType::Boolean),
        "null" => Ok(SchemaType::Null),
        other => Err(SchemaError::UnknownType(other.to_string())),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SchemaState — state machine for tracking generation progress
// ─────────────────────────────────────────────────────────────────────────────

/// A context frame on the schema-state stack.
#[derive(Debug, Clone)]
#[allow(clippy::enum_variant_names)]
enum ContextFrame {
    /// Inside an object: tracking which keys have been emitted.
    ObjectStart {
        schema: SchemaType,
        emitted_keys: Vec<String>,
        pending_value: bool,
    },
    /// Inside an array: tracking element count.
    ArrayStart { schema: SchemaType, count: usize },
    /// Inside a string literal.
    StringStart {
        constraints: Option<(Option<usize>, Option<usize>)>,
    },
    /// Inside a number literal.
    NumberStart,
    /// Expecting a value matching a specific schema.
    ValueStart { schema: SchemaType },
}

/// State machine that tracks JSON generation progress against a schema.
///
/// Characters are fed one at a time; the machine maintains a stack of context
/// frames mirroring the nesting structure of the JSON being generated.
#[derive(Debug, Clone)]
pub struct SchemaState {
    /// Stack of context frames.
    stack: Vec<ContextFrame>,
    /// Characters generated so far.
    buffer: String,
    /// Whether we've completed the root value.
    pub is_complete: bool,
}

impl SchemaState {
    /// Create a new state machine rooted at the given schema.
    pub fn new(schema: &SchemaType) -> Self {
        Self {
            stack: vec![ContextFrame::ValueStart {
                schema: schema.clone(),
            }],
            buffer: String::new(),
            is_complete: false,
        }
    }

    /// Feed a character and check if it's valid according to the schema.
    ///
    /// Returns `Ok(true)` if the character was accepted, `Ok(false)` if it
    /// was rejected but didn't violate hard constraints, or `Err` on a
    /// definite schema violation.
    pub fn feed_char(&mut self, ch: char) -> Result<bool, SchemaError> {
        self.buffer.push(ch);

        if self.is_complete {
            return Err(SchemaError::SchemaViolation {
                pos: self.buffer.len(),
                msg: "input continues after root value is complete".into(),
            });
        }

        if self.stack.is_empty() {
            self.is_complete = true;
            return Ok(true);
        }

        // Peek at the top frame to decide what to do.
        let accepted = self.process_char(ch)?;
        Ok(accepted)
    }

    /// Internal: process one character against the current top frame.
    fn process_char(&mut self, ch: char) -> Result<bool, SchemaError> {
        // We need to pop the top frame, inspect it, and potentially push
        // replacement frames.
        let frame = match self.stack.last() {
            Some(_) => self.stack.pop(),
            None => {
                self.is_complete = true;
                return Ok(ch.is_ascii_whitespace());
            }
        };

        match frame {
            Some(ContextFrame::ValueStart { schema }) => {
                // We're expecting a new value. The first character tells us
                // what kind of JSON value this will be.
                match ch {
                    '{' => {
                        if let SchemaType::Object { .. } | SchemaType::AnyOf(_) = &schema {
                            self.stack.push(ContextFrame::ObjectStart {
                                schema,
                                emitted_keys: Vec::new(),
                                pending_value: false,
                            });
                            Ok(true)
                        } else {
                            Err(SchemaError::SchemaViolation {
                                pos: self.buffer.len(),
                                msg: format!("expected {}, got object", schema.type_name()),
                            })
                        }
                    }
                    '[' => {
                        if let SchemaType::Array { .. } | SchemaType::AnyOf(_) = &schema {
                            self.stack
                                .push(ContextFrame::ArrayStart { schema, count: 0 });
                            Ok(true)
                        } else {
                            Err(SchemaError::SchemaViolation {
                                pos: self.buffer.len(),
                                msg: format!("expected {}, got array", schema.type_name()),
                            })
                        }
                    }
                    '"' => match &schema {
                        SchemaType::String {
                            min_length,
                            max_length,
                            ..
                        } => {
                            self.stack.push(ContextFrame::StringStart {
                                constraints: Some((*min_length, *max_length)),
                            });
                            Ok(true)
                        }
                        SchemaType::AnyOf(_) => {
                            self.stack
                                .push(ContextFrame::StringStart { constraints: None });
                            Ok(true)
                        }
                        _ => Err(SchemaError::SchemaViolation {
                            pos: self.buffer.len(),
                            msg: format!("expected {}, got string", schema.type_name()),
                        }),
                    },
                    't' | 'f' => {
                        if matches!(&schema, SchemaType::Boolean | SchemaType::AnyOf(_)) {
                            // We'll just accept subsequent chars of true/false
                            self.stack.push(ContextFrame::ValueStart { schema });
                            Ok(true)
                        } else {
                            Err(SchemaError::SchemaViolation {
                                pos: self.buffer.len(),
                                msg: format!("expected {}, got boolean", schema.type_name()),
                            })
                        }
                    }
                    'r' | 'u' | 'e' | 'a' | 'l' | 's' => {
                        // Continuation of true/false/null keywords
                        Ok(true)
                    }
                    'n' => {
                        if matches!(
                            &schema,
                            SchemaType::Null | SchemaType::AnyOf(_) | SchemaType::Boolean
                        ) {
                            self.stack.push(ContextFrame::ValueStart { schema });
                            Ok(true)
                        } else {
                            Err(SchemaError::SchemaViolation {
                                pos: self.buffer.len(),
                                msg: format!("expected {}, got null", schema.type_name()),
                            })
                        }
                    }
                    '-' | '0'..='9' => {
                        if matches!(
                            &schema,
                            SchemaType::Number { .. }
                                | SchemaType::Integer { .. }
                                | SchemaType::AnyOf(_)
                        ) {
                            self.stack.push(ContextFrame::NumberStart);
                            Ok(true)
                        } else {
                            Err(SchemaError::SchemaViolation {
                                pos: self.buffer.len(),
                                msg: format!("expected {}, got number", schema.type_name()),
                            })
                        }
                    }
                    c if c.is_ascii_whitespace() => {
                        // Skip whitespace before value
                        self.stack.push(ContextFrame::ValueStart { schema });
                        Ok(true)
                    }
                    _ => Err(SchemaError::SchemaViolation {
                        pos: self.buffer.len(),
                        msg: format!("unexpected character '{ch}'"),
                    }),
                }
            }
            Some(ContextFrame::ObjectStart {
                schema,
                emitted_keys,
                pending_value,
            }) => {
                if pending_value {
                    // We just finished reading a key, expecting ':'
                    if ch == ':' || ch.is_ascii_whitespace() {
                        self.stack.push(ContextFrame::ObjectStart {
                            schema,
                            emitted_keys,
                            pending_value: ch != ':',
                        });
                        Ok(true)
                    } else {
                        Err(SchemaError::SchemaViolation {
                            pos: self.buffer.len(),
                            msg: format!("expected ':' in object, got '{ch}'"),
                        })
                    }
                } else {
                    match ch {
                        '}' => Ok(true),
                        '"' => {
                            // Start of a key
                            self.stack.push(ContextFrame::ObjectStart {
                                schema,
                                emitted_keys,
                                pending_value: true,
                            });
                            Ok(true)
                        }
                        ',' | ' ' | '\n' | '\r' | '\t' => {
                            self.stack.push(ContextFrame::ObjectStart {
                                schema,
                                emitted_keys,
                                pending_value: false,
                            });
                            Ok(true)
                        }
                        _ => Ok(true), // Accept other chars during object parsing
                    }
                }
            }
            Some(ContextFrame::ArrayStart { schema, count }) => match ch {
                ']' => Ok(true),
                ',' => {
                    self.stack.push(ContextFrame::ArrayStart {
                        schema,
                        count: count + 1,
                    });
                    Ok(true)
                }
                _ => {
                    self.stack.push(ContextFrame::ArrayStart { schema, count });
                    Ok(true)
                }
            },
            Some(ContextFrame::StringStart { constraints }) => match ch {
                '"' => Ok(true), // End of string
                '\\' => {
                    self.stack.push(ContextFrame::StringStart { constraints });
                    Ok(true)
                }
                _ => {
                    self.stack.push(ContextFrame::StringStart { constraints });
                    Ok(true)
                }
            },
            Some(ContextFrame::NumberStart) => {
                if ch.is_ascii_digit()
                    || ch == '.'
                    || ch == '-'
                    || ch == 'e'
                    || ch == 'E'
                    || ch == '+'
                {
                    self.stack.push(ContextFrame::NumberStart);
                    Ok(true)
                } else {
                    // Number ended, this char belongs to the parent
                    Ok(true)
                }
            }
            None => {
                self.is_complete = true;
                Ok(ch.is_ascii_whitespace())
            }
        }
    }

    /// Get the set of valid next characters at the current position.
    ///
    /// This is a simplified heuristic — for complex schemas, the set may be
    /// conservative (allowing more than strictly valid).
    pub fn valid_next_chars(&self) -> Vec<char> {
        match self.stack.last() {
            None => vec![],
            Some(ContextFrame::ValueStart { schema }) => match schema {
                SchemaType::Object { .. } => vec!['{', ' ', '\n'],
                SchemaType::Array { .. } => vec!['[', ' ', '\n'],
                SchemaType::String { .. } => vec!['"'],
                SchemaType::Number { .. } | SchemaType::Integer { .. } => {
                    vec!['-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                }
                SchemaType::Boolean => vec!['t', 'f'],
                SchemaType::Null => vec!['n'],
                SchemaType::AnyOf(_) => {
                    vec![
                        '{', '[', '"', 't', 'f', 'n', '-', '0', '1', '2', '3', '4', '5', '6', '7',
                        '8', '9',
                    ]
                }
            },
            Some(ContextFrame::ObjectStart { pending_value, .. }) => {
                if *pending_value {
                    vec![':', ' ']
                } else {
                    vec!['"', '}', ',', ' ', '\n']
                }
            }
            Some(ContextFrame::ArrayStart { .. }) => {
                vec![
                    ']', ',', '"', '{', '[', 't', 'f', 'n', '-', '0', '1', '2', '3', '4', '5', '6',
                    '7', '8', '9', ' ', '\n',
                ]
            }
            Some(ContextFrame::StringStart { .. }) => {
                // Almost any character is valid inside a string
                let mut chars: Vec<char> = (0x20u8..=0x7Eu8).map(|b| b as char).collect();
                chars.push('\n');
                chars
            }
            Some(ContextFrame::NumberStart) => {
                vec![
                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-', 'e', 'E', '+',
                ]
            }
        }
    }

    /// Get a regex pattern for valid continuations (for masking logits).
    pub fn continuation_pattern(&self) -> String {
        match self.stack.last() {
            None => String::new(),
            Some(ContextFrame::ValueStart { schema }) => match schema {
                SchemaType::Object { .. } => "\\{".to_string(),
                SchemaType::Array { .. } => "\\[".to_string(),
                SchemaType::String { .. } => "\"".to_string(),
                SchemaType::Number { .. } | SchemaType::Integer { .. } => "-?[0-9]".to_string(),
                SchemaType::Boolean => "[tf]".to_string(),
                SchemaType::Null => "n".to_string(),
                SchemaType::AnyOf(_) => "[\\{\\[\"\\-0-9tfn]".to_string(),
            },
            Some(ContextFrame::ObjectStart { .. }) => "[\"\\},: \\n]".to_string(),
            Some(ContextFrame::ArrayStart { .. }) => "[\\]\\[,0-9tfn\"\\{\\ \\n]".to_string(),
            Some(ContextFrame::StringStart { .. }) => "[^\"\\\\]|\\\\.".to_string(),
            Some(ContextFrame::NumberStart) => "[0-9.\\-eE+]".to_string(),
        }
    }

    /// Returns a reference to the characters generated so far.
    pub fn buffer(&self) -> &str {
        &self.buffer
    }

    /// Returns the current nesting depth (number of frames on the stack).
    pub fn depth(&self) -> usize {
        self.stack.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation
// ─────────────────────────────────────────────────────────────────────────────

/// Validate that `text` is valid JSON matching the given `schema`.
///
/// This parses the text as JSON and checks structural conformance.
pub fn validate_against_schema(text: &str, schema: &SchemaType) -> Result<bool, SchemaError> {
    let (value, rest) = parse_json_value(text)
        .map_err(|e| SchemaError::ValidationError(format!("JSON parse error: {e}")))?;
    let rest_trimmed = skip_ws(rest);
    if !rest_trimmed.is_empty() {
        return Err(SchemaError::ValidationError(
            "trailing characters after JSON value".into(),
        ));
    }
    validate_value_against_schema(&value, schema)
}

/// Recursively validate a parsed value against a schema.
fn validate_value_against_schema(
    value: &JsonValue,
    schema: &SchemaType,
) -> Result<bool, SchemaError> {
    match schema {
        SchemaType::Object {
            properties,
            required,
        } => {
            let pairs = match value.as_object() {
                Some(p) => p,
                None => return Ok(false),
            };
            // Check required fields
            for key in required {
                if !pairs.iter().any(|(k, _)| k == key) {
                    return Ok(false);
                }
            }
            // Validate each known property
            for (k, v) in pairs {
                if let Some(prop_schema) = properties.get(k) {
                    if !validate_value_against_schema(v, prop_schema)? {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        }
        SchemaType::Array {
            items,
            min_items,
            max_items,
        } => {
            let arr = match value.as_array() {
                Some(a) => a,
                None => return Ok(false),
            };
            if let Some(min) = min_items {
                if arr.len() < *min {
                    return Ok(false);
                }
            }
            if let Some(max) = max_items {
                if arr.len() > *max {
                    return Ok(false);
                }
            }
            if let Some(item_schema) = items {
                for elem in arr {
                    if !validate_value_against_schema(elem, item_schema)? {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        }
        SchemaType::String {
            enum_values,
            min_length,
            max_length,
        } => {
            let s = match value.as_str() {
                Some(s) => s,
                None => return Ok(false),
            };
            if let Some(enums) = enum_values {
                if !enums.iter().any(|e| e == s) {
                    return Ok(false);
                }
            }
            if let Some(min) = min_length {
                if s.chars().count() < *min {
                    return Ok(false);
                }
            }
            if let Some(max) = max_length {
                if s.chars().count() > *max {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        SchemaType::Number { minimum, maximum } => {
            let n = match value.as_f64() {
                Some(n) => n,
                None => return Ok(false),
            };
            if let Some(min) = minimum {
                if n < *min {
                    return Ok(false);
                }
            }
            if let Some(max) = maximum {
                if n > *max {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        SchemaType::Integer { minimum, maximum } => {
            let n = match value.as_f64() {
                Some(n) => n,
                None => return Ok(false),
            };
            // Check it's actually an integer
            if n.fract() != 0.0 {
                return Ok(false);
            }
            let i = n as i64;
            if let Some(min) = minimum {
                if i < *min {
                    return Ok(false);
                }
            }
            if let Some(max) = maximum {
                if i > *max {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        SchemaType::Boolean => match value {
            JsonValue::Bool(_) => Ok(true),
            _ => Ok(false),
        },
        SchemaType::Null => match value {
            JsonValue::Null => Ok(true),
            _ => Ok(false),
        },
        SchemaType::AnyOf(schemas) => {
            for sub_schema in schemas {
                if validate_value_against_schema(value, sub_schema)? {
                    return Ok(true);
                }
            }
            Ok(false)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Template / example generation
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a template/skeleton from a schema (for prompting).
///
/// The template uses placeholder values to illustrate the expected shape.
pub fn schema_template(schema: &SchemaType) -> String {
    match schema {
        SchemaType::Object {
            properties,
            required,
        } => {
            if properties.is_empty() {
                return "{}".to_string();
            }
            let mut parts = Vec::new();
            // Emit required properties first, then optional
            let mut sorted_keys: Vec<&String> = properties.keys().collect();
            sorted_keys.sort();
            for key in &sorted_keys {
                let sub = properties.get(*key).expect("key exists in map");
                let marker = if required.iter().any(|r| r == *key) {
                    " /* required */"
                } else {
                    " /* optional */"
                };
                parts.push(format!("  \"{key}\": {}{marker}", schema_template(sub)));
            }
            format!("{{\n{}\n}}", parts.join(",\n"))
        }
        SchemaType::Array { items, .. } => match items {
            Some(item_schema) => format!("[{}]", schema_template(item_schema)),
            None => "[]".to_string(),
        },
        SchemaType::String { enum_values, .. } => {
            if let Some(enums) = enum_values {
                if let Some(first) = enums.first() {
                    return format!("\"{first}\"");
                }
            }
            "\"<string>\"".to_string()
        }
        SchemaType::Number { .. } => "0.0".to_string(),
        SchemaType::Integer { .. } => "0".to_string(),
        SchemaType::Boolean => "true".to_string(),
        SchemaType::Null => "null".to_string(),
        SchemaType::AnyOf(schemas) => {
            if let Some(first) = schemas.first() {
                schema_template(first)
            } else {
                "null".to_string()
            }
        }
    }
}

/// Generate an example JSON string matching the schema.
///
/// Produces valid JSON using sensible default values.
pub fn schema_example(schema: &SchemaType) -> String {
    match schema {
        SchemaType::Object {
            properties,
            required: _,
        } => {
            if properties.is_empty() {
                return "{}".to_string();
            }
            let mut parts = Vec::new();
            let mut sorted_keys: Vec<&String> = properties.keys().collect();
            sorted_keys.sort();
            for key in &sorted_keys {
                // In the example, emit all properties (required + optional)
                let sub = properties.get(*key).expect("key exists in map");
                parts.push(format!("\"{}\":{}", key, schema_example(sub)));
            }
            format!("{{{}}}", parts.join(","))
        }
        SchemaType::Array {
            items, min_items, ..
        } => {
            let count = min_items.unwrap_or(1).max(1);
            match items {
                Some(item_schema) => {
                    let elems: Vec<String> =
                        (0..count).map(|_| schema_example(item_schema)).collect();
                    format!("[{}]", elems.join(","))
                }
                None => "[]".to_string(),
            }
        }
        SchemaType::String {
            enum_values,
            min_length,
            ..
        } => {
            if let Some(enums) = enum_values {
                if let Some(first) = enums.first() {
                    return format!("\"{first}\"");
                }
            }
            let min_len = min_length.unwrap_or(0);
            let example = if min_len > 0 {
                "x".repeat(min_len)
            } else {
                "example".to_string()
            };
            format!("\"{example}\"")
        }
        SchemaType::Number { minimum, .. } => {
            let val = minimum.unwrap_or(0.0);
            if val == val.floor() {
                format!("{val:.1}")
            } else {
                format!("{val}")
            }
        }
        SchemaType::Integer { minimum, .. } => {
            let val = minimum.unwrap_or(0);
            format!("{val}")
        }
        SchemaType::Boolean => "true".to_string(),
        SchemaType::Null => "null".to_string(),
        SchemaType::AnyOf(schemas) => {
            if let Some(first) = schemas.first() {
                schema_example(first)
            } else {
                "null".to_string()
            }
        }
    }
}

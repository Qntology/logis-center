//! Tests for the `json_schema` module.

use oxibonsai_runtime::json_schema::{
    parse_schema, schema_example, schema_template, validate_against_schema, SchemaError,
    SchemaState, SchemaType,
};

// ─────────────────────────────────────────────────────────────────────────────
// Parsing tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn parse_schema_object() {
    let s = r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#;
    let schema = parse_schema(s).expect("should parse");
    assert_eq!(schema.type_name(), "object");
    assert!(schema.is_required_property("name"));
    assert!(!schema.is_required_property("age"));
}

#[test]
fn parse_schema_array() {
    let s = r#"{"type":"array","items":{"type":"number"}}"#;
    let schema = parse_schema(s).expect("should parse");
    assert_eq!(schema.type_name(), "array");
}

#[test]
fn parse_schema_string_enum() {
    let s = r#"{"type":"string","enum":["a","b","c"]}"#;
    let schema = parse_schema(s).expect("should parse");
    assert_eq!(schema.type_name(), "string");
    if let SchemaType::String { enum_values, .. } = &schema {
        let enums = enum_values.as_ref().expect("should have enum_values");
        assert_eq!(enums.len(), 3);
        assert_eq!(enums[0], "a");
        assert_eq!(enums[1], "b");
        assert_eq!(enums[2], "c");
    } else {
        panic!("expected String variant");
    }
}

#[test]
fn parse_schema_boolean() {
    let s = r#"{"type":"boolean"}"#;
    let schema = parse_schema(s).expect("should parse");
    assert_eq!(schema.type_name(), "boolean");
}

#[test]
fn parse_schema_null() {
    let s = r#"{"type":"null"}"#;
    let schema = parse_schema(s).expect("should parse");
    assert_eq!(schema.type_name(), "null");
}

#[test]
fn parse_schema_integer_range() {
    let s = r#"{"type":"integer","minimum":0,"maximum":100}"#;
    let schema = parse_schema(s).expect("should parse");
    assert_eq!(schema.type_name(), "integer");
    if let SchemaType::Integer { minimum, maximum } = &schema {
        assert_eq!(*minimum, Some(0));
        assert_eq!(*maximum, Some(100));
    } else {
        panic!("expected Integer variant");
    }
}

#[test]
fn parse_schema_missing_type_error() {
    let s = r#"{}"#;
    let err = parse_schema(s).expect_err("should fail");
    assert!(matches!(err, SchemaError::MissingType));
}

#[test]
fn parse_schema_unknown_type_error() {
    let s = r#"{"type":"xyz"}"#;
    let err = parse_schema(s).expect_err("should fail");
    assert!(matches!(err, SchemaError::UnknownType(_)));
}

// ─────────────────────────────────────────────────────────────────────────────
// type_name tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn schema_type_name() {
    use std::collections::HashMap;

    let cases: Vec<(SchemaType, &str)> = vec![
        (
            SchemaType::Object {
                properties: HashMap::new(),
                required: vec![],
            },
            "object",
        ),
        (
            SchemaType::Array {
                items: None,
                min_items: None,
                max_items: None,
            },
            "array",
        ),
        (
            SchemaType::String {
                enum_values: None,
                min_length: None,
                max_length: None,
            },
            "string",
        ),
        (
            SchemaType::Number {
                minimum: None,
                maximum: None,
            },
            "number",
        ),
        (
            SchemaType::Integer {
                minimum: None,
                maximum: None,
            },
            "integer",
        ),
        (SchemaType::Boolean, "boolean"),
        (SchemaType::Null, "null"),
        (SchemaType::AnyOf(vec![]), "anyOf"),
    ];
    for (schema, expected_name) in &cases {
        assert_eq!(schema.type_name(), *expected_name);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Validation tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn validate_simple_object() {
    let s = r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#;
    let schema = parse_schema(s).expect("should parse");
    let result = validate_against_schema(r#"{"name":"test"}"#, &schema).expect("should validate");
    assert!(result);
}

#[test]
fn validate_number_in_range() {
    let s = r#"{"type":"number","minimum":0,"maximum":100}"#;
    let schema = parse_schema(s).expect("should parse");
    let result = validate_against_schema("42", &schema).expect("should validate");
    assert!(result);
    // Out of range
    let result2 = validate_against_schema("200", &schema).expect("should validate");
    assert!(!result2);
}

#[test]
fn validate_array_with_items() {
    let s = r#"{"type":"array","items":{"type":"number"}}"#;
    let schema = parse_schema(s).expect("should parse");
    let result = validate_against_schema("[1,2,3]", &schema).expect("should validate");
    assert!(result);
}

#[test]
fn validate_boolean_true() {
    let s = r#"{"type":"boolean"}"#;
    let schema = parse_schema(s).expect("should parse");
    let result = validate_against_schema("true", &schema).expect("should validate");
    assert!(result);
}

#[test]
fn validate_null() {
    let s = r#"{"type":"null"}"#;
    let schema = parse_schema(s).expect("should parse");
    let result = validate_against_schema("null", &schema).expect("should validate");
    assert!(result);
}

#[test]
fn validate_invalid_rejects() {
    let s = r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#;
    let schema = parse_schema(s).expect("should parse");
    // name is a number instead of a string — should fail
    let result = validate_against_schema(r#"{"name":42}"#, &schema).expect("should validate");
    assert!(!result);
}

// ─────────────────────────────────────────────────────────────────────────────
// Template / example tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn schema_template_object() {
    let s = r#"{"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"}},"required":["name"]}"#;
    let schema = parse_schema(s).expect("should parse");
    let tmpl = schema_template(&schema);
    assert!(tmpl.contains("name"), "template should contain 'name'");
    assert!(tmpl.contains("age"), "template should contain 'age'");
    assert!(
        tmpl.contains("required"),
        "template should show required marker"
    );
}

#[test]
fn schema_example_object() {
    let s = r#"{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}"#;
    let schema = parse_schema(s).expect("should parse");
    let example = schema_example(&schema);
    // The example should be valid JSON matching the schema
    assert!(example.contains("\"name\""));
    let is_valid = validate_against_schema(&example, &schema).expect("should validate");
    assert!(is_valid, "generated example should pass validation");
}

#[test]
fn schema_example_string() {
    let s = r#"{"type":"string"}"#;
    let schema = parse_schema(s).expect("should parse");
    let example = schema_example(&schema);
    assert!(example.starts_with('"') && example.ends_with('"'));
}

// ─────────────────────────────────────────────────────────────────────────────
// SchemaState tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn schema_state_new() {
    let s = r#"{"type":"object","properties":{}}"#;
    let schema = parse_schema(s).expect("should parse");
    let state = SchemaState::new(&schema);
    assert!(!state.is_complete);
    assert!(state.buffer().is_empty());
}

#[test]
fn schema_state_depth() {
    let s = r#"{"type":"object","properties":{"name":{"type":"string"}}}"#;
    let schema = parse_schema(s).expect("should parse");
    let state = SchemaState::new(&schema);
    // Initially has one frame (ValueStart)
    assert_eq!(state.depth(), 1);
}

#[test]
fn schema_state_feed_object_start() {
    let s = r#"{"type":"object","properties":{}}"#;
    let schema = parse_schema(s).expect("should parse");
    let mut state = SchemaState::new(&schema);
    let accepted = state.feed_char('{').expect("should accept");
    assert!(accepted);
    assert!(!state.is_complete);
}

#[test]
fn schema_state_valid_next_chars() {
    let s = r#"{"type":"boolean"}"#;
    let schema = parse_schema(s).expect("should parse");
    let state = SchemaState::new(&schema);
    let valid = state.valid_next_chars();
    assert!(valid.contains(&'t'));
    assert!(valid.contains(&'f'));
}

#[test]
fn schema_state_continuation_pattern_not_empty() {
    let s = r#"{"type":"number"}"#;
    let schema = parse_schema(s).expect("should parse");
    let state = SchemaState::new(&schema);
    let pattern = state.continuation_pattern();
    assert!(!pattern.is_empty());
}

#[test]
fn parse_schema_number_with_bounds() {
    let s = r#"{"type":"number","minimum":-10.5,"maximum":99.9}"#;
    let schema = parse_schema(s).expect("should parse");
    if let SchemaType::Number { minimum, maximum } = &schema {
        assert!((minimum.expect("has min") - (-10.5)).abs() < f64::EPSILON);
        assert!((maximum.expect("has max") - 99.9).abs() < f64::EPSILON);
    } else {
        panic!("expected Number variant");
    }
}

#[test]
fn parse_schema_string_with_length_constraints() {
    let s = r#"{"type":"string","minLength":2,"maxLength":10}"#;
    let schema = parse_schema(s).expect("should parse");
    if let SchemaType::String {
        min_length,
        max_length,
        ..
    } = &schema
    {
        assert_eq!(*min_length, Some(2));
        assert_eq!(*max_length, Some(10));
    } else {
        panic!("expected String variant");
    }
}

#[test]
fn validate_string_length_constraints() {
    let s = r#"{"type":"string","minLength":2,"maxLength":5}"#;
    let schema = parse_schema(s).expect("should parse");
    // Too short
    let result = validate_against_schema(r#""a""#, &schema).expect("should validate");
    assert!(!result);
    // Just right
    let result2 = validate_against_schema(r#""abc""#, &schema).expect("should validate");
    assert!(result2);
    // Too long
    let result3 = validate_against_schema(r#""abcdef""#, &schema).expect("should validate");
    assert!(!result3);
}

#[test]
fn validate_enum_string() {
    let s = r#"{"type":"string","enum":["red","green","blue"]}"#;
    let schema = parse_schema(s).expect("should parse");
    let result = validate_against_schema(r#""red""#, &schema).expect("should validate");
    assert!(result);
    let result2 = validate_against_schema(r#""yellow""#, &schema).expect("should validate");
    assert!(!result2);
}

#[test]
fn validate_integer_range() {
    let s = r#"{"type":"integer","minimum":0,"maximum":100}"#;
    let schema = parse_schema(s).expect("should parse");
    let result = validate_against_schema("50", &schema).expect("should validate");
    assert!(result);
    let result2 = validate_against_schema("150", &schema).expect("should validate");
    assert!(!result2);
    // Non-integer should fail
    let result3 = validate_against_schema("3.14", &schema).expect("should validate");
    assert!(!result3);
}

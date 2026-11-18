use std::io::{Read, Write};
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;
use anyhow::Result;
use serde_json::Value;

pub fn decompress_to_value(binary_data: &[u8]) -> Result<Value> {
    let mut decoder = GzDecoder::new(binary_data);
    let mut json_str = String::new();
    decoder.read_to_string(&mut json_str)?;
    Ok(serde_json::from_str(&json_str)?)
}

pub fn compress_value(value: &Value) -> Result<Vec<u8>> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    let mut sorted = value.clone();
    // JSON 객체 키 순서를 예전(키 정렬)과 같게: serde_json preserve_order 가 켜져도 같은 바이트 (logis_schema::json_order 와 같은 규칙)
    sorted.sort_all_objects();
    encoder.write_all(sorted.to_string().as_bytes())?;
    Ok(encoder.finish()?)
}
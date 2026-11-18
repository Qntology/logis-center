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
    encoder.write_all(value.to_string().as_bytes())?;
    Ok(encoder.finish()?)
}
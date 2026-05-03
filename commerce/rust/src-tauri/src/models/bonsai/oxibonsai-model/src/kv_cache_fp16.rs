//! FP16 KV cache — halves memory usage by storing keys/values in half precision.
//!
//! Quantizes on write (f32 -> f16), dequantizes on read (f16 -> f32).
//! This trades a small amount of numerical precision for a 2x reduction
//! in KV cache memory, which is critical for long-context inference.

use half::f16;

use crate::error::{ModelError, ModelResult};

/// KV cache that stores keys and values in FP16 (half precision).
///
/// Memory layout: `[layer][pos * num_kv_heads * head_dim]` stored contiguously
/// per layer for cache-friendly sequential access during attention.
#[derive(Debug)]
pub struct KvCacheFp16 {
    /// Key storage: one `Vec<f16>` per layer, holding all heads and positions.
    keys: Vec<Vec<f16>>,
    /// Value storage: one `Vec<f16>` per layer.
    values: Vec<Vec<f16>>,
    /// Number of transformer layers.
    num_layers: usize,
    /// Number of KV heads per layer.
    num_kv_heads: usize,
    /// Dimension per head.
    head_dim: usize,
    /// Maximum sequence length (capacity).
    max_seq_len: usize,
    /// Current number of tokens stored.
    current_len: usize,
}

impl KvCacheFp16 {
    /// Create a new FP16 KV cache with the given dimensions.
    ///
    /// Pre-allocates storage for `max_seq_len` positions across all layers and heads.
    pub fn new(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let per_layer_size = num_kv_heads * max_seq_len * head_dim;
        let keys = (0..num_layers)
            .map(|_| vec![f16::ZERO; per_layer_size])
            .collect();
        let values = (0..num_layers)
            .map(|_| vec![f16::ZERO; per_layer_size])
            .collect();

        Self {
            keys,
            values,
            num_layers,
            num_kv_heads,
            head_dim,
            max_seq_len,
            current_len: 0,
        }
    }

    /// Store key/value for a given layer and head at a position.
    ///
    /// Quantizes f32 inputs to f16 on write.
    pub fn store(
        &mut self,
        layer: usize,
        head: usize,
        pos: usize,
        key: &[f32],
        value: &[f32],
    ) -> ModelResult<()> {
        if layer >= self.num_layers {
            return Err(ModelError::ShapeMismatch {
                name: "kv_cache_fp16 layer".to_string(),
                expected: vec![self.num_layers],
                actual: vec![layer],
            });
        }
        if head >= self.num_kv_heads {
            return Err(ModelError::ShapeMismatch {
                name: "kv_cache_fp16 head".to_string(),
                expected: vec![self.num_kv_heads],
                actual: vec![head],
            });
        }
        if pos >= self.max_seq_len {
            return Err(ModelError::SequenceTooLong {
                seq_len: pos + 1,
                max_ctx: self.max_seq_len,
            });
        }
        if key.len() != self.head_dim || value.len() != self.head_dim {
            return Err(ModelError::ShapeMismatch {
                name: "kv_cache_fp16 key/value dim".to_string(),
                expected: vec![self.head_dim],
                actual: vec![key.len()],
            });
        }

        let offset = self.offset(head, pos);

        let layer_keys = &mut self.keys[layer];
        let layer_values = &mut self.values[layer];

        for (i, &k) in key.iter().enumerate() {
            layer_keys[offset + i] = f16::from_f32(k);
        }
        for (i, &v) in value.iter().enumerate() {
            layer_values[offset + i] = f16::from_f32(v);
        }

        if pos >= self.current_len {
            self.current_len = pos + 1;
        }

        Ok(())
    }

    /// Retrieve key for a given layer, head, and position (dequantize f16 -> f32).
    pub fn get_key(&self, layer: usize, head: usize, pos: usize) -> ModelResult<Vec<f32>> {
        self.validate_indices(layer, head, pos)?;
        let offset = self.offset(head, pos);
        let result: Vec<f32> = self.keys[layer][offset..offset + self.head_dim]
            .iter()
            .map(|h| h.to_f32())
            .collect();
        Ok(result)
    }

    /// Retrieve value for a given layer, head, and position (dequantize f16 -> f32).
    pub fn get_value(&self, layer: usize, head: usize, pos: usize) -> ModelResult<Vec<f32>> {
        self.validate_indices(layer, head, pos)?;
        let offset = self.offset(head, pos);
        let result: Vec<f32> = self.values[layer][offset..offset + self.head_dim]
            .iter()
            .map(|h| h.to_f32())
            .collect();
        Ok(result)
    }

    /// Get all keys for a layer/head up to `end_pos` (exclusive), dequantized to f32.
    ///
    /// Returns a contiguous Vec of `[end_pos * head_dim]` floats in row-major order.
    pub fn get_keys_range(
        &self,
        layer: usize,
        head: usize,
        end_pos: usize,
    ) -> ModelResult<Vec<f32>> {
        if layer >= self.num_layers || head >= self.num_kv_heads {
            return Err(ModelError::ShapeMismatch {
                name: "kv_cache_fp16 range indices".to_string(),
                expected: vec![self.num_layers, self.num_kv_heads],
                actual: vec![layer, head],
            });
        }
        let end = end_pos.min(self.current_len);
        let start_offset = self.offset(head, 0);
        let total_elements = end * self.head_dim;

        let result: Vec<f32> = self.keys[layer][start_offset..start_offset + total_elements]
            .iter()
            .map(|h| h.to_f32())
            .collect();
        Ok(result)
    }

    /// Get all values for a layer/head up to `end_pos` (exclusive), dequantized to f32.
    pub fn get_values_range(
        &self,
        layer: usize,
        head: usize,
        end_pos: usize,
    ) -> ModelResult<Vec<f32>> {
        if layer >= self.num_layers || head >= self.num_kv_heads {
            return Err(ModelError::ShapeMismatch {
                name: "kv_cache_fp16 range indices".to_string(),
                expected: vec![self.num_layers, self.num_kv_heads],
                actual: vec![layer, head],
            });
        }
        let end = end_pos.min(self.current_len);
        let start_offset = self.offset(head, 0);
        let total_elements = end * self.head_dim;

        let result: Vec<f32> = self.values[layer][start_offset..start_offset + total_elements]
            .iter()
            .map(|h| h.to_f32())
            .collect();
        Ok(result)
    }

    /// Current number of tokens stored in the cache.
    pub fn current_len(&self) -> usize {
        self.current_len
    }

    /// Maximum sequence length (capacity).
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Reset the cache, clearing all stored positions.
    pub fn reset(&mut self) {
        self.current_len = 0;
        for layer_keys in &mut self.keys {
            for v in layer_keys.iter_mut() {
                *v = f16::ZERO;
            }
        }
        for layer_values in &mut self.values {
            for v in layer_values.iter_mut() {
                *v = f16::ZERO;
            }
        }
    }

    /// Total memory used by this cache in bytes.
    ///
    /// Only counts the FP16 data storage, not struct overhead.
    pub fn memory_usage_bytes(&self) -> usize {
        let per_layer = self.num_kv_heads * self.max_seq_len * self.head_dim;
        // Each f16 is 2 bytes, keys + values = 2x
        self.num_layers * per_layer * std::mem::size_of::<f16>() * 2
    }

    /// Compute the flat offset into a layer's storage for a given head and position.
    fn offset(&self, head: usize, pos: usize) -> usize {
        (head * self.max_seq_len + pos) * self.head_dim
    }

    /// Validate layer, head, and position indices.
    fn validate_indices(&self, layer: usize, head: usize, pos: usize) -> ModelResult<()> {
        if layer >= self.num_layers {
            return Err(ModelError::ShapeMismatch {
                name: "kv_cache_fp16 layer".to_string(),
                expected: vec![self.num_layers],
                actual: vec![layer],
            });
        }
        if head >= self.num_kv_heads {
            return Err(ModelError::ShapeMismatch {
                name: "kv_cache_fp16 head".to_string(),
                expected: vec![self.num_kv_heads],
                actual: vec![head],
            });
        }
        if pos >= self.max_seq_len {
            return Err(ModelError::SequenceTooLong {
                seq_len: pos + 1,
                max_ctx: self.max_seq_len,
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_and_retrieve_roundtrip() {
        let mut cache = KvCacheFp16::new(2, 4, 64, 16);

        let key: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let value: Vec<f32> = (0..64).map(|i| (i as f32 + 1.0) * 0.05).collect();

        cache
            .store(0, 0, 0, &key, &value)
            .expect("store should succeed");

        let retrieved_key = cache.get_key(0, 0, 0).expect("get_key should succeed");
        let retrieved_value = cache.get_value(0, 0, 0).expect("get_value should succeed");

        // FP16 has ~3 decimal digits of precision; tolerance ~0.01 for values up to 6.3
        for (orig, retrieved) in key.iter().zip(retrieved_key.iter()) {
            assert!(
                (orig - retrieved).abs() < 0.02,
                "key mismatch: orig={orig}, retrieved={retrieved}"
            );
        }
        for (orig, retrieved) in value.iter().zip(retrieved_value.iter()) {
            assert!(
                (orig - retrieved).abs() < 0.02,
                "value mismatch: orig={orig}, retrieved={retrieved}"
            );
        }
    }

    #[test]
    fn store_multiple_positions() {
        let mut cache = KvCacheFp16::new(1, 1, 4, 8);

        let k0 = vec![1.0, 2.0, 3.0, 4.0];
        let v0 = vec![5.0, 6.0, 7.0, 8.0];
        let k1 = vec![9.0, 10.0, 11.0, 12.0];
        let v1 = vec![13.0, 14.0, 15.0, 16.0];

        cache.store(0, 0, 0, &k0, &v0).expect("store pos 0");
        cache.store(0, 0, 1, &k1, &v1).expect("store pos 1");

        assert_eq!(cache.current_len(), 2);

        let keys_range = cache.get_keys_range(0, 0, 2).expect("get_keys_range");
        assert_eq!(keys_range.len(), 8);
        // Check first position
        assert!((keys_range[0] - 1.0).abs() < 0.02);
        // Check second position
        assert!((keys_range[4] - 9.0).abs() < 0.1);
    }

    #[test]
    fn memory_usage_calculation() {
        let cache = KvCacheFp16::new(36, 8, 128, 4096);
        // 36 layers * 8 heads * 4096 seq * 128 dim * 2 bytes * 2 (K+V)
        let expected = 36 * 8 * 4096 * 128 * 2 * 2;
        assert_eq!(cache.memory_usage_bytes(), expected);
    }

    #[test]
    fn memory_usage_half_of_fp32() {
        let num_layers = 36;
        let num_kv_heads = 8;
        let head_dim = 128;
        let max_seq_len = 4096;

        let fp16_cache = KvCacheFp16::new(num_layers, num_kv_heads, head_dim, max_seq_len);
        let fp32_cache =
            crate::kv_cache::KvCache::new(num_layers, num_kv_heads, head_dim, max_seq_len);

        assert_eq!(
            fp16_cache.memory_usage_bytes() * 2,
            fp32_cache.memory_bytes()
        );
    }

    #[test]
    fn reset_clears_state() {
        let mut cache = KvCacheFp16::new(1, 1, 4, 8);
        let key = vec![1.0, 2.0, 3.0, 4.0];
        let value = vec![5.0, 6.0, 7.0, 8.0];

        cache.store(0, 0, 0, &key, &value).expect("store");
        assert_eq!(cache.current_len(), 1);

        cache.reset();
        assert_eq!(cache.current_len(), 0);
    }

    #[test]
    fn capacity_boundary() {
        let mut cache = KvCacheFp16::new(1, 1, 4, 2);
        let key = vec![1.0; 4];
        let value = vec![2.0; 4];

        cache.store(0, 0, 0, &key, &value).expect("pos 0");
        cache.store(0, 0, 1, &key, &value).expect("pos 1");

        // Position 2 should fail (max_seq_len=2, so valid positions are 0,1)
        let result = cache.store(0, 0, 2, &key, &value);
        assert!(result.is_err());
    }

    #[test]
    fn invalid_layer_returns_error() {
        let cache = KvCacheFp16::new(2, 4, 64, 16);
        let result = cache.get_key(5, 0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn invalid_head_returns_error() {
        let cache = KvCacheFp16::new(2, 4, 64, 16);
        let result = cache.get_key(0, 10, 0);
        assert!(result.is_err());
    }

    #[test]
    fn fp16_precision_small_values() {
        let mut cache = KvCacheFp16::new(1, 1, 4, 4);
        // Small values that should be well-represented in f16
        let key = vec![0.001, 0.01, 0.1, 1.0];
        let value = vec![-0.5, 0.0, 0.5, 1.5];

        cache.store(0, 0, 0, &key, &value).expect("store");

        let retrieved_key = cache.get_key(0, 0, 0).expect("get_key");
        let retrieved_value = cache.get_value(0, 0, 0).expect("get_value");

        for (orig, retrieved) in key.iter().zip(retrieved_key.iter()) {
            let tolerance = orig.abs() * 0.01 + 0.001; // relative + absolute tolerance
            assert!(
                (orig - retrieved).abs() < tolerance,
                "key precision: orig={orig}, retrieved={retrieved}"
            );
        }
        for (orig, retrieved) in value.iter().zip(retrieved_value.iter()) {
            let tolerance = orig.abs() * 0.01 + 0.001;
            assert!(
                (orig - retrieved).abs() < tolerance,
                "value precision: orig={orig}, retrieved={retrieved}"
            );
        }
    }
}

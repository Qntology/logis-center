//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(test)]
use crate::model::types::BonsaiModel;
#[cfg(test)]
use crate::model_registry::ModelVariant;
#[cfg(test)]
use oxibonsai_core::Qwen3Config;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn model_creation() {
        let config = Qwen3Config::bonsai_8b();
        let model = BonsaiModel::new(config);
        assert_eq!(model.config().num_layers, 36);
        assert_eq!(model.config().hidden_size, 4096);
    }
    #[test]
    fn model_new_has_empty_blocks() {
        let config = Qwen3Config::bonsai_8b();
        let model = BonsaiModel::new(config);
        assert_eq!(model.blocks.len(), 0);
    }
    #[test]
    fn model_variant_detection() {
        let model_8b = BonsaiModel::new(Qwen3Config::bonsai_8b());
        assert_eq!(model_8b.variant(), ModelVariant::Bonsai8B);
        let model_4b = BonsaiModel::new(Qwen3Config::bonsai_4b());
        assert_eq!(model_4b.variant(), ModelVariant::Bonsai4B);
        let model_1_7b = BonsaiModel::new(Qwen3Config::bonsai_1_7b());
        assert_eq!(model_1_7b.variant(), ModelVariant::Bonsai1_7B);
    }
    #[test]
    fn model_info_methods() {
        let model = BonsaiModel::new(Qwen3Config::bonsai_8b());
        assert_eq!(model.num_layers(), 36);
        assert_eq!(model.hidden_size(), 4096);
        assert_eq!(model.context_length(), 65536);
        assert!(model.num_parameters() > 0);
        assert!(model.model_size_bytes() > 0);
    }
    #[test]
    fn model_reset_cache() {
        let mut model = BonsaiModel::new(Qwen3Config::bonsai_8b());
        model.reset_cache();
        assert_eq!(model.kv_cache_mut().seq_len(), 0);
    }
    #[test]
    fn model_kv_cache_memory() {
        let model = BonsaiModel::new(Qwen3Config::bonsai_8b());
        assert!(model.kv_cache_memory_bytes() > 0);
    }
}

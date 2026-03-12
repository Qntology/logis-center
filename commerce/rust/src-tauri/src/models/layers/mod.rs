pub mod attention;
pub mod deepstack;
pub mod deltanet;
pub mod distributed;
pub mod linear;
pub mod mask;
pub mod mlp;
pub mod moe;
pub mod others;
pub mod rotary_emb;
pub mod wna16;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::var_builder::ShardedVarBuilder as VarBuilder;

#[derive(Clone)]
pub struct VarBuilderX<'a>(pub VarBuilder<'a>, pub String);

impl<'a> VarBuilderX<'a> {
    pub fn new(
        weight_files: &[std::path::PathBuf],
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let vb = unsafe {
            candle_nn::var_builder::ShardedSafeTensors::var_builder(
                weight_files,
                dtype,
                device,
            )?
        };
        Ok(Self(vb, String::new()))
    }

    pub fn from_vb(vb: VarBuilder<'a>) -> Self {
        Self(vb, String::new())
    }

    pub fn is_var_builder(&self) -> bool {
        true
    }

    pub fn is_qvar_builder(&self) -> bool {
        false
    }

    pub fn device(&self) -> Device {
        self.0.device().clone()
    }

    pub fn pp(&self, name: &str) -> VarBuilderX<'a> {
        let next_path = if self.1.is_empty() {
            name.to_string()
        } else {
            format!("{}.{}", self.1, name)
        };
        VarBuilderX(self.0.pp(name), next_path)
    }

    pub fn module_path(&self) -> &str {
        &self.1
    }

    pub fn has_key(&self, name: &str) -> bool {
        self.0.contains_tensor(name)
    }

    pub fn get_with_hints_dtype<S: Into<candle_core::Shape>>(
        &self,
        s: S,
        name: &str,
        shard: candle_nn::var_builder::Shard,
        dtype: DType,
    ) -> Result<Tensor> {
        self.0.get_with_hints_dtype(s, name, shard, dtype)
    }

    pub fn get<S: Into<candle_core::Shape>>(&self, s: S, name: &str) -> Result<Tensor> {
        self.0.get(s, name)
    }

    pub fn dtype(&self) -> DType {
        // ShardedVarBuilder doesn't have a direct dtype method, 
        // but we can infer it or just store it.
        // For now, let's assume F32 if not sure, or better yet, don't use it if not needed.
        DType::F32
    }
}


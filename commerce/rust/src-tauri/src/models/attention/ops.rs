use candle_core::shape::Dim;
use candle_core::{CpuStorage, CustomOp1, Error, Layout, Shape, WithDType};
use candle_core::{Result, Tensor};
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

pub struct NonZero {}

impl NonZero {
    fn nonzero_t<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Vec<u32> {
        let dims = layout.dims();
        let n = dims.len();
        let mut result = Vec::new();
        let mut indices = vec![0u32; n];
        for (i, v) in vs.iter().enumerate() {
            if !v.is_zero() {
                let mut idx = i;
                for (dim_index, dim) in dims.iter().enumerate().rev() {
                    let d = idx % dim;
                    indices[dim_index] = d as u32;
                    idx /= dim;
                }
                result.extend_from_slice(&indices);
            }
        }
        result
    }
}

impl CustomOp1 for NonZero {
    fn name(&self) -> &'static str { "nonzero" }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous() { return Err(Error::RequiresContiguous { op: "nonzero" }); }
        let result = match storage {
            CpuStorage::U8(vs) => self.nonzero_t(vs, layout),
            CpuStorage::U32(vs) => self.nonzero_t(vs, layout),
            CpuStorage::I64(vs) => self.nonzero_t(vs, layout),
            CpuStorage::BF16(vs) => self.nonzero_t(vs, layout),
            CpuStorage::F16(vs) => self.nonzero_t(vs, layout),
            CpuStorage::F32(vs) => self.nonzero_t(vs, layout),
            CpuStorage::F64(vs) => self.nonzero_t(vs, layout),
        };
        let index_len = layout.dims().len();
        let result_len = result.len() / index_len;
        Ok((CpuStorage::U32(result), Shape::from_dims(&[result_len, index_len])))
    }
}

pub trait NonZeroOp {
    fn nonzero(&self) -> Result<Tensor>;
}

impl NonZeroOp for Tensor {
    fn nonzero(&self) -> Result<Tensor> {
        let original_device = self.device();
        self.to_device(&candle_core::Device::Cpu)?
            .apply_op1_no_bwd(&NonZero {})?
            .to_device(original_device)
    }
}

pub trait SplitOp {
    fn split<D: Dim>(&self, splits: &[usize], dim: D) -> Result<Vec<Tensor>>;
    fn split2<D: Dim>(&self, splits: &[usize], dim: D) -> Result<(Tensor, Tensor)>;
}

impl SplitOp for Tensor {
    fn split<D: Dim>(&self, splits: &[usize], dim: D) -> Result<Vec<Tensor>> {
        let dim = dim.to_index(self.shape(), "split")?;
        let mut split_res = Vec::new();
        let mut index = 0;
        for split in splits {
            split_res.push(self.narrow(dim, index, *split)?);
            index += *split;
        }
        Ok(split_res)
    }

    fn split2<D: Dim>(&self, splits: &[usize], dim: D) -> Result<(Tensor, Tensor)> {
        let dim = dim.to_index(self.shape(), "split2")?;
        Ok((self.narrow(dim, 0, splits[0])?, self.narrow(dim, splits[0], splits[1])?))
    }
}

pub trait BincountOp {
    fn bincount(&self, minlength: u32) -> Result<Vec<u32>>;
}

impl BincountOp for Tensor {
    fn bincount(&self, minlength: u32) -> Result<Vec<u32>> {
        let values = self.to_vec1::<u32>()?;
        let max_val = values.par_iter().max().copied().unwrap_or(0);
        let result_len = (max_val + 1).max(minlength) as usize;
        let counts = values.par_iter().fold(|| vec![0u32; result_len], |mut acc, &v| {
            acc[v as usize] += 1; acc
        }).reduce(|| vec![0u32; result_len], |mut a, b| {
            for (i, v) in b.into_iter().enumerate() { a[i] += v; } a
        });
        Ok(counts)
    }
}

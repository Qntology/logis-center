//! 🌟 [EMBED-CPU-PAR] CPU 순전파의 원소별 연산을 **같은 식**으로 여러 코어에서 계산합니다.
//!
//! candle CPU 의 단항·이항 연산은 한 스레드에서 돌아, granite 임베딩 순전파에서 RmsNorm 의 `powf(2)` 와
//! MLP GLU 의 `silu(gate) * up`(+ narrow 복사 2회)이 3~4코어에서 25~30% 를 차지했습니다(macos-15 · 리눅스 VM
//! 프로파일). 원소마다 candle 과 같은 f32 식을 같은 순서로 계산하므로 결과 비트가 바뀌지 않습니다
//! (per-text · 배치 비트 동일과 텍스트 캐시 공간 그대로, tests/it/models_cpu.rs 가 단언).
//! 루프 본체는 dev 프로필에서도 opt-level 3 인 `embed-cpu-kernels` 에 있습니다.
//! CPU 가 아니거나 F32 연속이 아니면 원래 candle 연산을 그대로 씁니다.

use candle_core::{CpuStorage, DType, Result, Storage, Tensor, D};

fn with_cpu_f32<R>(x: &Tensor, f: impl FnOnce(&[f32]) -> R) -> Option<R> {
    if !x.device().is_cpu() || x.dtype() != DType::F32 || !x.is_contiguous() {
        return None;
    }
    let (s, l) = x.storage_and_layout();
    match &*s {
        Storage::Cpu(CpuStorage::F32(d)) => {
            let o = l.start_offset();
            Some(f(&d[o..o + x.elem_count()]))
        }
        _ => None,
    }
}

/// `x.powf(2.0)` — 같은 비트.
pub fn powf2(x: &Tensor) -> Result<Tensor> {
    let par = with_cpu_f32(x, |s| {
        let mut o = vec![0f32; s.len()];
        embed_cpu_kernels::powf2(s, &mut o);
        o
    });
    match par {
        Some(o) => Tensor::from_vec(o, x.shape().clone(), x.device()),
        None => x.powf(2.0),
    }
}

/// MLP GLU: `silu(x[.., :I]) * x[.., I:]` — 같은 비트, gate/up 복사 없음.
pub fn silu_mul_halves(x: &Tensor) -> Result<Tensor> {
    let last = x.dim(D::Minus1)?;
    let i = last / 2;
    let par = with_cpu_f32(x, |s| {
        let mut o = vec![0f32; s.len() / 2];
        embed_cpu_kernels::silu_mul_halves(s, i, &mut o);
        o
    });
    match par {
        Some(o) => {
            let mut dims = x.dims().to_vec();
            *dims.last_mut().unwrap() = i;
            Tensor::from_vec(o, dims, x.device())
        }
        None => {
            let gate = x.narrow(D::Minus1, 0, i)?.contiguous()?;
            let up = x.narrow(D::Minus1, i, i)?.contiguous()?;
            (candle_nn::ops::silu(&gate)? * up)?.contiguous()
        }
    }
}

/// 마지막 축 softmax. CPU F32 는 융합 `softmax_last_dim`(합성 softmax 와 같은 비트 — max · exp · 합 · 나눗셈
/// 순서가 같음, 실측 0/6.3M 차이 — 이고 긴 텍스트에서 3~4배 빠름), GPU 는 커널이 달라 기존 합성 연산 그대로.
pub fn softmax_last(x: &Tensor) -> Result<Tensor> {
    if x.device().is_cpu() && x.dtype() == DType::F32 && x.is_contiguous() {
        candle_nn::ops::softmax_last_dim(x)
    } else {
        candle_nn::ops::softmax(x, D::Minus1)
    }
}

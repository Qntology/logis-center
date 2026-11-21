//! Part 1 (Pack B): VLM 텐서 계층 1-Hop Halo 빈 병합 그룹 프루닝.
//!
//! 1차 SigLIP2 전체 페이지 스캔에는 절대 개입하지 않으며, 2차 VLM 비전 블록 루프
//! (`Qwen3VLVisionModel`) 진입 직전에만 `merge_size x merge_size` (28x28) 병합 그룹 단위로
//! 빈 종이 영역을 제거해 O(N^2) 비전 어텐션 연산을 절감합니다.
//!
//! - 3x3 Max-Pool (1-Hop Halo) 팽창으로 글자 외곽 획·소수점·쉼표·얇은 표 괘선 주변 1칸(28px)을
//!   100% 보호합니다.
//! - 살아남은 패치의 원본 `(t, row, col)` 위치 임베딩과 3D RoPE `(cos, sin)` 인덱스를 그대로
//!   보존하므로 공간 위치 정보가 전혀 훼손되지 않습니다.
//! - `total_groups <= 32` 이거나 생존 비율이 92% 이상이면 재인덱싱 오버헤드 없이 즉시 `Ok(None)` 을
//!   반환합니다.

use candle_core::{DType, Device, Result, Tensor};
use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PruneMode {
    Off,
    /// 비전 타워만 압축 실행 후 Merger 출력에서 원래 `H_m * W_m` 위치로 대표 공백 임베딩을 채워 복원.
    /// 기존 프롬프트 플레이스홀더 토큰 수(`n_image_tokens == H_m * W_m`)와 100% 호환됩니다.
    Halo1HopScatter,
    /// 언어 모델 Prefill 까지 압축 유지 (호출부가 원본 3D M-RoPE 위치 인덱스를 별도 전달할 때 사용).
    Halo1HopDrop,
}

impl PruneMode {
    pub fn tag(self) -> u8 {
        match self {
            PruneMode::Off => 0,
            PruneMode::Halo1HopScatter => 1,
            PruneMode::Halo1HopDrop => 2,
        }
    }
}

static PRUNE_MODE_OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// 테스트에서 동일 프로세스 내 `PruneMode` 를 강제 지정합니다 (`None` = 환경변수 복원).
pub fn set_vision_prune_override(mode: Option<PruneMode>) {
    let code = match mode {
        None => 0,
        Some(PruneMode::Off) => 1,
        Some(PruneMode::Halo1HopScatter) => 2,
        Some(PruneMode::Halo1HopDrop) => 3,
    };
    PRUNE_MODE_OVERRIDE.store(code, Ordering::Relaxed);
}

/// 환경변수 `LOGIS_VISION_PRUNE`:
/// - `off` / `0` / `false` / `no` -> `PruneMode::Off`
/// - `drop` -> `PruneMode::Halo1HopDrop`
/// - 기본값 / `scatter` / `1` / `on` / `halo` -> `PruneMode::Halo1HopScatter`
pub fn vision_prune_mode() -> PruneMode {
    match PRUNE_MODE_OVERRIDE.load(Ordering::Relaxed) {
        1 => return PruneMode::Off,
        2 => return PruneMode::Halo1HopScatter,
        3 => return PruneMode::Halo1HopDrop,
        _ => {}
    }
    static MODE: std::sync::OnceLock<PruneMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let v = std::env::var("LOGIS_VISION_PRUNE")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase();
        match v.as_str() {
            "0" | "off" | "false" | "no" => PruneMode::Off,
            "drop" | "halo_drop" => PruneMode::Halo1HopDrop,
            _ => PruneMode::Halo1HopScatter,
        }
    })
}

/// 정규화 화소(`[-1, 1]` 범위) 기준 기본 명암 폭 임계치.
/// 255 단계 중 약 3~4 계조 차이에 불과해 흐린 인쇄·소수점·얇은 괘선도 전부 활성으로 잡습니다.
pub const DEFAULT_SPREAD_THRESH: f32 = 0.03;

pub struct PrunedVisionPatches {
    /// 살아남은 패치 화소 `[N_kept * merge^2, C_patch]`
    pub kept_pixels: Tensor,
    /// 살아남은 패치들의 원본 패치 인덱스 `[N_kept * merge^2]` (U32, 원본 pos_embed & RoPE 보존)
    pub kept_rotary_idx: Tensor,
    /// 병합 그룹 단위 생존 인덱스 `[N_kept]` (U32, `0..orig_merge_groups`)
    pub merge_keep_indices: Tensor,
    /// `scatter_pruned_merge_groups` 용 `[orig_merge_groups]` gather 인덱스 (U32)
    pub scatter_gather_idx: Tensor,
    /// 원본 병합 그룹 총수 (`H_merge * W_merge`)
    pub orig_merge_groups: usize,
    /// 살아남은 병합 그룹 수 (`N_kept`, scatter 시 대표 공백 그룹 1개 포함)
    pub kept_merge_groups: usize,
    /// 대표 공백 그룹의 `merge_keep_indices` 내 위치 (있으면 `Some`)
    pub rep_blank_pos: Option<usize>,
}

/// 단일 이미지(`grid_thw.0 == 1`)의 병합 그룹(`merge_size x merge_size`) 중 명암 폭이
/// `spread_thresh` 이하인 빈 종이 그룹을 찾고, 3x3 Max-Pool (1-Hop Halo) 팽창으로
/// 콘텐츠 인접 8방위 그룹을 보호한 프루닝 계획을 생성합니다.
pub fn prune_blank_merge_groups_1hop(
    pixel_values: &Tensor,
    grid_thw: (u32, u32, u32),
    merge_size: usize,
    spread_thresh: f32,
) -> Result<Option<PrunedVisionPatches>> {
    let (t, h, w) = (grid_thw.0 as usize, grid_thw.1 as usize, grid_thw.2 as usize);
    if t != 1 || merge_size == 0 || h % merge_size != 0 || w % merge_size != 0 {
        return Ok(None);
    }
    let (hm, wm) = (h / merge_size, w / merge_size);
    let total_groups = hm * wm;

    // 소형 크롭(그룹 <= 32)은 재인덱싱 오버헤드 방지를 위해 즉시 우회합니다.
    if total_groups <= 32 {
        return Ok(None);
    }

    let merge_sq = merge_size * merge_size;
    let total_patches = total_groups * merge_sq;
    let (n_rows, c_patch) = match pixel_values.dims() {
        [n, c] if *n == total_patches => (*n, *c),
        _ => return Ok(None),
    };
    let _ = n_rows;

    // CPU F32 단일 스윕으로 각 병합 그룹의 (max - min) 명암 폭을 계산합니다.
    let flat = pixel_values
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let elems_per_group = merge_sq * c_patch;
    if flat.len() != total_groups * elems_per_group {
        return Ok(None);
    }

    let mut active = vec![false; total_groups];
    for g in 0..total_groups {
        let slice = &flat[g * elems_per_group..(g + 1) * elems_per_group];
        let mut mn = f32::INFINITY;
        let mut mx = f32::NEG_INFINITY;
        for &v in slice {
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
        if (mx - mn) > spread_thresh {
            active[g] = true;
        }
    }

    // 3x3 Max-Pool (1-Hop Halo): 활성 그룹의 상하좌우·대각선 8방위 그룹을 강제 보호합니다.
    let mut keep_mask = vec![false; total_groups];
    let mut kept_cnt = 0usize;
    for r in 0..hm {
        let r0 = r.saturating_sub(1);
        let r1 = (r + 1).min(hm - 1);
        for c in 0..wm {
            let c0 = c.saturating_sub(1);
            let c1 = (c + 1).min(wm - 1);
            let mut hit = false;
            'nbr: for nr in r0..=r1 {
                let row_off = nr * wm;
                for nc in c0..=c1 {
                    if active[row_off + nc] {
                        hit = true;
                        break 'nbr;
                    }
                }
            }
            if hit {
                keep_mask[r * wm + c] = true;
                kept_cnt += 1;
            }
        }
    }

    // 전체가 비어 있으면 최소 1개(좌상단)만 유지하지 말고 안전하게 우회합니다.
    if kept_cnt == 0 {
        return Ok(None);
    }

    // Scatter 복원 시 프루닝된 빈 그룹에 진짜 ViT 공백 임베딩을 채우기 위해
    // 첫 번째 빈 그룹 1개를 대표(representative blank)로 포함합니다.
    let first_blank = keep_mask.iter().position(|&k| !k);
    let Some(rep_blank_group) = first_blank else {
        return Ok(None);
    };
    let final_kept = kept_cnt + 1;

    // 92% 이상 생존 시 재인덱싱 실익이 없으므로 빠른 탈출합니다.
    if final_kept * 100 >= total_groups * 92 {
        return Ok(None);
    }

    let mut keep_groups: Vec<u32> = Vec::with_capacity(final_kept);
    let mut group_to_kept_pos = vec![0u32; total_groups];
    let mut rep_blank_pos = 0usize;

    for g in 0..total_groups {
        if keep_mask[g] || g == rep_blank_group {
            let pos = keep_groups.len();
            if g == rep_blank_group {
                rep_blank_pos = pos;
            }
            group_to_kept_pos[g] = pos as u32;
            keep_groups.push(g as u32);
        }
    }
    for g in 0..total_groups {
        if !keep_mask[g] && g != rep_blank_group {
            group_to_kept_pos[g] = rep_blank_pos as u32;
        }
    }

    let mut patch_indices: Vec<u32> = Vec::with_capacity(keep_groups.len() * merge_sq);
    for &g in &keep_groups {
        let base = g * (merge_sq as u32);
        for off in 0..(merge_sq as u32) {
            patch_indices.push(base + off);
        }
    }

    let dev = pixel_values.device();
    let kept_rotary_idx = Tensor::from_vec(patch_indices, (keep_groups.len() * merge_sq,), dev)?;
    let kept_pixels = pixel_values.index_select(&kept_rotary_idx, 0)?.contiguous()?;
    let merge_keep_indices = Tensor::from_vec(keep_groups, (final_kept,), dev)?;
    let scatter_gather_idx = Tensor::from_vec(group_to_kept_pos, (total_groups,), dev)?;

    Ok(Some(PrunedVisionPatches {
        kept_pixels,
        kept_rotary_idx,
        merge_keep_indices,
        scatter_gather_idx,
        orig_merge_groups: total_groups,
        kept_merge_groups: final_kept,
        rep_blank_pos: Some(rep_blank_pos),
    }))
}

/// `Halo1HopScatter` 모드에서 Merger / DeepStack 출력 `[N_kept, D]` 를 원래 격자 길이
/// `[orig_merge_groups, D]` 로 복원합니다 (빈 그룹은 대표 공백 그룹 임베딩으로 채움).
pub fn scatter_pruned_merge_groups(
    kept_merged: &Tensor,
    plan: &PrunedVisionPatches,
) -> Result<Tensor> {
    kept_merged
        .index_select(&plan.scatter_gather_idx, 0)?
        .contiguous()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_grid_and_dense_grid_fast_exit() -> Result<()> {
        let dev = Device::Cpu;
        // 4x4 merge groups = 16 <= 32 -> fast exit None
        let px_small = Tensor::zeros((16 * 4, 12), DType::F32, &dev)?;
        assert!(prune_blank_merge_groups_1hop(&px_small, (1, 8, 8), 2, DEFAULT_SPREAD_THRESH)?.is_none());

        // 8x8 merge groups = 64, all active -> >= 92% kept -> fast exit None
        let mut dense = vec![0.0f32; 64 * 4 * 12];
        for g in 0..64 {
            dense[g * 4 * 12] = 1.0;
        }
        let px_dense = Tensor::from_vec(dense, (64 * 4, 12), &dev)?;
        assert!(prune_blank_merge_groups_1hop(&px_dense, (1, 16, 16), 2, DEFAULT_SPREAD_THRESH)?.is_none());
        Ok(())
    }

    #[test]
    fn halo_1hop_protects_8_neighbors_and_scatters_exact_shape() -> Result<()> {
        let dev = Device::Cpu;
        // 10x10 merge groups = 100 groups (grid_thw = (1, 20, 20), merge_size = 2)
        // Put ink only in center group (r=5, c=5) -> 3x3 halo keeps 9 groups + 1 rep blank = 10 groups (90% pruned!)
        let mut data = vec![1.0f32; 100 * 4 * 8];
        let center = 5 * 10 + 5;
        data[center * 4 * 8] = -0.5;
        let px = Tensor::from_vec(data, (100 * 4, 8), &dev)?;
        let plan = prune_blank_merge_groups_1hop(&px, (1, 20, 20), 2, DEFAULT_SPREAD_THRESH)?
            .expect("sparse 10x10 grid should be pruned");

        assert_eq!(plan.orig_merge_groups, 100);
        assert_eq!(plan.kept_merge_groups, 10); // 9 halo + 1 rep blank
        assert_eq!(plan.kept_pixels.dims(), &[40, 8]);
        assert_eq!(plan.kept_rotary_idx.dims(), &[40]);

        let kept_groups = plan.merge_keep_indices.to_vec1::<u32>()?;
        for dr in 4u32..=6 {
            for dc in 4u32..=6 {
                let g = dr * 10 + dc;
                assert!(kept_groups.contains(&g), "1-hop halo neighbor ({dr},{dc}) must be kept");
            }
        }

        // Scatter restores [100, 6] with every blank group matching rep_blank_pos
        let fake_merged = Tensor::arange(0f32, 60.0, &dev)?.reshape((10, 6))?;
        let scattered = scatter_pruned_merge_groups(&fake_merged, &plan)?;
        assert_eq!(scattered.dims(), &[100, 6]);
        let rep = plan.rep_blank_pos.unwrap();
        let sc = scattered.to_vec2::<f32>()?;
        let fk = fake_merged.to_vec2::<f32>()?;
        // Corner (0,0) was blank -> must equal rep_blank embedding
        assert_eq!(sc[0], fk[rep]);
        // Center (5,5) must equal its kept position embedding
        let center_pos = kept_groups.iter().position(|&g| g == center as u32).unwrap();
        assert_eq!(sc[center], fk[center_pos]);
        Ok(())
    }
}

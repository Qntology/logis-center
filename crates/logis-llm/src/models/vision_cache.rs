use anyhow::{Result, anyhow};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;
use once_cell::sync::Lazy;

const CACHE_VERSION: u32 = 1;
/// 캐시 디렉터리 총량 상한 (기본 2GB)
const DEFAULT_MAX_BYTES: u64 = 2_000_000_000;

#[derive(Clone, Debug)]
pub struct CachedVisionEntry {
    pub dir: PathBuf,
    pub embed_dims: Vec<usize>,
    pub deepstack_count: usize,
    pub byte_size: u64,
    pub last_accessed: std::time::SystemTime,
}

pub struct VisionEmbedCache {
    cache_dir: PathBuf,
    index: RwLock<HashMap<u64, CachedVisionEntry>>,
    max_bytes: u64,
    pub hits: std::sync::atomic::AtomicUsize,
    pub misses: std::sync::atomic::AtomicUsize,
}

impl VisionEmbedCache {
    pub fn new(cache_dir: PathBuf, max_bytes: u64) -> Self {
        if !cache_dir.exists() {
            let _ = std::fs::create_dir_all(&cache_dir);
        }
        let cache = Self {
            cache_dir,
            index: RwLock::new(HashMap::new()),
            max_bytes,
            hits: std::sync::atomic::AtomicUsize::new(0),
            misses: std::sync::atomic::AtomicUsize::new(0),
        };
        cache.rebuild_index_from_disk();
        cache
    }

    /// 앱 재시작 후에도 기존 캐시를 재사용하기 위해 디스크를 스캔해 인덱스를 복원합니다.
    fn rebuild_index_from_disk(&self) {
        let entries = match std::fs::read_dir(&self.cache_dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        let mut idx = self.index.write().unwrap();
        let mut restored = 0usize;

        for e in entries.flatten() {
            let path = e.path();
            if !path.is_dir() { continue; }
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n,
                None => continue,
            };
            let key = match u64::from_str_radix(name, 16) {
                Ok(k) => k,
                Err(_) => continue,
            };

            let meta_path = path.join("meta.json");
            let meta_raw = match std::fs::read(&meta_path) {
                Ok(d) => d,
                Err(_) => continue,
            };
            let meta: serde_json::Value = match serde_json::from_slice(&meta_raw) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if meta.get("version").and_then(|v| v.as_u64()).unwrap_or(0) != CACHE_VERSION as u64 {
                // 스키마가 바뀐 캐시는 폐기합니다.
                let _ = std::fs::remove_dir_all(&path);
                continue;
            }

            let dims: Vec<usize> = meta.get("embed_dims")
                .and_then(|v| v.as_array())
                .map(|a| a.iter().filter_map(|x| x.as_u64().map(|u| u as usize)).collect())
                .unwrap_or_default();
            if dims.is_empty() { continue; }

            let ds_count = meta.get("deepstack_count").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
            let byte_size = Self::dir_size(&path);
            let accessed = e.metadata().and_then(|m| m.accessed()).unwrap_or(std::time::SystemTime::UNIX_EPOCH);

            idx.insert(key, CachedVisionEntry {
                dir: path,
                embed_dims: dims,
                deepstack_count: ds_count,
                byte_size,
                last_accessed: accessed,
            });
            restored += 1;
        }

        if restored > 0 {
            println!("[VISION-CACHE] Restored {} cached vision embeddings from disk.", restored);
        }
    }

    fn dir_size(path: &PathBuf) -> u64 {
        let mut total = 0u64;
        if let Ok(rd) = std::fs::read_dir(path) {
            for f in rd.flatten() {
                if let Ok(m) = f.metadata() {
                    if m.is_file() { total += m.len(); }
                }
            }
        }
        total
    }

    /// 🌟 pixel_values 의 전체 원시 바이트와 grid_thw · 프루닝 모드를 함께 해싱합니다.
    /// (BUG(13) 수정: 스트라이드 샘플링을 제거하고 전 원소를 혼합해 1픽셀 변경도 캐시 충돌 0건 보장)
    pub fn compute_key(pixel_values: &Tensor, grid_thw: &Tensor) -> Result<u64> {
        let mode = crate::models::common::vision_prune::vision_prune_mode();
        Self::compute_key_with_context(pixel_values, grid_thw, None, mode)
    }

    /// `(orig_w, orig_h)` 메타데이터와 `prune_mode` 까지 해시 시드에 혼합하는 확장 키 계산기.
    pub fn compute_key_with_context(
        pixel_values: &Tensor,
        grid_thw: &Tensor,
        orig_dims: Option<(u32, u32)>,
        prune_mode: crate::models::common::vision_prune::PruneMode,
    ) -> Result<u64> {
        use std::hash::Hasher;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        hasher.write_u32(CACHE_VERSION);
        hasher.write_u8(prune_mode.tag());
        if let Some((ow, oh)) = orig_dims {
            hasher.write_u8(1);
            hasher.write_u32(ow);
            hasher.write_u32(oh);
        } else {
            hasher.write_u8(0);
        }

        // shape / dtype
        for d in pixel_values.shape().dims() {
            hasher.write_usize(*d);
        }
        hasher.write(format!("{:?}", pixel_values.dtype()).as_bytes());

        // grid_thw (작으므로 전부 반영)
        let thw = grid_thw
            .to_device(&Device::Cpu)?
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        for v in &thw {
            hasher.write_u32(*v);
        }

        // pixel 본문: F32 로 통일해 CPU 로 내린 뒤 전 원소 100% 해싱 (스트라이드 누락 없음)
        let flat = pixel_values
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        hasher.write_usize(flat.len());
        // 빠른 64비트 블록 혼합 (FNV-1a / splitmix64 보조 누적)으로 1M float 도 1ms 이내 전량 해싱
        let mut h_mix: u64 = 0xcbf29ce484222325;
        let mut i = 0usize;
        let n = flat.len();
        while i + 2 <= n {
            let pair = (flat[i].to_bits() as u64) | ((flat[i + 1].to_bits() as u64) << 32);
            h_mix ^= pair;
            h_mix = h_mix.wrapping_mul(0x100000001b3);
            i += 2;
        }
        if i < n {
            h_mix ^= flat[i].to_bits() as u64;
            h_mix = h_mix.wrapping_mul(0x100000001b3);
        }
        hasher.write_u64(h_mix);

        Ok(hasher.finish())
    }

    /// 캐시 히트 시 image_embeds 와 deepstack_embeds 를 디스크에서 복원합니다.
    pub fn try_load(&self, key: u64, device: &Device, target_dtype: DType) -> Option<(Tensor, Vec<Tensor>)> {
        let entry = {
            let idx = self.index.read().unwrap();
            idx.get(&key).cloned()?
        };

        let embed_path = entry.dir.join("embeds.st");
        let data = std::fs::read(&embed_path).ok()?;
        let tensors = candle_core::safetensors::load_buffer(&data, &Device::Cpu).ok()?;
        let embeds_cpu = tensors.get("image_embeds")?.clone();
        let embeds = embeds_cpu.to_device(device).ok()?.to_dtype(target_dtype).ok()?;

        let mut deepstacks = Vec::with_capacity(entry.deepstack_count);
        for i in 0..entry.deepstack_count {
            let p = entry.dir.join(format!("deepstack_{}.st", i));
            let d = match std::fs::read(&p) { Ok(v) => v, Err(_) => return None };
            let t = match candle_core::safetensors::load_buffer(&d, &Device::Cpu) { Ok(v) => v, Err(_) => return None };
            let ds = t.get("deepstack")?.clone();
            deepstacks.push(ds.to_device(device).ok()?.to_dtype(target_dtype).ok()?);
        }

        // 접근 시각 갱신 (LRU)
        {
            let mut idx = self.index.write().unwrap();
            if let Some(e) = idx.get_mut(&key) {
                e.last_accessed = std::time::SystemTime::now();
            }
        }

        self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        println!(
            "[VISION-CACHE] HIT key={:016x} | embeds {:?} | deepstack {} | ViT 27-block attention skipped.",
            key, embeds.shape().dims(), entry.deepstack_count
        );
        Some((embeds, deepstacks))
    }

    /// ViT 연산 결과를 디스크에 저장합니다.
    pub fn save(&self, key: u64, embeds: &Tensor, deepstacks: &[Tensor]) -> Result<()> {
        let dir = self.cache_dir.join(format!("{:016x}", key));
        if !dir.exists() { std::fs::create_dir_all(&dir)?; }

        // 저장은 항상 F32 CPU 로 통일해 dtype 혼선을 없앱니다.
        let embeds_cpu = embeds.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.contiguous()?;
        let dims: Vec<usize> = embeds_cpu.shape().dims().to_vec();

        let mut map = HashMap::new();
        map.insert("image_embeds".to_string(), embeds_cpu);
        let tmp = dir.join("embeds.st.tmp");
        candle_core::safetensors::save(&map, &tmp)?;
        std::fs::rename(&tmp, dir.join("embeds.st"))?;

        for (i, ds) in deepstacks.iter().enumerate() {
            let ds_cpu = ds.to_device(&Device::Cpu)?.to_dtype(DType::F32)?.contiguous()?;
            let mut m = HashMap::new();
            m.insert("deepstack".to_string(), ds_cpu);
            let tmp_ds = dir.join(format!("deepstack_{}.st.tmp", i));
            candle_core::safetensors::save(&m, &tmp_ds)?;
            std::fs::rename(&tmp_ds, dir.join(format!("deepstack_{}.st", i)))?;
        }

        let meta = serde_json::json!({
            "version": CACHE_VERSION,
            "embed_dims": dims,
            "deepstack_count": deepstacks.len(),
        });
        std::fs::write(dir.join("meta.json"), serde_json::to_vec_pretty(&logis_schema::json_order::Sorted(&meta))?)?;

        let byte_size = Self::dir_size(&dir);
        {
            let mut idx = self.index.write().unwrap();
            idx.insert(key, CachedVisionEntry {
                dir,
                embed_dims: dims,
                deepstack_count: deepstacks.len(),
                byte_size,
                last_accessed: std::time::SystemTime::now(),
            });
        }

        self.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        println!("[VISION-CACHE] SAVED key={:016x} | {:.1} KB", key, byte_size as f64 / 1024.0);

        self.evict_if_needed();
        Ok(())
    }

    /// LRU 기반 정리. 총량이 max_bytes 를 넘으면 오래된 것부터 삭제합니다.
    pub fn evict_if_needed(&self) {
        let mut entries: Vec<(u64, CachedVisionEntry)> = {
            let idx = self.index.read().unwrap();
            idx.iter().map(|(k, v)| (*k, v.clone())).collect()
        };

        let total: u64 = entries.iter().map(|(_, e)| e.byte_size).sum();
        if total <= self.max_bytes { return; }

        entries.sort_by_key(|(_, e)| e.last_accessed);

        let mut freed = 0u64;
        let mut removed = 0usize;
        let mut idx = self.index.write().unwrap();
        for (k, e) in entries {
            if total - freed <= self.max_bytes { break; }
            let _ = std::fs::remove_dir_all(&e.dir);
            idx.remove(&k);
            freed += e.byte_size;
            removed += 1;
        }
        if removed > 0 {
            println!(
                "[VISION-CACHE] LRU evicted {} entries ({:.1} MB). Total now {:.1} MB / limit {:.1} MB.",
                removed,
                freed as f64 / 1e6,
                (total - freed) as f64 / 1e6,
                self.max_bytes as f64 / 1e6
            );
        }
    }

    pub fn stats(&self) -> (usize, usize) {
        (
            self.hits.load(std::sync::atomic::Ordering::Relaxed),
            self.misses.load(std::sync::atomic::Ordering::Relaxed),
        )
    }

    pub fn clear_all(&self) {
        let mut idx = self.index.write().unwrap();
        for (_, e) in idx.iter() {
            let _ = std::fs::remove_dir_all(&e.dir);
        }
        idx.clear();
        println!("[VISION-CACHE] All cached vision embeddings cleared.");
    }
}

/// 🌟 전역 싱글턴. 모델 인스턴스가 파기/재생성되어도 캐시는 살아남아야 의미가 있습니다.
pub static VISION_CACHE: Lazy<VisionEmbedCache> = Lazy::new(|| {
    let dir = crate::utils::get_app_dir().join("cache").join("vision_embeds");
    VisionEmbedCache::new(dir, DEFAULT_MAX_BYTES)
});

/// 🌟 [VISION-CACHE] 이미 초기화된 경우에만 캐시를 돌려줍니다 (퍼지의 통계 출력이 디스크 스캔을 유발하지 않게).
pub fn vision_cache_if_initialized() -> Option<&'static VisionEmbedCache> {
    Lazy::get(&VISION_CACHE)
}

/// 🌟 [SAFETY] 캐시 히트 결과가 실제 프롬프트의 이미지 토큰 수와 맞는지 검증합니다.
/// 불일치 시 Err 를 반환해 ViT 재계산으로 안전하게 폴백시킵니다.
pub fn validate_embed_shape(embeds: &Tensor, expected_tokens: usize) -> Result<()> {
    validate_embed_shape_for_mode(
        embeds,
        expected_tokens,
        crate::models::common::vision_prune::PruneMode::Off,
    )
}

/// 프루닝 모드(`Halo1HopDrop` 포함)를 인식하는 확장 형상 검증기.
/// - `Off` / `Halo1HopScatter`: `embeds.dim(0) == expected_tokens` 필수
/// - `Halo1HopDrop`: 압축된 토큰 수 `1 <= got <= expected_tokens` 허용
pub fn validate_embed_shape_for_mode(
    embeds: &Tensor,
    expected_tokens: usize,
    prune_mode: crate::models::common::vision_prune::PruneMode,
) -> Result<()> {
    let got = embeds.dim(0).map_err(|e| anyhow!("embeds dim0 read failed: {e}"))?;
    match prune_mode {
        crate::models::common::vision_prune::PruneMode::Halo1HopDrop => {
            if got == 0 || got > expected_tokens {
                return Err(anyhow!(
                    "cached pruned embeds token count out of bounds: got {} vs max {}",
                    got, expected_tokens
                ));
            }
        }
        _ => {
            if got != expected_tokens {
                return Err(anyhow!(
                    "cached embeds token mismatch: cached {} vs expected {}",
                    got, expected_tokens
                ));
            }
        }
    }
    Ok(())
}
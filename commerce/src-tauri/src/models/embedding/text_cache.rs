// =====================================================================
// 🌟 [EMBED-CACHE] 프로세스 전역 텍스트 임베딩 캐시 (퍼지 무관 · LRU · 고정 계층)
// ---------------------------------------------------------------------
//  ── 무엇이 문제였나 ──
//   임베딩 캐시는 LogisModel 필드(HashMap)였고, unload_embedding /
//   deep_purge_resources / Qwen3.5 릴레이가 매번 cache.clear() 를 했습니다.
//   검색은 시작과 끝에서 deep_purge 를 부르므로, 질의와 무관한 뱅크 구
//   (속성 구 1115개 · 필터 437개 · 동사 500개 · 레이아웃 294개 …)가
//   검색마다, 그리고 중간 스왑마다 처음부터 다시 임베딩되었습니다.
//   캐시는 CPU RAM 만 쓰고 VRAM 상주와 무관하므로 비워서 얻는 것이 없습니다.
//   상한 초과 시에도 '전체 비우기' 라 뱅크 벡터까지 한꺼번에 날아갔습니다.
//
//  ── 해결 ──
//   ① 프로세스 static 으로 옮겨 퍼지·LogisModel 재생성과 무관하게 유지합니다.
//   ② 키 = (공간 id, 텍스트). 공간 id 는 모델 지문(config.json 내용 ·
//      가중치/토크나이저 크기·수정시각 · 장치 dtype 계열) + 순전파 버전 태그의 해시입니다.
//      모델 파일이나 연산 경로(per-text / batched)가 바뀌면 다른 공간이 되어
//      서로 섞이지 않습니다.
//   ③ 뱅크 구는 '고정(pinned)' 계층에 넣어 퇴출하지 않고,
//      나머지(질의·아이템 텍스트)는 LRU 로 오래된 10% 씩만 퇴출합니다.
//   ④ 0벡터·비유한 벡터(실패 산물)는 절대 저장하지 않습니다.
//      영구 캐시에 0벡터가 들어가면 그 구가 채점에서 조용히 사라지기 때문입니다.
//
//  ── 정확도 ──
//   같은 공간 안에서는 '처음 계산된 벡터' 를 그대로 돌려줍니다.
//   현재 per-text 순전파는 결정론적이므로 다시 계산해도 같은 비트이고,
//   캐시는 재계산을 생략할 뿐 결과를 바꾸지 않습니다.
// =====================================================================
use candle_core::Device;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// per-text 순전파(오늘의 embed()) 결과 공간의 태그.
/// 순전파 수치가 바뀌는 변경을 하면 반드시 이 태그를 올려야 합니다.
pub const FORWARD_TAG_PER_TEXT: &str = "granite-r2/fwd-pertext/v1";
/// 배치 순전파(embed_many, 고정 모양 · CUDA/ROCm) 결과 공간의 태그.
pub const FORWARD_TAG_BATCHED: &str = "granite-r2/fwd-batched/v1";
/// 배치 순전파(embed_many, 동적 행 수 · CPU/Vulkan) 결과 공간의 태그.
pub const FORWARD_TAG_BATCHED_DYN: &str = "granite-r2/fwd-batched-dyn/v1";

/// LRU 계층 기본 상한(엔트리). 384×f32 = 1.5KB 이므로 약 38MB.
pub const DEFAULT_LRU_CAP: usize = 24_576;
/// 고정 계층 기본 상한(엔트리). 약 25MB. 넘치면 새 고정 요청은 LRU 로 들어갑니다.
pub const DEFAULT_PINNED_CAP: usize = 16_384;

/// 캐시에 넣어도 되는 벡터인지 판정합니다. (비어 있음 · 전부 0 · NaN/Inf 는 거부)
pub fn is_cacheable(v: &[f32]) -> bool {
    !v.is_empty() && v.iter().all(|x| x.is_finite()) && v.iter().any(|&x| x != 0.0)
}

struct Entry {
    vec: Arc<[f32]>,
    tick: u64,
    pinned: bool,
}

#[derive(Default)]
struct Inner {
    spaces: HashMap<u64, HashMap<String, Entry>>,
    tick: u64,
    lru_len: usize,
    pinned_len: usize,
    hits: u64,
    misses: u64,
    evicted: u64,
    rejected: u64,
}

/// 캐시 상태 스냅샷 (로그·테스트용).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CacheStats {
    pub entries: usize,
    pub pinned: usize,
    pub lru: usize,
    pub hits: u64,
    pub misses: u64,
    pub evicted: u64,
    pub rejected: u64,
}

pub struct TextEmbedCache {
    inner: Mutex<Inner>,
    lru_cap: usize,
    pinned_cap: usize,
}

impl TextEmbedCache {
    pub fn new(lru_cap: usize, pinned_cap: usize) -> Self {
        Self { inner: Mutex::new(Inner::default()), lru_cap: lru_cap.max(1), pinned_cap }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        // 다른 스레드가 패닉해도 캐시는 단순 데이터이므로 그대로 이어서 씁니다.
        self.inner.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// 단건 조회. 적중하면 LRU 시각을 갱신합니다.
    pub fn get(&self, space: u64, text: &str) -> Option<Arc<[f32]>> {
        let mut g = self.lock();
        g.tick += 1;
        let tick = g.tick;
        let hit = g
            .spaces
            .get_mut(&space)
            .and_then(|m| m.get_mut(text))
            .map(|e| {
                e.tick = tick;
                e.vec.clone()
            });
        if hit.is_some() {
            g.hits += 1;
        } else {
            g.misses += 1;
        }
        hit
    }

    /// 다건 조회. `pin` 이면 적중한 엔트리를 고정 계층으로 승격합니다(상한 안에서).
    pub fn get_many(&self, space: u64, texts: &[String], pin: bool) -> Vec<Option<Arc<[f32]>>> {
        let mut g = self.lock();
        let pinned_cap = self.pinned_cap;
        let mut out = Vec::with_capacity(texts.len());
        let (mut hits, mut misses, mut promoted) = (0u64, 0u64, 0usize);
        g.tick += 1;
        let tick = g.tick;
        let mut pinned_len = g.pinned_len;
        if let Some(m) = g.spaces.get_mut(&space) {
            for t in texts {
                match m.get_mut(t.as_str()) {
                    Some(e) => {
                        e.tick = tick;
                        if pin && !e.pinned && pinned_len < pinned_cap {
                            e.pinned = true;
                            pinned_len += 1;
                            promoted += 1;
                        }
                        hits += 1;
                        out.push(Some(e.vec.clone()));
                    }
                    None => {
                        misses += 1;
                        out.push(None);
                    }
                }
            }
        } else {
            misses = texts.len() as u64;
            out.resize(texts.len(), None);
        }
        g.pinned_len = pinned_len;
        g.lru_len -= promoted;
        g.hits += hits;
        g.misses += misses;
        out
    }

    /// 다건 적재. 새로 들어간 개수를 반환합니다.
    ///  - 0벡터·비유한 벡터는 거부합니다.
    ///  - 이미 있는 키는 '처음 계산된 벡터' 를 유지하고 시각/고정만 갱신합니다.
    ///  - `pinned` 요청이라도 고정 상한을 넘으면 LRU 계층에 들어갑니다.
    pub fn insert_many<I>(&self, space: u64, items: I, pinned: bool) -> usize
    where
        I: IntoIterator<Item = (String, Arc<[f32]>)>,
    {
        let mut g = self.lock();
        g.tick += 1;
        let tick = g.tick;
        let pinned_cap = self.pinned_cap;
        let mut inserted = 0usize;
        let mut rejected = 0u64;
        let (mut lru_len, mut pinned_len) = (g.lru_len, g.pinned_len);
        {
            let m = g.spaces.entry(space).or_default();
            for (text, vec) in items {
                if !is_cacheable(&vec) {
                    rejected += 1;
                    continue;
                }
                match m.get_mut(text.as_str()) {
                    Some(e) => {
                        e.tick = tick;
                        if pinned && !e.pinned && pinned_len < pinned_cap {
                            e.pinned = true;
                            pinned_len += 1;
                            lru_len -= 1;
                        }
                    }
                    None => {
                        let as_pinned = pinned && pinned_len < pinned_cap;
                        if as_pinned {
                            pinned_len += 1;
                        } else {
                            lru_len += 1;
                        }
                        m.insert(text, Entry { vec, tick, pinned: as_pinned });
                        inserted += 1;
                    }
                }
            }
        }
        g.spaces.retain(|_, m| !m.is_empty());
        g.lru_len = lru_len;
        g.pinned_len = pinned_len;
        g.rejected += rejected;
        if g.lru_len > self.lru_cap {
            Self::evict_locked(&mut g, self.lru_cap);
        }
        inserted
    }

    /// 단건 적재. 새로 들어갔으면 true.
    pub fn insert(&self, space: u64, text: String, vec: Arc<[f32]>, pinned: bool) -> bool {
        self.insert_many(space, std::iter::once((text, vec)), pinned) == 1
    }

    /// LRU 계층이 상한을 넘으면 가장 오래된 엔트리부터 상한의 90% 까지 퇴출합니다.
    /// 전체 비우기와 달리 최근에 쓴 벡터와 고정 계층은 그대로 남습니다.
    fn evict_locked(g: &mut Inner, lru_cap: usize) {
        let target = lru_cap - lru_cap / 10;
        if g.lru_len <= target {
            return;
        }
        let need = g.lru_len - target;
        // 키 문자열을 복제하지 않습니다: LRU 시각(u64)만 모아 need 번째 시각을 경계로 잡고,
        // 경계보다 오래된 것은 모두, 경계와 같은 시각은 모자란 만큼만 제자리에서 지웁니다.
        let mut ticks: Vec<u64> = Vec::with_capacity(g.lru_len);
        for m in g.spaces.values() {
            ticks.extend(m.values().filter(|e| !e.pinned).map(|e| e.tick));
        }
        if ticks.is_empty() {
            return;
        }
        let need = need.min(ticks.len());
        let (_, cut, _) = ticks.select_nth_unstable(need - 1);
        let cut = *cut;
        let older = ticks.iter().filter(|&&t| t < cut).count();
        drop(ticks);
        let mut ties_left = need - older;
        let mut removed = 0usize;
        for m in g.spaces.values_mut() {
            m.retain(|_, e| {
                if e.pinned || e.tick > cut {
                    return true;
                }
                if e.tick == cut {
                    if ties_left == 0 {
                        return true;
                    }
                    ties_left -= 1;
                }
                removed += 1;
                false
            });
        }
        g.spaces.retain(|_, m| !m.is_empty());
        g.lru_len -= removed;
        g.evicted += removed as u64;
    }

    /// 고정 계층은 남기고 LRU 계층만 비웁니다(RAM 비상용). 비운 개수를 반환합니다.
    pub fn drop_lru_tier(&self) -> usize {
        let mut g = self.lock();
        let mut removed = 0usize;
        for m in g.spaces.values_mut() {
            let before = m.len();
            m.retain(|_, e| e.pinned);
            removed += before - m.len();
        }
        g.spaces.retain(|_, m| !m.is_empty());
        g.lru_len = 0;
        g.evicted += removed as u64;
        removed
    }

    /// 한 공간을 통째로 버립니다(모델 파일이 바뀌어 지문이 달라졌을 때).
    pub fn forget_space(&self, space: u64) -> usize {
        let mut g = self.lock();
        let Some(m) = g.spaces.remove(&space) else { return 0 };
        let pinned = m.values().filter(|e| e.pinned).count();
        let n = m.len();
        g.pinned_len -= pinned;
        g.lru_len -= n - pinned;
        n
    }

    /// 전부 비웁니다. (테스트 전용 성격 — 앱 코드는 퍼지에서 이 함수를 부르지 않습니다)
    pub fn clear_all(&self) {
        let mut g = self.lock();
        *g = Inner::default();
    }

    pub fn len(&self) -> usize {
        let g = self.lock();
        g.lru_len + g.pinned_len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, space: u64, text: &str) -> bool {
        let g = self.lock();
        g.spaces.get(&space).map_or(false, |m| m.contains_key(text))
    }

    pub fn is_pinned(&self, space: u64, text: &str) -> bool {
        let g = self.lock();
        g.spaces.get(&space).and_then(|m| m.get(text)).map_or(false, |e| e.pinned)
    }

    pub fn stats(&self) -> CacheStats {
        let g = self.lock();
        CacheStats {
            entries: g.lru_len + g.pinned_len,
            pinned: g.pinned_len,
            lru: g.lru_len,
            hits: g.hits,
            misses: g.misses,
            evicted: g.evicted,
            rejected: g.rejected,
        }
    }
}

/// 🌟 granite 텍스트 임베딩 전역 캐시. 퍼지·모델 재로드·LogisModel 재생성과 무관하게 유지됩니다.
pub static GRANITE_TEXT_CACHE: Lazy<TextEmbedCache> =
    Lazy::new(|| TextEmbedCache::new(DEFAULT_LRU_CAP, DEFAULT_PINNED_CAP));

/// 퍼지 지점에서 호출합니다. 캐시는 비우지 않고 유지 사실만 남깁니다.
/// (과거의 embedding_cache.clear() 자리 — 캐시는 RAM 만 쓰므로 VRAM 반환과 무관합니다)
pub fn note_purge(reason: &str) {
    let s = GRANITE_TEXT_CACHE.stats();
    if s.entries > 0 {
        println!(
            "[EMBED-CACHE] 🛡️ {} — 임베딩 벡터 캐시는 퍼지와 무관하게 유지합니다 (총 {}건 · 고정 {}건 · LRU {}건 · 적중 {} / 미적중 {})",
            reason, s.entries, s.pinned, s.lru, s.hits, s.misses
        );
    }
}

// ---------------------------------------------------------------------
//  공간 id (모델 지문 + 순전파 태그)
// ---------------------------------------------------------------------

/// FNV-1a 64. 프로세스 간에도 같은 값을 내도록 고정 해시를 씁니다.
fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;

/// 장치 계열. 같은 가중치라도 BF16(GPU) 과 F32(CPU/Vulkan) 는 다른 벡터를 냅니다.
pub fn device_class(device: &Device) -> &'static str {
    if device.is_cuda() {
        "cuda-bf16"
    } else if device.is_rocm() {
        "rocm-bf16"
    } else if device.is_vulkan() {
        "vulkan-f32"
    } else if device.is_metal() {
        "metal-f32"
    } else if device.is_cpu() && super::cpu_packed::enabled() {
        // [EMBED-CPU-PACKED] 다른 커널(패킹 GEMM · 융합 RMSNorm)이라 벡터가 1e-6 수준 달라 공간을 나눕니다.
        "cpu-f32-packed"
    } else if device.is_cpu() {
        "cpu-f32"
    } else {
        "other"
    }
}

fn file_sig(h: u64, path: &Path) -> u64 {
    let name = path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let mut h = fnv1a(h, name.as_bytes());
    match std::fs::metadata(path) {
        Ok(m) => {
            h = fnv1a(h, &m.len().to_le_bytes());
            if let Ok(t) = m.modified() {
                if let Ok(d) = t.duration_since(std::time::UNIX_EPOCH) {
                    h = fnv1a(h, &d.as_nanos().to_le_bytes());
                }
            }
            h
        }
        Err(_) => fnv1a(h, b"<missing>"),
    }
}

/// 모델 지문: config.json 내용 + model.safetensors / tokenizer.json 의 (크기, 수정시각) + 장치 계열.
pub fn model_fingerprint(dir: &Path, device_class: &str) -> u64 {
    let mut h = FNV_OFFSET;
    h = fnv1a(h, dir.to_string_lossy().as_bytes());
    match std::fs::read(dir.join("config.json")) {
        Ok(b) => h = fnv1a(h, &b),
        Err(_) => h = fnv1a(h, b"<no-config>"),
    }
    h = file_sig(h, &dir.join("model.safetensors"));
    h = file_sig(h, &dir.join("tokenizer.json"));
    fnv1a(h, device_class.as_bytes())
}

/// 공간 id = hash(지문, 순전파 태그).
pub fn space_id(fingerprint: u64, forward_tag: &str) -> u64 {
    let h = fnv1a(FNV_OFFSET, &fingerprint.to_le_bytes());
    fnv1a(h, forward_tag.as_bytes())
}

static FINGERPRINTS: Lazy<Mutex<HashMap<(PathBuf, &'static str), u64>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// 지문을 프로세스 안에서 한 번만 계산합니다(캐시 조회 경로가 모델 로드 없이 동작하도록).
pub fn fingerprint_for(dir: &Path, class: &'static str) -> u64 {
    let key = (dir.to_path_buf(), class);
    if let Some(v) = FINGERPRINTS.lock().unwrap_or_else(|e| e.into_inner()).get(&key) {
        return *v;
    }
    let fp = model_fingerprint(dir, class);
    FINGERPRINTS.lock().unwrap_or_else(|e| e.into_inner()).insert(key, fp);
    fp
}

/// 모델을 (재)로드할 때 지문을 다시 계산합니다. 파일이 바뀌었으면 옛 공간을 버립니다.
pub fn refresh_fingerprint(dir: &Path, class: &'static str) -> u64 {
    let fp = model_fingerprint(dir, class);
    let old = FINGERPRINTS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert((dir.to_path_buf(), class), fp);
    if let Some(old) = old {
        if old != fp {
            let n = GRANITE_TEXT_CACHE.forget_space(space_id(old, FORWARD_TAG_PER_TEXT))
                + GRANITE_TEXT_CACHE.forget_space(space_id(old, FORWARD_TAG_BATCHED))
                + GRANITE_TEXT_CACHE.forget_space(space_id(old, FORWARD_TAG_BATCHED_DYN));
            println!(
                "[EMBED-CACHE] ♻️ 임베딩 모델 파일이 바뀌어 지문이 달라졌습니다. 옛 벡터 {}건을 버립니다.",
                n
            );
        }
    }
    fp
}

/// 이 장치에서 실제로 쓰이는 순전파 경로의 태그.
pub fn forward_tag(device: &Device) -> &'static str {
    if !super::batched_enabled(device) {
        FORWARD_TAG_PER_TEXT
    } else if super::BatchOpts::for_device(device).fixed_shape {
        FORWARD_TAG_BATCHED
    } else {
        FORWARD_TAG_BATCHED_DYN
    }
}

/// (모델 디렉터리, 장치) 의 캐시 공간 id.
pub fn space_for(dir: &Path, device: &Device) -> u64 {
    space_id(fingerprint_for(dir, device_class(device)), forward_tag(device))
}

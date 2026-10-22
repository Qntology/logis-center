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
            let n = GRANITE_TEXT_CACHE.forget_space(space_id_for_mode(old, FORWARD_TAG_PER_TEXT))
                + GRANITE_TEXT_CACHE.forget_space(space_id_for_mode(old, FORWARD_TAG_BATCHED))
                + GRANITE_TEXT_CACHE.forget_space(space_id_for_mode(old, FORWARD_TAG_BATCHED_DYN));
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

/// 순전파 판본까지 포함한 태그로 공간 id 를 만듭니다. Legacy 는 예전 태그 그대로라 예전 공간 id 와 같습니다.
fn space_id_for_mode(fingerprint: u64, tag: &str) -> u64 {
    match super::forward_mode().tag_suffix() {
        None => space_id(fingerprint, tag),
        Some(sfx) => space_id(fingerprint, &format!("{}{}", tag, sfx)),
    }
}

/// (모델 디렉터리, 장치) 의 캐시 공간 id.
/// 🌟 [EMBED-FORWARD] LOGIS_EMBED_FORWARD 로 고른 판본마다 다른 공간입니다(같은 텍스트의 두 벡터가 섞이지 않게).
pub fn space_for(dir: &Path, device: &Device) -> u64 {
    space_id_for_mode(fingerprint_for(dir, device_class(device)), forward_tag(device))
}

// =====================================================================
// 🌟 [EMBED-CACHE / DISK CHECKPOINT] 고정 계층(뱅크 구)을 공간별 파일로 남겨 다음 실행에서 복원합니다.
// ---------------------------------------------------------------------
//  ── 무엇이 문제였나 ──
//   SigLIP2 구(anchors.bin) · ViT 임베딩(VISION-CACHE) · 음역은 디스크에 남는데 granite 텍스트
//   임베딩은 프로세스 안에서만 살아서, 앱을 켤 때마다 bias.json 뱅크를 처음부터 다시 임베딩했습니다.
//   (무역 라벨 뱅크 2391구 = 200건 × 12회 ≈ 3.5초, 커머스 속성·필터·동사·프로브 뱅크 수 초)
//
//  ── 해결 ──
//   고정 계층에 새로 들어간 엔트리를 공간 id(모델 지문 + 순전파 태그) 별 파일에 덧붙이고, 그 공간을
//   처음 조회할 때 한 번 복원합니다. 고정 계층은 '질의와 무관한 bias.json 뱅크 구 전용' 이라는 계약이라
//   문서 내용이 디스크에 남지 않습니다. 공간 id 에 모델 파일 크기·수정시각 · 장치 계열 · 순전파 경로가
//   들어 있어 모델이나 경로가 바뀌면 다른 파일을 보므로 옛 벡터가 섞이지 않습니다.
//   복원한 벡터는 '같은 공간에서 처음 계산된 벡터' 이므로 프로세스 안 캐시와 의미가 같습니다.
//
//  ── 파일 형식 (덧붙이기 전용) ──
//   머리: magic "GTXT" u32 · version u32 · 0 u64
//   레코드: 텍스트 길이 u32 · 텍스트 UTF-8 · 차원 u32 · f32 × 차원 · 검사합 u64(텍스트 + 벡터 바이트 FNV-1a)
//   검사합이 어긋나거나 잘린 꼬리를 만나면 거기서 멈추고, 성한 앞부분만으로 파일을 다시 씁니다.
// =====================================================================

const DISK_MAGIC: u32 = 0x4754_5854; // "GTXT"
const DISK_VERSION: u32 = 1;
const DISK_HEAD: usize = 16;

/// `LOGIS_EMBED_DISK_CACHE=0` 이면 디스크 체크포인트를 쓰지 않습니다(프로세스 안 캐시만).
fn disk_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("LOGIS_EMBED_DISK_CACHE") {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "off" | "no"),
        Err(_) => true,
    })
}

#[derive(Default)]
struct DiskSpace {
    restored: bool,
    keys: std::collections::HashSet<String>,
}

static DISK: Lazy<Mutex<HashMap<u64, DiskSpace>>> = Lazy::new(|| Mutex::new(HashMap::new()));

fn disk_path(space: u64) -> PathBuf {
    crate::utils::get_app_dir()
        .join("cache")
        .join("granite_text")
        .join(format!("{:016x}.bin", space))
}

fn record_checksum(text: &[u8], vec: &[f32]) -> u64 {
    let mut h = fnv1a(FNV_OFFSET, text);
    for x in vec {
        h = fnv1a(h, &x.to_le_bytes());
    }
    h
}

/// 파일의 성한 레코드를 읽습니다. (레코드들, 성한 바이트 길이, 꼬리가 깨졌는지)
pub fn read_disk_records(path: &Path) -> (Vec<(String, Vec<f32>)>, usize, bool) {
    let buf = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return (Vec::new(), 0, false),
    };
    if buf.len() < DISK_HEAD {
        return (Vec::new(), 0, !buf.is_empty());
    }
    let u32_at = |o: usize| u32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]]);
    if u32_at(0) != DISK_MAGIC || u32_at(4) != DISK_VERSION {
        return (Vec::new(), 0, true);
    }
    let mut out: Vec<(String, Vec<f32>)> = Vec::new();
    let mut off = DISK_HEAD;
    let mut broken = false;
    while off < buf.len() {
        let parsed = (|| -> Option<(String, Vec<f32>, usize)> {
            let mut o = off;
            if o + 4 > buf.len() { return None; }
            let tl = u32_at(o) as usize;
            o += 4;
            if o + tl > buf.len() { return None; }
            let text = std::str::from_utf8(&buf[o..o + tl]).ok()?.to_string();
            o += tl;
            if o + 4 > buf.len() { return None; }
            let dim = u32_at(o) as usize;
            o += 4;
            if dim == 0 || dim > 65_536 || o + dim * 4 + 8 > buf.len() { return None; }
            let vec: Vec<f32> = (0..dim)
                .map(|k| {
                    let p = o + k * 4;
                    f32::from_le_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]])
                })
                .collect();
            o += dim * 4;
            let mut ck = [0u8; 8];
            ck.copy_from_slice(&buf[o..o + 8]);
            o += 8;
            if u64::from_le_bytes(ck) != record_checksum(text.as_bytes(), &vec) { return None; }
            Some((text, vec, o))
        })();
        match parsed {
            Some((text, vec, next)) => {
                out.push((text, vec));
                off = next;
            }
            None => {
                broken = true;
                break;
            }
        }
    }
    (out, off, broken)
}

fn encode_disk_records(items: &[(String, Arc<[f32]>)]) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    for (text, vec) in items {
        let t = text.as_bytes();
        buf.extend_from_slice(&(t.len() as u32).to_le_bytes());
        buf.extend_from_slice(t);
        buf.extend_from_slice(&(vec.len() as u32).to_le_bytes());
        for x in vec.iter() {
            buf.extend_from_slice(&x.to_le_bytes());
        }
        buf.extend_from_slice(&record_checksum(t, vec).to_le_bytes());
    }
    buf
}

fn disk_head() -> Vec<u8> {
    let mut h = Vec::with_capacity(DISK_HEAD);
    h.extend_from_slice(&DISK_MAGIC.to_le_bytes());
    h.extend_from_slice(&DISK_VERSION.to_le_bytes());
    h.extend_from_slice(&0u64.to_le_bytes());
    h
}

/// 레코드를 파일 끝에 덧붙입니다(파일이 없으면 머리부터 씁니다).
pub fn append_disk_records(path: &Path, items: &[(String, Arc<[f32]>)]) -> std::io::Result<()> {
    use std::io::Write;
    if items.is_empty() {
        return Ok(());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let is_new = std::fs::metadata(path).map(|m| m.len() == 0).unwrap_or(true);
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    let mut buf = if is_new { disk_head() } else { Vec::new() };
    buf.extend_from_slice(&encode_disk_records(items));
    f.write_all(&buf)
}

/// 이 공간의 고정 계층을 디스크에서 한 번 복원합니다. 복원해 넣은 건수를 돌려줍니다.
pub fn restore_pinned(space: u64) -> usize {
    if !disk_enabled() {
        return 0;
    }
    let mut disk = DISK.lock().unwrap_or_else(|e| e.into_inner());
    let entry = disk.entry(space).or_default();
    if entry.restored {
        return 0;
    }
    entry.restored = true;
    let path = disk_path(space);
    let (records, good_len, broken) = read_disk_records(&path);
    if broken {
        // 잘린 꼬리 뒤에 덧붙이면 그 뒤 레코드는 영영 읽히지 않으므로 성한 앞부분만으로 다시 씁니다.
        let items: Vec<(String, Arc<[f32]>)> = records
            .iter()
            .map(|(t, v)| (t.clone(), Arc::from(v.as_slice())))
            .collect();
        let mut bytes = disk_head();
        bytes.extend_from_slice(&encode_disk_records(&items));
        let tmp = path.with_extension("bin.tmp");
        let ok = std::fs::write(&tmp, &bytes).and_then(|_| std::fs::rename(&tmp, &path)).is_ok();
        if !ok {
            let _ = std::fs::remove_file(&tmp);
            let _ = std::fs::remove_file(&path);
        }
        println!(
            "[EMBED-CACHE] 🩹 [DISK CHECKPOINT] {} 의 꼬리가 깨져 있어 성한 레코드 {}건({}바이트)만 남겼습니다.",
            path.display(),
            records.len(),
            good_len
        );
    }
    if records.is_empty() {
        return 0;
    }
    let mut items: Vec<(String, Arc<[f32]>)> = Vec::with_capacity(records.len());
    for (t, v) in records.into_iter() {
        entry.keys.insert(t.clone());
        items.push((t, Arc::from(v.as_slice())));
    }
    drop(disk);
    let n = items.len();
    let inserted = GRANITE_TEXT_CACHE.insert_many(space, items, true);
    println!(
        "[EMBED-CACHE] 💽 [DISK CHECKPOINT] 뱅크 구 {}건을 디스크에서 복원했습니다 (새로 적재 {}건) | {}",
        n,
        inserted,
        path.display()
    );
    inserted
}

/// 고정 계층에 있는 뱅크 구 중 디스크에 아직 없는 것을 덧붙입니다.
/// (새로 계산된 구뿐 아니라 LRU 에서 고정으로 승격된 구도 함께 남깁니다)
pub fn persist_pinned<'a, I>(space: u64, items: I)
where
    I: IntoIterator<Item = (&'a str, &'a [f32])>,
{
    if !disk_enabled() {
        return;
    }
    let fresh: Vec<(String, Arc<[f32]>)> = {
        let mut disk = DISK.lock().unwrap_or_else(|e| e.into_inner());
        let entry = disk.entry(space).or_default();
        let mut fresh: Vec<(String, Arc<[f32]>)> = Vec::new();
        for (t, v) in items {
            if entry.keys.len() >= DEFAULT_PINNED_CAP {
                break;
            }
            if entry.keys.contains(t) || !is_cacheable(v) || !GRANITE_TEXT_CACHE.is_pinned(space, t) {
                continue;
            }
            entry.keys.insert(t.to_string());
            fresh.push((t.to_string(), Arc::from(v)));
        }
        fresh
    };
    if fresh.is_empty() {
        return;
    }
    if let Err(e) = append_disk_records(&disk_path(space), &fresh) {
        println!("[EMBED-CACHE] ⚠️ [DISK CHECKPOINT] 뱅크 구 {}건 기록 실패(메모리 캐시는 유지): {}", fresh.len(), e);
    }
}

#[cfg(test)]
mod disk_checkpoint_tests {
    use super::*;

    #[test]
    fn records_roundtrip_and_a_torn_tail_is_cut() {
        let dir = std::env::temp_dir().join(format!("gtxt_{}_{}", std::process::id(), chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)));
        let path = dir.join("space.bin");
        let a: Vec<(String, Arc<[f32]>)> = vec![
            ("issue date".to_string(), Arc::from(vec![0.25f32, -1.5, 3.0].as_slice())),
            ("발행일".to_string(), Arc::from(vec![1.0f32, 0.0, -0.125].as_slice())),
        ];
        append_disk_records(&path, &a).unwrap();
        append_disk_records(&path, &[("B/L No.".to_string(), Arc::from(vec![9.0f32, 8.0, 7.0].as_slice()))]).unwrap();
        let (recs, good, broken) = read_disk_records(&path);
        assert!(!broken);
        assert_eq!(good as u64, std::fs::metadata(&path).unwrap().len());
        assert_eq!(recs.len(), 3);
        assert_eq!(recs[1].0, "발행일");
        assert_eq!(recs[2].1, vec![9.0f32, 8.0, 7.0]);

        // 쓰다 끊긴 레코드(꼬리 5바이트만)는 성한 앞부분까지만 읽습니다.
        {
            use std::io::Write;
            let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
            f.write_all(&[7, 0, 0, 0, b'x']).unwrap();
        }
        let (recs2, good2, broken2) = read_disk_records(&path);
        assert!(broken2);
        assert_eq!(recs2.len(), 3);
        assert_eq!(good2, good);

        // 검사합이 어긋난 레코드도 받아들이지 않습니다.
        let mut bytes = std::fs::read(&path).unwrap();
        let last_f32 = good - 8 - 4;
        bytes[last_f32] ^= 0x01;
        std::fs::write(&path, &bytes[..good]).unwrap();
        let (recs3, _, broken3) = read_disk_records(&path);
        assert!(broken3);
        assert_eq!(recs3.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

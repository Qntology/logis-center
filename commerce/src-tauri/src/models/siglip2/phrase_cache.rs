// =====================================================================
// 🌟 [SIGLIP2 PHRASE CACHE] 앵커 구 임베딩 2단 영구 캐시
// ---------------------------------------------------------------------
//  ── 왜 필요한가 ──
//   SigLIP2 텍스트 인코더는 27층 × 64토큰 × 1152차원입니다.
//   구 하나를 벡터로 만드는 데 약 26 GFLOP 이 듭니다.
//   그런데 우리가 인코딩하는 문자열은 전부 정적입니다.
//     · logic.rs 의 TRADE_GROUPS / TRADE_GROUP_CODES / TRADE_DOC_TITLES
//     · logic.rs 의 VISION_CHROME_ANCHOR
//     · bias.json 의 trade_schema 필드 semantic / bias
//   실측 문서 하나가 uniq 345구를 인코딩하므로 약 9 TFLOP,
//   그리고 그 계산을 위해 1.4GB 짜리 텍스트 인코더를 VRAM 에 올려야 합니다.
//   같은 문자열을 두 번째 문서에서 또 계산하는 것은 순수 낭비입니다.
//
//  ── 왜 디스크까지 가는가 ──
//   scheduler.rs 의 음차 캐시가 (메모리 → Dexie) 2단인 것과 같은 이유입니다.
//   앱을 재시작해도 앵커 사전은 바뀌지 않으므로, 프로세스 경계를 넘어 살아남아야
//   '두 번째 실행부터는 텍스트 인코더를 아예 올리지 않는' 상태가 됩니다.
//   vision_cache.rs 가 Qwen ViT 결과를 디스크에 굽는 것과 같은 계열입니다.
//
//  ── 정밀도 ──
//   저장하는 값은 encode_phrases 가 만든 L2 정규화 f32 벡터 원본 그대로입니다.
//   양자화도 차원 축소도 하지 않으므로 캐시 히트와 재계산의 결과가 비트 단위로 같습니다.
//
//  ── 용량 ──
//   구 하나 = 8B(키) + 1152 × 4B = 4,616B.
//   실측 345구 = 1.6MB. 상한 50,000구를 잡아도 231MB 입니다.
// =====================================================================
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

const CACHE_MAGIC: u32 = 0x5347_4C32; // "SGL2"
const CACHE_VERSION: u32 = 1;
const EMBED_DIM: usize = 1152;
/// 디스크 레코드 상한. 앵커 사전은 수백 구 규모이므로 사실상 도달하지 않습니다.
const MAX_RECORDS: usize = 50_000;

pub struct PhraseCache {
    dim: usize,
    path: PathBuf,
    mem: RwLock<HashMap<u64, Arc<Vec<f32>>>>,
    hits: AtomicUsize,
    misses: AtomicUsize,
    loaded: RwLock<bool>,
}

impl PhraseCache {
    fn new(path: PathBuf, dim: usize) -> Self {
        Self {
            dim,
            path,
            mem: RwLock::new(HashMap::new()),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            loaded: RwLock::new(false),
        }
    }

    /// 🌟 캐시 키. 토크나이저의 canonicalize 는 결정론이므로 원문만 해싱해도 안전합니다.
    ///    dim / version 을 함께 넣어 모델 교체 시 자동 무효화됩니다.
    pub fn key_of(phrase: &str) -> u64 {
        use std::hash::Hasher;
        let mut h = std::collections::hash_map::DefaultHasher::new();
        h.write_u32(CACHE_VERSION);
        h.write_usize(EMBED_DIM);
        h.write(phrase.as_bytes());
        h.write_u8(0xFF);
        h.finish()
    }

    fn ensure_loaded(&self) {
        {
            if *self.loaded.read().unwrap() {
                return;
            }
        }
        let mut loaded = self.loaded.write().unwrap();
        if *loaded {
            return;
        }
        *loaded = true;

        let mut f = match std::fs::File::open(&self.path) {
            Ok(f) => f,
            Err(_) => return,
        };
        let mut head = [0u8; 16];
        if f.read_exact(&mut head).is_err() {
            return;
        }
        let magic = u32::from_le_bytes([head[0], head[1], head[2], head[3]]);
        let ver = u32::from_le_bytes([head[4], head[5], head[6], head[7]]);
        let dim = u32::from_le_bytes([head[8], head[9], head[10], head[11]]) as usize;
        if magic != CACHE_MAGIC || ver != CACHE_VERSION || dim != self.dim {
            // 스키마가 다르면 통째로 폐기합니다. 잘못된 벡터를 쓰는 것보다 재계산이 안전합니다.
            let _ = std::fs::remove_file(&self.path);
            println!("[SIGLIP2-PHRASE] 캐시 스키마 불일치 → 폐기 후 재생성합니다.");
            return;
        }

        let rec = 8 + self.dim * 4;
        let mut buf = Vec::new();
        if f.read_to_end(&mut buf).is_err() {
            return;
        }
        let mut map = self.mem.write().unwrap();
        let mut n = 0usize;
        let mut off = 0usize;
        while off + rec <= buf.len() {
            let key = u64::from_le_bytes([
                buf[off], buf[off + 1], buf[off + 2], buf[off + 3],
                buf[off + 4], buf[off + 5], buf[off + 6], buf[off + 7],
            ]);
            let mut v = Vec::with_capacity(self.dim);
            let base = off + 8;
            for k in 0..self.dim {
                let p = base + k * 4;
                v.push(f32::from_le_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]]));
            }
            map.insert(key, Arc::new(v));
            n += 1;
            off += rec;
        }
        println!(
            "[SIGLIP2-PHRASE] 디스크 캐시 복원: 구 {}개 ({:.1} MB) | {:?}",
            n,
            (n * rec) as f64 / 1e6,
            self.path
        );
    }

    pub fn get(&self, phrase: &str) -> Option<Arc<Vec<f32>>> {
        self.ensure_loaded();
        let k = Self::key_of(phrase);
        let hit = self.mem.read().unwrap().get(&k).cloned();
        match hit {
            Some(v) => {
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(v)
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// 새로 계산한 구를 메모리에 넣고 디스크에 append 합니다.
    pub fn put_batch(&self, items: &[(String, Arc<Vec<f32>>)]) {
        if items.is_empty() {
            return;
        }
        self.ensure_loaded();

        let mut fresh: Vec<(u64, &Arc<Vec<f32>>)> = Vec::with_capacity(items.len());
        {
            let mut map = self.mem.write().unwrap();
            if map.len() >= MAX_RECORDS {
                println!(
                    "[SIGLIP2-PHRASE] 캐시가 상한 {}구에 도달했습니다. 디스크 기록을 중단합니다.",
                    MAX_RECORDS
                );
                return;
            }
            for (p, v) in items.iter() {
                if v.len() != self.dim {
                    continue;
                }
                let k = Self::key_of(p);
                if map.contains_key(&k) {
                    continue;
                }
                map.insert(k, v.clone());
                fresh.push((k, v));
            }
        }
        if fresh.is_empty() {
            return;
        }

        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let is_new = !self.path.exists();
        let mut f = match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        {
            Ok(f) => f,
            Err(e) => {
                println!("[SIGLIP2-PHRASE] 디스크 기록 실패(메모리 캐시는 유지): {}", e);
                return;
            }
        };
        if is_new {
            let mut head = [0u8; 16];
            head[0..4].copy_from_slice(&CACHE_MAGIC.to_le_bytes());
            head[4..8].copy_from_slice(&CACHE_VERSION.to_le_bytes());
            head[8..12].copy_from_slice(&(self.dim as u32).to_le_bytes());
            let _ = f.write_all(&head);
        }
        let mut buf: Vec<u8> = Vec::with_capacity(fresh.len() * (8 + self.dim * 4));
        for (k, v) in fresh.iter() {
            buf.extend_from_slice(&k.to_le_bytes());
            for x in v.iter() {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        let _ = f.write_all(&buf);
        println!(
            "[SIGLIP2-PHRASE] 신규 구 {}개 디스크 기록 ({:.1} KB)",
            fresh.len(),
            buf.len() as f64 / 1024.0
        );
    }

    /// 🌟 [LOAD GATE] 이 구 목록이 전부 캐시에 있으면 텍스트 인코더를 올릴 필요가 없습니다.
    pub fn all_cached(&self, phrases: &[String]) -> bool {
        self.ensure_loaded();
        let map = self.mem.read().unwrap();
        phrases.iter().all(|p| map.contains_key(&Self::key_of(p)))
    }

    pub fn stats(&self) -> (usize, usize, usize) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
            self.mem.read().unwrap().len(),
        )
    }

    pub fn clear_all(&self) {
        self.mem.write().unwrap().clear();
        let _ = std::fs::remove_file(&self.path);
        println!("[SIGLIP2-PHRASE] 캐시를 전량 삭제했습니다.");
    }
}

/// 🌟 전역 싱글턴. 모델 인스턴스가 파기/재생성되어도 캐시는 살아남아야 의미가 있습니다.
///    (vision_cache.rs 의 VISION_CACHE 와 같은 수명 정책)
pub static SIGLIP2_PHRASE_CACHE: Lazy<PhraseCache> = Lazy::new(|| {
    let p = crate::utils::get_app_dir()
        .join("cache")
        .join("siglip2_phrases")
        .join("anchors.bin");
    PhraseCache::new(p, EMBED_DIM)
});
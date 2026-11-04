//! 🌟 [RESIDENCY] 생성 모델 ↔ 임베딩 모델 교차 상주의 상태 · 실측 · 전환 구간
//!
//! 예전에는 전역 static 9개(`CROSSOVER_PHASE` · `EMBED_RESIDENT_MB` · `GEN_RESIDENT_MB` · `GEN_COST_BY_KIND` ·
//! `PEAK_RESIDENT_MB` · `SWAP_COUNT` · `COEXIST_COUNT` · `LAST_SWAP_MS` · `ACTIVATION_HEADROOM_MB`)와
//! `AtomicU32` 홀드 카운터 + 120ms 폴링이 이 일을 나눠 맡았습니다. 여기서 넷으로 묶습니다.
//!
//! - [`Residency`] : 지금 무엇이 올라와 있는가. **슬롯에서만 유도하고 따로 적어 두지 않습니다.**
//!   예전 원장(`CROSSOVER_PHASE`)은 단계 퍼지가 예산 안이라 임베딩을 남긴 뒤에도 `mark_crossover_idle` 이
//!   IDLE 로 덮어써 사실과 어긋났고, 피크도 그 원장으로 셌습니다.
//! - [`CostBook`] : 모델 파일 × 장치의 실측 비용과 전환 통계. 프로세스에 하나이며 `LogisModel` 을 새로 만들어도
//!   유지됩니다 (실측은 인스턴스가 아니라 장치와 파일의 성질입니다). 예산 · 측정 규칙이 메서드라 장치 없이 시험합니다.
//! - [`HoldGate`] : 생성 모델 전환 구간. 마지막 가드([`GenerationHold`])가 풀리는 순간 기다리던 임베딩 로드를 깨웁니다.
//! - [`SlotLease`] : 슬롯을 쥔 채 계산 스레드로 넘기는 가드. 쥐고 있는 동안 아무도 그 모델을 내릴 수 없습니다.
//!
//! 모델 객체의 소유는 지금처럼 `LogisModel` 의 슬롯이 가집니다. 슬롯이 사실이고, 이 모듈은 그 사실에서 상태를 읽습니다.

use std::fmt;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{Notify, OwnedMutexGuard};

use crate::models::embedding::EmbeddingModel;

/// 생성 모델 종류 수: `LogisModel::gen_kind` 의 0 (Qwen 0.6B) · 1 (Qwen3) · 2 (Qwen3.5-2B) 와 언어별 4B.
pub const GEN_KINDS: usize = 4;
/// 언어별 Qwen3.5-4B 의 실측 칸.
pub const GEN_KIND_LANG4B: usize = 3;

/// 지금 장치에 무엇이 상주하는가.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    /// 아무것도 상주하지 않음 (퍼지 직후)
    Idle,
    /// 임베딩 모델만 상주
    Embedding,
    /// 생성 모델만 상주
    Generation,
    /// 둘 다 상주 (예산이 허용한 경우에만)
    Coexist,
}

impl Residency {
    pub fn from_slots(generation: bool, embedding: bool) -> Self {
        match (generation, embedding) {
            (true, true) => Self::Coexist,
            (true, false) => Self::Generation,
            (false, true) => Self::Embedding,
            (false, false) => Self::Idle,
        }
    }

    pub fn has_generation(self) -> bool {
        matches!(self, Self::Generation | Self::Coexist)
    }

    pub fn has_embedding(self) -> bool {
        matches!(self, Self::Embedding | Self::Coexist)
    }
}

impl fmt::Display for Residency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Idle => "IDLE",
            Self::Embedding => "EMBEDDING",
            Self::Generation => "GENERATION",
            Self::Coexist => "COEXIST",
        })
    }
}

/// 상주 비용 실측(MB)과 전환 통계. 모든 값이 원자값이라 잠금 없이 읽고 씁니다.
pub struct CostBook {
    /// 임베딩 가중치 (로드 전후 free 차이)
    embed_mb: AtomicU64,
    /// 마지막으로 깨끗하게 잰 생성 모델 (종류 무관)
    gen_mb: AtomicU64,
    /// 생성 모델 종류별 실측
    gen_by_kind: [AtomicU64; GEN_KINDS],
    /// 연산 중 가중치 위에 더 쓴 양의 최대값 (activation)
    activation_headroom_mb: AtomicU64,
    /// 상주 상태가 차지한 가중치의 최대값
    peak_mb: AtomicU64,
    swaps: AtomicU64,
    /// 동시 상주로 스왑을 피한 횟수
    coexists: AtomicU64,
    /// 마지막 스왑 시각 (epoch ms). 왕복 진단용.
    last_swap_ms: AtomicU64,
}

static PROCESS_COSTS: CostBook = CostBook::new();

impl Default for CostBook {
    fn default() -> Self {
        Self::new()
    }
}

impl CostBook {
    pub const fn new() -> Self {
        Self {
            embed_mb: AtomicU64::new(0),
            gen_mb: AtomicU64::new(0),
            gen_by_kind: [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)],
            activation_headroom_mb: AtomicU64::new(0),
            peak_mb: AtomicU64::new(0),
            swaps: AtomicU64::new(0),
            coexists: AtomicU64::new(0),
            last_swap_ms: AtomicU64::new(0),
        }
    }

    /// 프로세스에 하나뿐인 장부. 모든 `LogisModel` 이 이것을 씁니다.
    pub fn process() -> &'static CostBook {
        &PROCESS_COSTS
    }

    pub fn embedding_mb(&self) -> u64 {
        self.embed_mb.load(Ordering::SeqCst)
    }

    /// 이 종류의 생성 모델 실측. 모르는 종류나 아직 못 잰 종류는 0 입니다.
    pub fn generation_mb(&self, kind: usize) -> u64 {
        self.gen_by_kind.get(kind).map_or(0, |c| c.load(Ordering::SeqCst))
    }

    pub fn last_generation_mb(&self) -> u64 {
        self.gen_mb.load(Ordering::SeqCst)
    }

    pub fn activation_headroom_mb(&self) -> u64 {
        self.activation_headroom_mb.load(Ordering::SeqCst)
    }

    pub fn peak_mb(&self) -> u64 {
        self.peak_mb.load(Ordering::SeqCst)
    }

    pub fn swap_count(&self) -> u64 {
        self.swaps.load(Ordering::SeqCst)
    }

    pub fn coexist_count(&self) -> u64 {
        self.coexists.load(Ordering::SeqCst)
    }

    /// 임베딩을 올리는 데 필요한 VRAM(MB) = 실측(없으면 디스크 추정) + activation 여유.
    /// 디스크 추정은 실측이 없을 때만 계산합니다 (가중치 폴더를 훑는 일이라).
    pub fn embedding_budget_mb(&self, disk_mb: impl FnOnce() -> u64) -> u64 {
        let base = match self.embedding_mb() {
            0 => disk_mb().max(1),
            measured => measured,
        };
        base + self.activation_headroom_mb()
    }

    /// 생성 모델을 올리는 데 필요한 VRAM(MB) = max(디스크, 실측) + activation 여유.
    /// 이 종류의 실측이 없으면 마지막으로 잰 생성 모델 값을 씁니다. 디스크 크기가 바닥이라 과소평가로 가지 않습니다.
    ///
    /// 편견 벡터의 VOCAB_CHUNK 블록 버퍼(134MB)는 계산 중에만 있는 activation 이라 activation 여유 축이 맡습니다.
    /// 여기서 또 더하면 이중 계산입니다.
    pub fn generation_budget_mb(&self, kind: usize, disk_mb: u64) -> u64 {
        let own = self.generation_mb(kind);
        let observed = if own > 0 { own } else { self.last_generation_mb() };
        disk_mb.max(observed).max(1) + self.activation_headroom_mb()
    }

    /// 임베딩 로드 전후의 free(MB) 로 상주 비용을 기록합니다.
    /// `free_before` 는 호출부가 정리를 끝낸 직후 값입니다. 로드 중 다른 정리로 free 가 늘었으면(오염) 기록하지 않습니다.
    pub fn record_embedding_load(&self, free_before: u64, free_after: u64) -> Option<u64> {
        let cost = free_before.checked_sub(free_after).filter(|&c| c > 0)?;
        self.embed_mb.store(cost, Ordering::SeqCst);
        Some(cost)
    }

    /// 생성 모델 비용은 깨끗한 기준선에서만 기록합니다
    /// (로드 전에 생성 슬롯이 비어 있었고, 임베딩 · 비전 상주가 로드 전후로 같을 때).
    ///
    /// 퍼지 후 로드를 퍼지 전 free 로 재면 내려간 만큼 상쇄되어 과소평가되고, 과소평가는 곧 OOM 쪽 오판입니다.
    /// 깨끗하지 않으면 기록하지 않고 디스크 추정(과대평가 쪽이라 안전)을 유지합니다.
    pub fn record_generation_load(&self, kind: usize, free_before: u64, free_after: u64, clean: bool) -> Option<u64> {
        if !clean {
            return None;
        }
        let cost = free_before.checked_sub(free_after).filter(|&c| c > 0)?;
        if let Some(slot) = self.gen_by_kind.get(kind) {
            slot.store(cost, Ordering::SeqCst);
        }
        self.gen_mb.store(cost, Ordering::SeqCst);
        Some(cost)
    }

    /// 대량 임베딩 직후에 부릅니다. 가중치 위에 연산 중 더 쓴 양의 누적 최대값을 학습해 다음 예산 판정에 더합니다.
    pub fn observe_activation(&self, free_before: u64, free_now: u64) {
        if let Some(used) = free_before.checked_sub(free_now) {
            self.activation_headroom_mb.fetch_max(used, Ordering::SeqCst);
        }
    }

    /// 지금 상주 상태가 차지하는 가중치(MB)를 피크에 반영하고 돌려줍니다.
    pub fn observe_residency(&self, residency: Residency) -> u64 {
        let mut occupied = 0;
        if residency.has_embedding() {
            occupied += self.embedding_mb();
        }
        if residency.has_generation() {
            occupied += self.last_generation_mb();
        }
        self.peak_mb.fetch_max(occupied, Ordering::SeqCst);
        occupied
    }

    pub fn mark_swap(&self) {
        self.swaps.fetch_add(1, Ordering::SeqCst);
        self.last_swap_ms
            .store(chrono::Utc::now().timestamp_millis().max(0) as u64, Ordering::SeqCst);
    }

    /// 동시 상주로 스왑을 피했음을 기록하고 누적 횟수를 돌려줍니다.
    pub fn mark_coexist(&self) -> u64 {
        self.coexists.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// 크로스오버 진단 요약. 태스크 종료 시 한 줄로 남깁니다.
    pub fn report(&self) -> String {
        format!(
            "[CROSSOVER REPORT] 스왑 {}회 | 동시상주 회피 {}회 | 임베딩 실측 {}MB | 생성 실측 {}MB | 관측 피크 {}MB | activation 여유 {}MB",
            self.swap_count(),
            self.coexist_count(),
            self.embedding_mb(),
            self.last_generation_mb(),
            self.peak_mb(),
            self.activation_headroom_mb(),
        )
    }
}

/// 생성 모델 전환 구간. 가드가 하나라도 살아 있으면 임베딩 로드는 양보합니다.
///
/// 예전에는 카운터만 있어 기다리는 쪽이 120ms 마다 다시 읽었고, 전환이 끝나도 최대 120ms 를 더 잤습니다.
/// 이제 마지막 가드가 풀릴 때 기다리는 쪽을 바로 깨웁니다.
#[derive(Default)]
pub struct HoldGate {
    holders: AtomicU32,
    released: Notify,
}

/// [`HoldGate::hold`] 의 가드. 스코프를 벗어나면(정상 · 에러 · 조기 반환 모두) 풀립니다.
pub struct GenerationHold(Arc<HoldGate>);

impl HoldGate {
    pub fn hold(self: &Arc<Self>) -> GenerationHold {
        self.holders.fetch_add(1, Ordering::SeqCst);
        GenerationHold(Arc::clone(self))
    }

    pub fn is_held(&self) -> bool {
        self.holders.load(Ordering::SeqCst) > 0
    }

    /// 가드가 모두 풀릴 때까지 기다립니다. 풀렸으면 true, `limit` 에 닿았으면 false.
    pub async fn wait_released(&self, limit: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + limit;
        loop {
            let notified = self.released.notified();
            tokio::pin!(notified);
            // 확인보다 먼저 등록해야, 확인과 대기 사이에 마지막 가드가 풀려도 신호를 놓치지 않습니다.
            notified.as_mut().enable();
            if !self.is_held() {
                return true;
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return !self.is_held();
            }
        }
    }
}

impl Drop for GenerationHold {
    fn drop(&mut self) {
        if self.0.holders.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.0.released.notify_waiters();
        }
    }
}

/// 찬 슬롯을 쥔 가드. 슬롯이 차 있을 때만 만들어지므로 꺼낼 때 다시 확인하지 않습니다.
///
/// 예전 임베딩 경로는 `ensure_embedding` 이 끝난 뒤 계산 스레드가 슬롯을 다시 잠갔습니다. 그 사이 생성 모델 전환이
/// 임베딩을 내리면 계산 스레드는 빈 슬롯을 보고 0 벡터를 돌려줬습니다 (오류 없이 틀린 값).
/// 이 가드를 계산 스레드로 넘기면 계산이 끝날 때까지 다른 작업의 `unload_embedding` · 퍼지는 슬롯을 기다립니다.
pub struct SlotLease<T>(OwnedMutexGuard<Option<T>>);

/// 임베딩 계산에 쓰는 [`SlotLease`].
pub type EmbeddingLease = SlotLease<EmbeddingModel>;

impl<T> SlotLease<T> {
    /// 슬롯이 비어 있으면 `None` 입니다 (가드는 그대로 풀립니다).
    pub fn new(slot: OwnedMutexGuard<Option<T>>) -> Option<Self> {
        if slot.is_some() {
            Some(Self(slot))
        } else {
            None
        }
    }
}

impl<T> std::ops::Deref for SlotLease<T> {
    type Target = T;

    fn deref(&self) -> &T {
        // `new` 가 빈 슬롯을 거르고, 가드를 쥔 동안에는 아무도 슬롯을 비울 수 없습니다.
        self.0.as_ref().expect("SlotLease 는 찬 슬롯에서만 만들어집니다")
    }
}

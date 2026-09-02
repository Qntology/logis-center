use crate::utils;
use crate::models::qwen::generate::QwenVLGenerateModel;
use crate::models::qwen3_5::generate::Qwen3_5GenerateModel;
use crate::models::embedding::EmbeddingModel;
use candle_core::DType;
use tokio::sync::Mutex as TokioMutex;
use crate::models::qwen3::generate::Qwen3GenerateModel;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering as XOrder};

pub mod lifecycle;
pub mod vision;
pub mod inference;
pub mod research;
pub mod query_shipping;
pub mod query_commerce;
pub mod merge;
pub mod query_analytic;

pub use merge::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelSize {
    Qwen,    // 0.6B for Ingestion (기존 Small)
    Qwen3,   // Qwen3 Text Model (기존 Large, /qwen3/ 로직 전용)
    Qwen3_5, // 2B Qwen 3.5 (Text Optimized)
}

#[derive(Clone)]
pub struct LogisModel {
    pub app_handle: tauri::AppHandle,
    pub generator: Arc<TokioMutex<Option<QwenVLGenerateModel>>>, 
    pub qwen3_generator: Arc<TokioMutex<Option<Qwen3GenerateModel>>>, 
    pub qwen3_5_generator: Arc<TokioMutex<Option<Qwen3_5GenerateModel>>>,
    
    pub embedding_model: Arc<TokioMutex<Option<EmbeddingModel>>>,
    pub embedding_cache: Arc<TokioMutex<std::collections::HashMap<String, Vec<f32>>>>,
    // 🌟 [GENERATION HOLD] 생성 모델 '전환 구간' 동안 임베딩 재로드를 막는 카운터입니다.
    //
    //  ── 무엇을 막는가 ──
    //   deep_purge_resources 는 Step 2 에서 임베딩 락을 블록 스코프로 잡았다 놓고,
    //   Step 3 의 spawn_blocking CUDA 동기화에서 최대 10초 블로킹됩니다.
    //   그 창으로 백그라운드 인덱싱의 ensure_embedding() 이 들어와
    //   방금 내린 임베딩을 다시 올립니다.
    //   (log.txt 실측: 퍼지 블록 '내부'에 [MODEL] Loading Embedding Model 이 끼어 있음)
    //   ensure_embedding 의 기존 VRAM GATE 는 free_mb < 350 일 때만 작동하는데
    //   퍼지 직후엔 4GB 가 비어 있어 이 레이스를 구조적으로 못 막습니다.
    //
    //  ── 왜 '금지' 가 아니라 '홀드' 인가 ──
    //   STAGE-3 은 Qwen3 상주 + 임베딩 동시 사용이 정상 경로입니다
    //   ([VECTORIZING] 직후 [KV-PLAN] ... VRAM free 2392MB → Vram).
    //   따라서 상주 자체를 금지하면 안 되고, '전환 구간에만' 양보시켜야 합니다.
    //   로드가 끝나면 홀드가 풀려 임베딩이 정상적으로 다시 올라옵니다.
    //
    //  ── 왜 Arc 인가 ──
    //   LogisModel 은 Clone 이고 내부가 전부 Arc 슬롯입니다.
    //   백그라운드 인덱싱이 clone 을 들고 있어도 같은 카운터를 봐야 합니다.
    pub generation_hold: Arc<std::sync::atomic::AtomicU32>,
    pub is_cpu_mode: bool, 
    pub is_disk_swap: bool,
    pub dual_mode_enabled: bool,
    
    // Config for Lazy Reloading
    qwen_model_path: String,      // 🌟 (기존 small_model_path 대신 이름 맞춤)
    qwen3_model_path: String,     // 🌟 Qwen3 모델 경로 추가
    qwen3_5_model_path: String,
    embedding_path: std::path::PathBuf,
    pub device_config: utils::DeviceConfig,
    max_tokens_limit: u32,
    _dtype: Option<DType>, 
    current_size: Arc<TokioMutex<Option<ModelSize>>>,

    pub siglip2_model: Arc<TokioMutex<Option<crate::models::siglip2::Siglip2Model>>>,
    pub siglip2_config: Option<crate::models::siglip2::Siglip2Config>,
    pub siglip2_model_path: String,
}

/// 🌟 [RAII] 생성 모델 전환 구간을 나타내는 가드입니다.
///
///  Drop 시 카운터를 되돌리므로, `?` 로 조기 반환하거나 패닉이 나도
///  홀드가 영구히 걸린 채 남지 않습니다. 명시적 해제 호출을 두면
///  에러 경로에서 반드시 새는데, 그 순간 임베딩이 영영 못 올라옵니다.
/// 🌟 [RAII] 생성 모델 전환 구간을 나타내는 가드입니다.
///
///  Drop 시 카운터를 되돌리므로, `?` 로 조기 반환하거나 패닉이 나도
///  홀드가 영구히 걸린 채 남지 않습니다. 명시적 해제 호출을 두면
///  에러 경로에서 반드시 새는데, 그 순간 임베딩이 영영 못 올라갑니다.
pub struct GenerationHold(Arc<std::sync::atomic::AtomicU32>);

impl Drop for GenerationHold {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

impl LogisModel {
    /// 생성 모델 전환 구간 진입을 선언합니다.
    /// 반환된 가드가 살아 있는 동안 ensure_embedding 은 로드를 양보합니다.
    pub fn hold_generation(&self) -> GenerationHold {
        self.generation_hold.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        GenerationHold(self.generation_hold.clone())
    }

    /// 지금 생성 모델 전환 구간인지 확인합니다.
    pub fn is_generation_held(&self) -> bool {
        self.generation_hold.load(std::sync::atomic::Ordering::SeqCst) > 0
    }
}

// =====================================================================
// 🌟 [CROSSOVER SWITCH] 생성 모델 ↔ 임베딩 모델 교차 상주 스케줄러
// ---------------------------------------------------------------------
//  ── 무엇이 문제였나 ──
//   한 태스크 안에서 두 종류의 모델을 번갈아 씁니다.
//     · 생성 모델  : Qwen3(0.6B) / Qwen3.5(2B) / Qwen(0.6B VL)
//     · 임베딩 모델: granite-embedding-97m
//   그런데 '어느 쪽이 상주해야 하는가' 를 선언하는 지점이 없습니다.
//   secure_vram_relay 는 자기 앞의 것을 정리하지만, 그 반대 방향
//   (생성 모델이 올라온 뒤 get_embedding 을 부르는 경로)에는
//   어떤 정리도 없어 두 가중치가 그대로 더해집니다. 이것이 피크입니다.
//
//  ── 왜 '무조건 스왑' 이 답이 아닌가 ──
//   임베딩 호출마다 생성 모델을 내리면 아이템 20개 루프에서 40회 왕복이
//   발생합니다. 가중치 재로드(디스크 → VRAM)가 연산보다 비쌉니다.
//   실제로 STAGE-3 은 [VECTORIZING] 직후 곧바로 LLM 을 호출합니다.
//
//  ── 해결 : 예산 기반 크로스오버 ──
//   ① 상주 비용을 '디스크 가중치 크기' 로 유도하고,
//      로드 전후 free VRAM 차이로 즉시 실측값으로 교체합니다.
//      (하드코딩 임계치 2600MB 를 제거하는 근거가 여기 있습니다)
//   ② 올리기 전에 "지금 올려도 여유가 남는가" 를 판정합니다.
//      남으면 동시 상주(스왑 0회), 모자라면 상대를 먼저 내립니다.
//   ③ 배치 적응: 여유가 적으면 임베딩 배치를 쪼개
//      가중치가 아니라 '연산 중 순간 점유(activation)' 를 깎습니다.
//
//  ── 왜 전역 static 인가 ──
//   LogisModel 은 Clone 이고 내부가 전부 Arc 슬롯입니다.
//   generation_hold 와 같은 이유로 clone 간 상태가 공유되어야 하는데,
//   구조체에 필드를 추가하면 LogisModel::new(lifecycle.rs)를 함께 고쳐야
//   합니다. 프로세스 전역에 모델은 하나뿐이므로 static 이 동일하게
//   동작하며, TRANSLIT_MEM_CACHE 가 이미 같은 선례입니다.
// =====================================================================

/// 아무것도 상주하지 않는 상태 (퍼지 직후)
pub const PHASE_IDLE: u8 = 0;
/// 임베딩 모델만 상주
pub const PHASE_EMBEDDING: u8 = 1;
/// 생성 모델만 상주
pub const PHASE_GENERATION: u8 = 2;
/// 둘 다 상주 (예산이 허용한 경우에만)
pub const PHASE_BOTH: u8 = 3;

static CROSSOVER_PHASE: AtomicU8 = AtomicU8::new(PHASE_IDLE);
/// 임베딩 실측 상주 비용(MB). 0 이면 아직 미관측 → 디스크 크기로 대체합니다.
static EMBED_RESIDENT_MB: AtomicU64 = AtomicU64::new(0);
/// 마지막으로 올린 생성 모델의 실측 상주 비용(MB).
static GEN_RESIDENT_MB: AtomicU64 = AtomicU64::new(0);
/// 관측된 최대 동시 점유(MB). 진단 전용.
static PEAK_RESIDENT_MB: AtomicU64 = AtomicU64::new(0);
/// 스왑(한쪽을 내리고 다른 쪽을 올린) 횟수.
static SWAP_COUNT: AtomicU64 = AtomicU64::new(0);
/// 동시 상주로 스왑을 회피한 횟수.
static COEXIST_COUNT: AtomicU64 = AtomicU64::new(0);
/// 마지막 스왑 시각(epoch ms). 왕복 진단용.
static LAST_SWAP_MS: AtomicU64 = AtomicU64::new(0);
/// 관측된 activation 여유(MB). 로드 직후 free 와 연산 중 free 의 차이입니다.
static ACTIVATION_HEADROOM_MB: AtomicU64 = AtomicU64::new(0);

impl LogisModel {
    // ── 상주 비용 추정 ──────────────────────────────────────────────
    /// 가중치 파일들의 총 바이트를 MB 로 환산합니다.
    ///
    ///  ── 왜 파일 크기인가 ──
    ///   양자화 GGUF 는 파일 크기와 VRAM 가중치 크기가 사실상 1:1 입니다.
    ///   safetensors 도 dtype 을 낮추지 않으면 1:1 이고, 낮추면 이 값이
    ///   과대평가가 되는데 예산 판정에서 과대평가는 '보수적' 방향이라
    ///   OOM 을 유발하지 않습니다. 실측이 들어오면 즉시 교체됩니다.
    ///
    ///  ── mmproj 제외 ──
    ///   Qwen3.5 디렉터리에는 비전 프로젝터(mmproj-BF16.gguf)가 함께 있는데
    ///   텍스트 전용 경로에서는 올라가지 않으므로 합산하지 않습니다.
    fn path_footprint_mb(path: &std::path::Path) -> u64 {
        fn walk(p: &std::path::Path, acc: &mut u64, depth: usize) {
            if depth > 4 { return; }
            let meta = match std::fs::metadata(p) { Ok(m) => m, Err(_) => return };
            if meta.is_file() {
                let name = p.file_name().and_then(|f| f.to_str()).unwrap_or("");
                if name.contains("mmproj") { return; }
                let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                if ext == "gguf" || ext == "safetensors" || ext == "bin" {
                    *acc += meta.len();
                }
                return;
            }
            if meta.is_dir() {
                if let Ok(rd) = std::fs::read_dir(p) {
                    for e in rd.flatten() { walk(&e.path(), acc, depth + 1); }
                }
            }
        }
        let mut bytes = 0u64;
        walk(path, &mut bytes, 0);
        bytes / (1024 * 1024)
    }

    /// 임베딩 모델을 올리는 데 필요한 VRAM(MB). 실측이 있으면 실측을 씁니다.
    pub fn embedding_budget_mb(&self) -> u64 {
        let observed = EMBED_RESIDENT_MB.load(XOrder::SeqCst);
        let base = if observed > 0 {
            observed
        } else {
            Self::path_footprint_mb(&self.embedding_path).max(1)
        };
        base + ACTIVATION_HEADROOM_MB.load(XOrder::SeqCst)
    }

    /// 생성 모델을 올리는 데 필요한 VRAM(MB).
    pub fn generation_budget_mb(&self, size: ModelSize) -> u64 {
        let dir = match size {
            ModelSize::Qwen => self.qwen_model_path.clone(),
            ModelSize::Qwen3 => self.qwen3_model_path.clone(),
            ModelSize::Qwen3_5 => self.qwen3_5_model_path.clone(),
        };
        let disk = Self::path_footprint_mb(std::path::Path::new(&dir));
        let observed = GEN_RESIDENT_MB.load(XOrder::SeqCst);
        // 실측은 '마지막에 올린 모델' 기준이라 크기가 다른 모델에는 부정확합니다.
        // 두 값 중 큰 쪽을 택해 보수적으로 판정합니다.
        disk.max(observed).max(1) + ACTIVATION_HEADROOM_MB.load(XOrder::SeqCst)
    }

    // ── 페이즈 상태 ────────────────────────────────────────────────
    pub fn crossover_phase(&self) -> u8 {
        CROSSOVER_PHASE.load(XOrder::SeqCst)
    }

    fn set_phase(p: u8) {
        CROSSOVER_PHASE.store(p, XOrder::SeqCst);
    }

    fn mark_swap() {
        SWAP_COUNT.fetch_add(1, XOrder::SeqCst);
        LAST_SWAP_MS.store(
            chrono::Utc::now().timestamp_millis().max(0) as u64,
            XOrder::SeqCst,
        );
    }

    /// 퍼지 직후 등 '아무것도 없음' 을 선언합니다.
    /// deep_purge_resources 를 직접 호출한 경로가 상태를 되돌릴 때 씁니다.
    /// 🌟 sync_crossover_phase 가 도입된 뒤로는 하위 호환용입니다.
    pub fn mark_crossover_idle(&self) {
        Self::set_phase(PHASE_IDLE);
    }

    // ── 실제 슬롯 조회 ─────────────────────────────────────────────
    //
    //  ── 왜 free VRAM 델타가 아니라 슬롯을 직접 읽는가 ──
    //   lifecycle.rs 대조 결과 secure_vram_relay 는 모델이 바뀌면
    //   deep_purge_resources 를 호출하고, 그 Step 2 가 임베딩을 함께 파기합니다.
    //   즉 '전환 후 임베딩이 남았는가' 는 추측할 필요가 없는 확정 사실이며,
    //   embedding_model 슬롯을 읽으면 그대로 나옵니다.
    //   free VRAM 델타는 다른 프로세스의 할당에 오염되므로 판정 근거로 부적절합니다.

    pub async fn embedding_resident(&self) -> bool {
        self.embedding_model.lock().await.is_some()
    }

    pub async fn generation_resident(&self) -> bool {
        self.generator.lock().await.is_some()
            || self.qwen3_generator.lock().await.is_some()
            || self.qwen3_5_generator.lock().await.is_some()
    }

    /// 🌟 실제 슬롯 상태를 읽어 페이즈 원장을 사실과 일치시킵니다.
    ///
    ///  각 lock 은 `.is_some()` 임시값이라 statement 끝에서 즉시 해제됩니다.
    ///  따라서 deep_purge_resources / unload_embedding 의 말미처럼
    ///  이미 가드를 놓은 지점에서 호출해도 재진입 데드락이 없습니다.
    pub async fn sync_crossover_phase(&self) -> u8 {
        let e = self.embedding_resident().await;
        let g = self.generation_resident().await;
        let p = match (g, e) {
            (true, true) => PHASE_BOTH,
            (true, false) => PHASE_GENERATION,
            (false, true) => PHASE_EMBEDDING,
            (false, false) => PHASE_IDLE,
        };
        Self::set_phase(p);
        p
    }

    /// 🌟 임베딩을 상주시킨 채 이 생성 모델을 올려도 되는가.
    ///
    ///  임베딩은 이미 VRAM 을 점유한 상태이므로, 지금 측정한 자유 메모리는
    ///  '임베딩을 남긴 채 쓸 수 있는 양' 그 자체입니다.
    ///  그 값이 생성 예산을 넘으면 임베딩을 내릴 이유가 없습니다.
    ///  이 판정은 lifecycle.rs 의 ensure_qwen3 / ensure_qwen3_5 도 공유합니다.
    pub fn embedding_coexist_ok(&self, size: ModelSize) -> bool {
        if self.is_cpu_mode { return true; }
        self.get_free_vram_mb() >= self.generation_budget_mb(size)
    }

    fn observe_embedding_cost(&self, free_before: u64) {
        if self.is_cpu_mode { return; }
        let after = self.get_free_vram_mb();
        // free_before 는 '이 함수를 부른 쪽이 정리를 끝낸 직후' 값입니다.
        // 로드 중 내부에서 퍼지가 일어났다면 after > free_before 가 되어
        // 아래 조건이 거짓이 되므로 오염된 값이 기록되지 않습니다.
        if free_before > after {
            let cost = free_before - after;
            if cost > 0 { EMBED_RESIDENT_MB.store(cost, XOrder::SeqCst); }
        }
        self.observe_peak(after);
    }

    /// 🌟 [STRICT MEASURE] 생성 모델 비용은 '아무것도 상주하지 않은 상태' 에서만 기록합니다.
    ///
    ///  ── 왜 조건을 다는가 ──
    ///   secure_vram_relay 는 퍼지 후 로드입니다. 퍼지 전 free 를 기준으로 재면
    ///   '임베딩이 내려간 만큼' 이 생성 비용에서 상쇄되어 과소평가됩니다.
    ///   예산이 과소평가되면 곧바로 OOM 방향의 오판이 되므로 허용할 수 없습니다.
    ///   측정이 깨끗하지 않으면 기록하지 않고 디스크 추정치를 유지합니다.
    ///   (디스크 추정치는 과대평가 방향이라 안전합니다)
    fn observe_generation_cost_strict(&self, free_before: u64, pre_empty: bool) {
        if self.is_cpu_mode { return; }
        let after = self.get_free_vram_mb();
        if pre_empty && free_before > after {
            let cost = free_before - after;
            if cost > 0 { GEN_RESIDENT_MB.store(cost, XOrder::SeqCst); }
        }
        self.observe_peak(after);
    }

    /// 지금까지 관측된 '가장 적게 남았던 순간' 을 피크로 환산해 기록합니다.
    fn observe_peak(&self, free_now: u64) {
        let e = EMBED_RESIDENT_MB.load(XOrder::SeqCst);
        let g = GEN_RESIDENT_MB.load(XOrder::SeqCst);
        let occupied = match CROSSOVER_PHASE.load(XOrder::SeqCst) {
            PHASE_BOTH => e + g,
            PHASE_EMBEDDING => e,
            PHASE_GENERATION => g,
            _ => 0,
        };
        let prev = PEAK_RESIDENT_MB.load(XOrder::SeqCst);
        if occupied > prev { PEAK_RESIDENT_MB.store(occupied, XOrder::SeqCst); }
        let _ = free_now;
    }

    // ── 크로스오버 진입점 ──────────────────────────────────────────
    /// 🌟 임베딩 페이즈로 전환합니다.
    ///
    ///  현재 생성 모델이 상주 중이면 예산을 판정하여
    ///    · 여유가 있으면 그대로 두고 임베딩만 얹습니다 (스왑 0회)
    ///    · 여유가 없으면 생성 슬롯만 반환시킵니다
    ///
    ///  이 함수를 호출한 뒤에 get_embedding / get_embedding_batch 를 쓰면
    ///  더 이상 암묵적 로드로 피크가 생기지 않습니다.
    pub async fn enter_embedding_phase(&self, reason: &str) -> anyhow::Result<()> {
        if self.is_cpu_mode {
            self.check_embedding_downloaded().await?;
            self.ensure_embedding().await?;
            self.sync_crossover_phase().await;
            return Ok(());
        }

        // 🌟 [FACT OVER LEDGER] 원장(CROSSOVER_PHASE)이 아니라 실제 슬롯을 봅니다.
        //    ensure_qwen3 / ensure_qwen3_5 / release_siglip2 등 원장을 거치지 않는
        //    경로가 여전히 존재하므로, 판정 근거는 항상 슬롯이어야 합니다.
        if self.embedding_resident().await {
            self.sync_crossover_phase().await;
            return Ok(());
        }

        if self.generation_resident().await {
            let free = self.get_free_vram_mb();
            let need = self.embedding_budget_mb();
            if free >= need {
                COEXIST_COUNT.fetch_add(1, XOrder::SeqCst);
                println!(
                    "[CROSSOVER] 🤝 [COEXIST] {} | 자유 {}MB >= 임베딩 예산 {}MB → 생성 모델을 유지한 채 임베딩을 얹습니다. (스왑 회피 누적 {}회)",
                    reason, free, need, COEXIST_COUNT.load(XOrder::SeqCst)
                );
                let before = self.get_free_vram_mb();
                self.check_embedding_downloaded().await?;
                self.ensure_embedding().await?;
                self.observe_embedding_cost(before);
                self.sync_crossover_phase().await;
                return Ok(());
            }
            println!(
                "[CROSSOVER] 🔻 [SWAP-OUT GEN] {} | 자유 {}MB < 임베딩 예산 {}MB → 생성 슬롯을 반환합니다.",
                reason, free, need
            );
            // 🌟 [PARTIAL UNLOAD] deep_purge_resources 대신 생성 슬롯만 반환합니다.
            //    전체 퍼지는 SigLIP2 까지 파기하고 CUDA 동기화에만 최대 10초를 씁니다.
            //    지금 필요한 것은 '생성 모델이 쥔 VRAM' 뿐이므로 그것만 놓습니다.
            self.unload_generation_slots(reason).await;
            Self::mark_swap();
        }

        let before = self.get_free_vram_mb();
        self.check_embedding_downloaded().await?;
        self.ensure_embedding().await?;
        self.observe_embedding_cost(before);
        let phase = self.sync_crossover_phase().await;
        println!(
            "[CROSSOVER] 🧬 [EMBEDDING PHASE] {} | 임베딩 상주 확정 (phase={} | 실측 {}MB | 자유 {}MB)",
            reason,
            phase,
            EMBED_RESIDENT_MB.load(XOrder::SeqCst),
            self.get_free_vram_mb()
        );
        Ok(())
    }

    /// 🌟 생성 페이즈로 전환합니다.
    ///
    ///  ── lifecycle.rs 대조로 확정된 사실 ──
    ///   secure_vram_relay 는 모델이 바뀌면 반드시 deep_purge_resources 를 호출하고,
    ///   그 Step 2 가 embedding_model 슬롯을 take() 로 비웁니다.
    ///   따라서 relay 를 타는 한 '임베딩 유지' 는 물리적으로 불가능합니다.
    ///   동시 상주를 원하면 relay 를 우회해 ensure_* 를 직접 불러야 하며,
    ///   그 ensure_* 도 임베딩을 퍼지 트리거에서 빼도록 함께 고쳐야 합니다.
    ///   (Part 7 의 ensure_qwen3 / ensure_qwen3_5 수정이 그 짝입니다)
    ///
    ///  ── relay 를 반드시 타야 하는 경우 ──
    ///   · session_id 가 있는 경우 : load_kv_snapshot 이 relay 안에만 있습니다.
    ///   · prefill(is_baking)      : 베이킹 전용 로드 플래그가 relay 를 경유합니다.
    ///   이 두 경우는 예산과 무관하게 relay 로 보냅니다.
    ///
    ///  전환 구간 전체를 hold_generation 으로 감싸므로,
    ///  백그라운드 인덱싱이 이 창으로 끼어들어 임베딩을 되올리지 못합니다.
    pub async fn enter_generation_phase(
        &self,
        size: ModelSize,
        session_id: Option<&str>,
        cancel: Option<Arc<AtomicBool>>,
        prefill: bool,
        kv_name: Option<String>,
        reason: &str,
    ) -> anyhow::Result<()> {
        let _hold = self.hold_generation();

        if self.is_cpu_mode {
            self.secure_vram_relay(size, session_id, cancel, prefill, kv_name).await?;
            self.sync_crossover_phase().await;
            return Ok(());
        }

        let embed_before = self.embedding_resident().await;
        let relay_required = session_id.is_some() || prefill;
        let coexist_ok = embed_before && !relay_required && self.embedding_coexist_ok(size);

        if embed_before && !coexist_ok {
            let free = self.get_free_vram_mb();
            let need = self.generation_budget_mb(size);
            println!(
                "[CROSSOVER] 🔻 [SWAP-OUT EMBED] {} | 자유 {}MB / 생성 예산 {}MB{} → 임베딩을 먼저 반환합니다.",
                reason, free, need,
                if relay_required { " | relay 전용 경로(KV 스냅샷/프리필)" } else { "" }
            );
            self.unload_embedding().await;
            Self::mark_swap();
        }

        // 🌟 [CLEAN BASELINE] 지금 슬롯이 전부 비어 있으면 free_before 가
        //    오염되지 않은 기준선이므로 생성 비용을 실측으로 확정할 수 있습니다.
        let pre_empty = !self.generation_resident().await
            && !self.embedding_resident().await;
        let before = self.get_free_vram_mb();

        if coexist_ok {
            COEXIST_COUNT.fetch_add(1, XOrder::SeqCst);
            println!(
                "[CROSSOVER] 🤝 [COEXIST] {} | 자유 {}MB >= 생성 예산 {}MB → 임베딩을 유지한 채 {:?} 를 올립니다. (스왑 회피 누적 {}회)",
                reason, before, self.generation_budget_mb(size), size,
                COEXIST_COUNT.load(XOrder::SeqCst)
            );
            match size {
                ModelSize::Qwen => self.ensure_generator_ext(ModelSize::Qwen, false, false).await?,
                ModelSize::Qwen3 => self.ensure_qwen3().await?,
                ModelSize::Qwen3_5 => self.ensure_qwen3_5(false).await?,
            };
        } else {
            self.secure_vram_relay(size, session_id, cancel, prefill, kv_name).await?;
        }

        self.observe_generation_cost_strict(before, pre_empty);
        let phase = self.sync_crossover_phase().await;
        println!(
            "[CROSSOVER] 🧠 [GENERATION PHASE] {} | {:?} 상주 확정 (phase={} | 생성 실측 {}MB | 자유 {}MB)",
            reason,
            size,
            phase,
            GEN_RESIDENT_MB.load(XOrder::SeqCst),
            self.get_free_vram_mb()
        );
        Ok(())
    }

    /// enter_generation_phase 의 축약형. 대부분의 호출부가 이 형태입니다.
    pub async fn switch_to_generation(
        &self,
        size: ModelSize,
        cancel: Option<Arc<AtomicBool>>,
        kv_name: Option<String>,
        reason: &str,
    ) -> anyhow::Result<()> {
        self.enter_generation_phase(size, None, cancel, false, kv_name, reason).await
    }

    /// 🌟 [DEPRECATED] 배치 크기로는 activation 을 줄일 수 없습니다.
    ///
    ///  ── 왜 무력한가 ──
    ///   models/embedding.rs 의 embed_batch 는 이름과 달리 배치 연산이 아닙니다.
    ///     for text in chunk { local_res.push(self.embed(text)...) }
    ///   내부는 1건씩 순회이고, 시퀀스 길이도 embed() 안에서 512 로 고정입니다.
    ///   따라서 '한 번에 몇 건을 넘기는가' 는 순간 점유에 영향이 없습니다.
    ///
    ///  ── 진짜 축은 무엇인가 ──
    ///   embed_batch 는 GPU 에서 3개 스레드를 띄워 동시에 순전파합니다.
    ///   Attention::forward 가 .contiguous() 를 12회 호출하며 매번 새 텐서를
    ///   할당하므로, 그 순간 점유가 스레드 수만큼 배가됩니다.
    ///   조절 대상은 그 스레드 수이며, embedding.rs 의 adaptive_thread_count 가
    ///   관측값을 근거로 3 → 2 → 1 로 줄입니다.
    ///
    ///  ── 왜 삭제하지 않는가 ──
    ///   호출부가 남아 있으면 컴파일 경고로 즉시 드러나야 하고,
    ///   나중에 '배치를 줄이면 되지 않나' 는 같은 판단이 재발하는 것을 막기 위해
    ///   무력한 이유를 코드에 남깁니다. 항상 요청값을 그대로 돌려줍니다.
    #[deprecated(
        since = "crossover-v2",
        note = "embed_batch 는 1건씩 순회하므로 배치 크기가 activation 에 영향이 없습니다. \
                embedding.rs 의 adaptive_thread_count 를 사용하십시오."
    )]
    #[allow(dead_code)]
    pub fn adaptive_embed_batch(&self, requested: usize) -> usize {
        requested.max(1)
    }

    /// 🌟 크로스오버 진단 요약. 태스크 종료 시 한 줄로 남깁니다.
    pub fn crossover_report(&self) -> String {
        format!(
            "[CROSSOVER REPORT] 스왑 {}회 | 동시상주 회피 {}회 | 임베딩 실측 {}MB | 생성 실측 {}MB | 관측 피크 {}MB | activation 여유 {}MB",
            SWAP_COUNT.load(XOrder::SeqCst),
            COEXIST_COUNT.load(XOrder::SeqCst),
            EMBED_RESIDENT_MB.load(XOrder::SeqCst),
            GEN_RESIDENT_MB.load(XOrder::SeqCst),
            PEAK_RESIDENT_MB.load(XOrder::SeqCst),
            ACTIVATION_HEADROOM_MB.load(XOrder::SeqCst),
        )
    }

    /// 🌟 activation 여유를 관측합니다.
    ///  대량 임베딩 배치 직후에 호출하면, 가중치 위에 얼마나 더 쓰였는지가
    ///  누적 최대값으로 학습되어 다음 예산 판정이 정확해집니다.
    pub fn observe_activation_headroom(&self, free_before_batch: u64) {
        if self.is_cpu_mode { return; }
        let now = self.get_free_vram_mb();
        if free_before_batch > now {
            let used = free_before_batch - now;
            let prev = ACTIVATION_HEADROOM_MB.load(XOrder::SeqCst);
            if used > prev { ACTIVATION_HEADROOM_MB.store(used, XOrder::SeqCst); }
        }
    }
}
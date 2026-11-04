use std::time::Duration;
use crate::utils;
use crate::models::qwen::generate::QwenVLGenerateModel;
use crate::models::qwen3_5::generate::Qwen3_5GenerateModel;
use crate::models::embedding::EmbeddingModel;
use candle_core::DType;
use tokio::sync::Mutex as TokioMutex;
use crate::models::qwen3::generate::Qwen3GenerateModel;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as XOrder};

pub mod residency;
pub mod lifecycle;
pub mod vision;
pub mod inference;
pub mod research;
pub mod query_shipping;
pub mod query_commerce;
pub mod merge;
pub mod query_analytic;
pub mod lang_llm;
pub mod lang_gguf;

pub use merge::*;
pub use residency::{CostBook, GenerationHold, Residency, GEN_KIND_LANG4B};

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
    /// 🌟 [QWEN4-EXP] 언어 엔진 슬롯이 Qwen4-Exp 체크포인트를 읽었을 때 쓰는 자리 (`qwen3_5_generator` 와 동시에 차지 않음).
    ///    `lang_llm::resident_variant()` 가 가리키는 엔진이 이 슬롯 또는 `qwen3_5_generator` 에 있습니다.
    pub qwen4_exp_generator: Arc<TokioMutex<Option<crate::models::qwen4_exp::Qwen4ExpGenerateModel>>>,
    
    pub embedding_model: Arc<TokioMutex<Option<EmbeddingModel>>>,
    pub embedding_cache: Arc<TokioMutex<std::collections::HashMap<String, Vec<f32>>>>,
    /// 생성 모델 전환 구간. clone 끼리 공유합니다 (`hold_generation`).
    pub hold_gate: Arc<residency::HoldGate>,
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
pub struct VramSampler {
    stop: Arc<std::sync::atomic::AtomicBool>,
    min_free: Arc<std::sync::atomic::AtomicU64>,
    phase: Arc<std::sync::Mutex<String>>,
    worst_phase: Arc<std::sync::Mutex<String>>,
    label: String,
    baseline: u64,
}

impl VramSampler {
    /// 지금부터의 단계 이름을 갱신합니다. 최저점 갱신 시 이 이름이 함께 기록됩니다.
    pub fn phase(&self, name: impl Into<String>) {
        if let Ok(mut p) = self.phase.lock() {
            *p = name.into();
        }
    }

    /// 중간 보고. 루프 안에서 주기적으로 찍어 추이를 볼 때 씁니다.
    pub fn snapshot(&self) -> (u64, u64, String) {
        let lo = self.min_free.load(std::sync::atomic::Ordering::SeqCst);
        let w = self.worst_phase.lock().map(|s| s.clone()).unwrap_or_default();
        (self.baseline, lo, w)
    }
}

impl Drop for VramSampler {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        let lo = self.min_free.load(std::sync::atomic::Ordering::SeqCst);
        if lo == u64::MAX { return; }
        let w = self.worst_phase.lock().map(|s| s.clone()).unwrap_or_default();
        if self.baseline > lo {
            println!(
                "[VRAM-PEAK] 🔺 {} | 진입 {}MB → 최저 {}MB | 순간 점유 +{}MB | 최저 시점 단계: '{}'",
                self.label, self.baseline, lo, self.baseline - lo, w
            );
        } else {
            println!(
                "[VRAM-PEAK] ✅ {} | 진입 {}MB → 최저 {}MB | 순간 증가 없음",
                self.label, self.baseline, lo
            );
        }
    }
}

impl LogisModel {
    /// 생성 모델 전환 구간 진입을 선언합니다.
    /// 반환된 가드가 살아 있는 동안 ensure_embedding 은 로드를 양보하고, 마지막 가드가 풀리면 바로 깨어납니다.
    pub fn hold_generation(&self) -> GenerationHold {
        self.hold_gate.hold()
    }

    /// 지금 생성 모델 전환 구간인지 확인합니다.
    pub fn is_generation_held(&self) -> bool {
        self.hold_gate.is_held()
    }

    /// 상주 비용 실측 장부 (프로세스에 하나).
    pub fn costs(&self) -> &'static CostBook {
        CostBook::process()
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
//  ── 상태는 어디에 있는가 (model/residency.rs) ──
//   · 무엇이 상주하는가(Residency) : 슬롯에서만 유도합니다. 따로 적는 원장이 없어
//     퍼지가 임베딩을 남겼는데 원장만 IDLE 인 식의 불일치가 생길 수 없습니다.
//   · 실측 비용 · 통계(CostBook)   : 장치와 파일의 성질이라 프로세스에 하나이고,
//     LogisModel 을 새로 만들어도 유지됩니다.
//   · 전환 구간(HoldGate)          : clone 끼리 공유하는 Arc 필드입니다.
// =====================================================================

pub const RAM_PRESSURE_MARGIN_BYTES: u64 = 2 * 1024 * 1024 * 1024;

pub fn ram_snapshot() -> (u64, u64) {
    let avail = crate::utils::resources::free_ram_bytes() as u64;
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    (avail, sys.total_memory())
}

pub fn ram_used_pct(avail: u64, total: u64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    (total.saturating_sub(avail) as f64 * 100.0 / total as f64) as f32
}

pub fn process_rss_bytes() -> Option<u64> {
    let pid = sysinfo::get_current_pid().ok()?;
    let sys = sysinfo::System::new_all();
    sys.process(pid).map(|p| p.memory())
}

pub fn ram_pressure_line(avail: u64, total: u64) -> String {
    let limit_pct = if total > 0 {
        (100.0 - RAM_PRESSURE_MARGIN_BYTES as f64 * 100.0 / total as f64).max(0.0)
    } else {
        0.0
    };
    format!(
        "RAM 사용 {:.1}% (여유 {}MB / 총 {}MB · 이 프로세스 작업 집합 {}) | 압박 기준: 여유 {}MB 미만 = 이 PC 에서 사용 {:.1}% 이상 (KV-PLAN 의 RAM margin 과 같은 값)",
        ram_used_pct(avail, total),
        avail / 1_000_000,
        total / 1_000_000,
        process_rss_bytes().map_or("-".to_string(), |b| format!("{}MB", b / 1_000_000)),
        RAM_PRESSURE_MARGIN_BYTES / 1_000_000,
        limit_pct
    )
}

/// 🌟 [HEAP-RELEASE] RAM 압박 시 메모리 반환.
/// 기본은 mimalloc 의 빈 힙 페이지만 OS 로 돌려줍니다(utils::release_heap). 예전처럼 작업 집합 전체를
/// 내쫓으면(SetProcessWorkingSetSizeEx) mmap 가중치·토크나이저 페이지가 곧바로 다시 읽혀 RAM 이 톱니처럼
/// 오르내리고 수 초씩 멈췄습니다. `LOGIS_MEM_TRIM=legacy` 이면 예전 동작입니다.
pub fn release_os_working_set() {
    if crate::utils::legacy_ws_trim_enabled() {
        crate::utils::force_working_set_trim();
    } else {
        crate::utils::release_heap();
    }
}

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
        self.costs()
            .embedding_budget_mb(|| Self::path_footprint_mb(&self.embedding_path))
    }

    /// 생성 모델을 올리는 데 필요한 VRAM(MB). 규칙은 `CostBook::generation_budget_mb`.
    pub fn generation_budget_mb(&self, size: ModelSize) -> u64 {
        let dir = match size {
            ModelSize::Qwen => &self.qwen_model_path,
            ModelSize::Qwen3 => &self.qwen3_model_path,
            ModelSize::Qwen3_5 => &self.qwen3_5_model_path,
        };
        let disk = Self::path_footprint_mb(std::path::Path::new(dir));
        self.costs().generation_budget_mb(Self::gen_kind(size), disk)
    }

    pub fn gen_kind(size: ModelSize) -> usize {
        match size {
            ModelSize::Qwen => 0,
            ModelSize::Qwen3 => 1,
            ModelSize::Qwen3_5 => 2,
        }
    }

    pub fn measured_generation_mb(kind: usize) -> u64 {
        CostBook::process().generation_mb(kind)
    }

    pub async fn resident_generation_cost_mb(&self) -> u64 {
        let qwen = self.generator.lock().await.is_some();
        let qwen3 = self.qwen3_generator.lock().await.is_some();
        let qwen3_5 = self.qwen3_5_generator.lock().await.is_some() || self.qwen4_exp_generator.lock().await.is_some();
        let mut total = 0u64;
        if qwen {
            total += Self::measured_generation_mb(0);
        }
        if qwen3 {
            total += Self::measured_generation_mb(1);
        }
        if qwen3_5 {
            let kind = if crate::model::lang_llm::resident_variant().is_some() { GEN_KIND_LANG4B } else { 2 };
            total += Self::measured_generation_mb(kind);
        }
        total
    }

    fn mark_swap() {
        CostBook::process().mark_swap();
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
            || self.qwen4_exp_generator.lock().await.is_some()
    }

    /// 🌟 실제 슬롯에서 지금 상주 상태를 읽고, 그 상태가 차지하는 가중치를 피크에 반영합니다.
    ///  슬롯이 바뀐 직후마다 부릅니다. 상태를 따로 적어 두지 않으므로 사실과 어긋날 수 없습니다.
    ///
    ///  각 lock 은 `.is_some()` 임시값이라 statement 끝에서 즉시 해제됩니다.
    ///  따라서 deep_purge_resources / unload_embedding 의 말미처럼
    ///  이미 가드를 놓은 지점에서 호출해도 재진입 데드락이 없습니다.
    ///  단, `EmbeddingLease` 를 쥔 채로 부르면 임베딩 슬롯을 기다리며 멈춥니다.
    pub async fn sync_crossover_phase(&self) -> Residency {
        let embedding = self.embedding_resident().await;
        let generation = self.generation_resident().await;
        let residency = Residency::from_slots(generation, embedding);
        self.costs().observe_residency(residency);
        residency
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

    // =====================================================================
    // 🌟 [PEAK SAMPLER] 순간 점유를 잡아내는 백그라운드 표본기
    // ---------------------------------------------------------------------
    //  ── 왜 동기 측정으로는 안 되는가 ──
    //   generate 앞뒤에서 get_free_vram_mb() 를 재면 '연산 전' 과 '연산 후' 만
    //   보입니다. 문제의 1.5GB 는 연산 '도중' 에만 존재하고 반환되므로
    //   앞뒤 값은 둘 다 평평하게 나옵니다.
    //   로그의 KV-PLAN free 가 2308~2375 로 붙박이인 이유가 정확히 이것입니다.
    //
    //  ── 왜 별도 태스크인가 ──
    //   generate 는 spawn_blocking 안에서 돕니다. 그 사이 tokio 런타임은
    //   비어 있으므로 50ms 폴링이 실제 연산을 방해하지 않습니다.
    //
    //  ── 왜 NVML 을 태스크 안에서 1회만 여는가 ──
    //   get_free_vram_mb() 는 호출마다 Nvml::init() 을 수행합니다.
    //   50ms 폴링에 그대로 쓰면 측정 자체가 부하가 됩니다.
    //
    //  ── phase 라벨 ──
    //   최저점을 갱신한 순간의 라벨을 함께 기록하므로,
    //   "어느 필드의 어느 단계에서 튀었는가" 가 한 줄로 확정됩니다.
    // =====================================================================
    pub fn spawn_vram_sampler(&self, label: impl Into<String>) -> VramSampler {
        let label = label.into();
        let baseline = self.get_free_vram_mb();
        let stop = Arc::new(AtomicBool::new(false));
        let min_free = Arc::new(AtomicU64::new(u64::MAX));
        let phase = Arc::new(std::sync::Mutex::new(String::from("(start)")));
        let worst_phase = Arc::new(std::sync::Mutex::new(String::new()));

        if !self.is_cpu_mode {
            let gpu = self.device_config.gpu_id as u32;
            let s = stop.clone();
            let m = min_free.clone();
            let p = phase.clone();
            let w = worst_phase.clone();
            tokio::spawn(async move {
                let probe = crate::utils::GpuMemProbe::new();
                if probe.free_bytes(gpu as usize).is_none() { return; }
                while !s.load(XOrder::SeqCst) {
                    if let Some(free) = probe.free_bytes(gpu as usize) {
                        let f = free / (1024 * 1024);
                        if f < m.load(XOrder::SeqCst) {
                            m.store(f, XOrder::SeqCst);
                            if let (Ok(cur), Ok(mut dst)) = (p.lock(), w.lock()) {
                                *dst = cur.clone();
                            }
                        }
                    }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            });
        }

        VramSampler { stop, min_free, phase, worst_phase, label, baseline }
    }

    /// 임베딩 로드 비용을 기록합니다. `free_before` 는 이 함수를 부른 쪽이 정리를 끝낸 직후 값입니다.
    /// 피크는 뒤이은 `sync_crossover_phase` 가 실제 슬롯 상태로 셉니다.
    fn observe_embedding_cost(&self, free_before: u64) {
        if self.is_cpu_mode { return; }
        self.costs().record_embedding_load(free_before, self.get_free_vram_mb());
    }

    /// 🌟 [STRICT MEASURE] 생성 모델 비용은 깨끗한 기준선에서만 기록합니다 (`CostBook::record_generation_load`).
    fn observe_generation_cost(&self, kind: usize, free_before: u64, clean: bool) -> bool {
        if self.is_cpu_mode { return false; }
        self.costs()
            .record_generation_load(kind, free_before, self.get_free_vram_mb(), clean)
            .is_some()
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

        // 🌟 [FACT OVER LEDGER] 판정 근거는 언제나 실제 슬롯입니다.
        if self.embedding_resident().await {
            self.sync_crossover_phase().await;
            return Ok(());
        }

        if self.generation_resident().await {
            let free = self.get_free_vram_mb();
            let need = self.embedding_budget_mb();
            if free >= need {
                let coexists = self.costs().mark_coexist();
                println!(
                    "[CROSSOVER] 🤝 [COEXIST] {} | 자유 {}MB >= 임베딩 예산 {}MB → 생성 모델을 유지한 채 임베딩을 얹습니다. (스왑 회피 누적 {}회)",
                    reason, free, need, coexists
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
            self.costs().embedding_mb(),
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
        let gen_empty_before = !self.generation_resident().await;
        let embed_held = self.embedding_resident().await;
        let vision_held = self.siglip2_resident().await;
        let before = self.get_free_vram_mb();

        if coexist_ok {
            let coexists = self.costs().mark_coexist();
            println!(
                "[CROSSOVER] 🤝 [COEXIST] {} | 자유 {}MB >= 생성 예산 {}MB → 임베딩을 유지한 채 {:?} 를 올립니다. (스왑 회피 누적 {}회)",
                reason, before, self.generation_budget_mb(size), size,
                coexists
            );
            match size {
                ModelSize::Qwen => self.ensure_generator_ext(ModelSize::Qwen, false, false).await?,
                ModelSize::Qwen3 => self.ensure_qwen3().await?,
                ModelSize::Qwen3_5 => self.ensure_qwen3_5(false).await?,
            };
        } else {
            self.secure_vram_relay(size, session_id, cancel, prefill, kv_name).await?;
        }

        let embed_after = self.embedding_resident().await;
        let vision_after = self.siglip2_resident().await;
        let clean = gen_empty_before && embed_after == embed_held && vision_after == vision_held;
        let measured = self.observe_generation_cost(Self::gen_kind(size), before, clean);
        let phase = self.sync_crossover_phase().await;
        println!(
            "[CROSSOVER] 🧠 [GENERATION PHASE] {} | {:?} 상주 확정 (phase={} | 생성 실측 {}MB{} | 자유 {}MB)",
            reason,
            size,
            phase,
            Self::measured_generation_mb(Self::gen_kind(size)),
            if measured && embed_held { " · 임베딩 상주 상태에서 이번 적재로 실측" } else { "" },
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
        self.costs().report()
    }

    /// 🌟 activation 여유를 관측합니다.
    ///  대량 임베딩 배치 직후에 호출하면, 가중치 위에 얼마나 더 쓰였는지가
    ///  누적 최대값으로 학습되어 다음 예산 판정이 정확해집니다.
    pub fn observe_activation_headroom(&self, free_before_batch: u64) {
        if self.is_cpu_mode { return; }
        self.costs().observe_activation(free_before_batch, self.get_free_vram_mb());
    }
}
//! 상주 상태 · 실측 장부 · 전환 구간 · 슬롯 가드 (model/residency.rs) · 요약 패스 진입 판정 (lang_llm). 장치 없이 돕니다.

use std::sync::Arc;
use std::time::Duration;

use tauri_app_lib::model::lang_llm::{synthesis_entry, SynthesisEntry};
use tauri_app_lib::model::residency::{CostBook, HoldGate, Residency, SlotLease, GEN_KIND_LANG4B};
use tokio::sync::Mutex as TokioMutex;

/// 폐기율 게이트가 닫힌 요약 패스: 엔진이 이미 있으면 관측, 없으면 LANG ROUTE 와 같은 주기(10번째)로만 올립니다.
#[test]
fn synthesis_entry_loads_only_to_observe() {
    let axis = "test.synthesis_entry|list_synthesis";
    // 엔진이 이미 올라와 있으면 전환 없이 관측합니다 (차례를 세지 않습니다).
    assert_eq!(synthesis_entry(true, false, axis), (SynthesisEntry::ObserveResident, 0));
    // 경로 판정이 관측용으로 연 차례면 그 차례를 따릅니다 (두 주기가 엇갈려 관측이 끊기지 않게).
    assert_eq!(synthesis_entry(false, true, axis), (SynthesisEntry::ObserveProbe, 0));
    // 엔진이 없으면 10번째마다만 올립니다.
    let seen: Vec<(SynthesisEntry, u64)> = (0..20).map(|_| synthesis_entry(false, false, axis)).collect();
    assert_eq!(seen[0], (SynthesisEntry::Skip, 9));
    assert_eq!(seen[8], (SynthesisEntry::Skip, 1));
    assert_eq!(seen[9], (SynthesisEntry::ObserveProbe, 0));
    assert_eq!(seen[10], (SynthesisEntry::Skip, 9));
    assert_eq!(seen.iter().filter(|(e, _)| *e == SynthesisEntry::ObserveProbe).count(), 2);
    // 축마다 따로 셉니다.
    assert_eq!(synthesis_entry(false, false, "test.synthesis_entry|detail_synthesis"), (SynthesisEntry::Skip, 9));
}

#[test]
fn residency_is_derived_from_slots() {
    assert_eq!(Residency::from_slots(false, false), Residency::Idle);
    assert_eq!(Residency::from_slots(false, true), Residency::Embedding);
    assert_eq!(Residency::from_slots(true, false), Residency::Generation);
    assert_eq!(Residency::from_slots(true, true), Residency::Coexist);

    assert!(Residency::Coexist.has_embedding() && Residency::Coexist.has_generation());
    assert!(Residency::Embedding.has_embedding() && !Residency::Embedding.has_generation());
    assert!(!Residency::Generation.has_embedding() && Residency::Generation.has_generation());
    assert!(!Residency::Idle.has_embedding() && !Residency::Idle.has_generation());
    assert_eq!(Residency::Coexist.to_string(), "COEXIST");
}

#[test]
fn embedding_budget_uses_disk_only_until_measured() {
    let book = CostBook::new();
    assert_eq!(book.embedding_budget_mb(|| 180), 180, "실측이 없으면 디스크 추정");
    assert_eq!(book.embedding_budget_mb(|| 0), 1, "디스크 추정이 0 이어도 최소 1MB");

    assert_eq!(book.record_embedding_load(4000, 3850), Some(150));
    let budget = book.embedding_budget_mb(|| panic!("실측이 있으면 가중치 폴더를 훑지 않습니다"));
    assert_eq!(budget, 150);

    book.observe_activation(3850, 3600);
    assert_eq!(book.activation_headroom_mb(), 250);
    assert_eq!(book.embedding_budget_mb(|| 0), 150 + 250, "activation 여유를 더합니다");
}

#[test]
fn embedding_cost_ignores_contaminated_measurement() {
    let book = CostBook::new();
    // 로드 중 다른 정리가 일어나 free 가 늘었으면 비용으로 기록하지 않습니다.
    assert_eq!(book.record_embedding_load(3000, 3400), None);
    assert_eq!(book.record_embedding_load(3000, 3000), None);
    assert_eq!(book.embedding_mb(), 0);
}

#[test]
fn generation_cost_is_recorded_only_on_a_clean_baseline() {
    let book = CostBook::new();
    assert_eq!(book.record_generation_load(1, 6000, 5100, false), None, "기준선이 오염되면 기록하지 않습니다");
    assert_eq!(book.generation_mb(1), 0);
    assert_eq!(book.record_generation_load(1, 6000, 6200, true), None, "free 가 늘었으면 기록하지 않습니다");

    assert_eq!(book.record_generation_load(1, 6000, 5100, true), Some(900));
    assert_eq!(book.generation_mb(1), 900);
    assert_eq!(book.last_generation_mb(), 900);
    assert_eq!(book.generation_mb(99), 0, "모르는 종류는 0");
    assert_eq!(book.record_generation_load(99, 6000, 5000, true), Some(1000), "모르는 종류도 마지막 실측으로는 남습니다");
    assert_eq!(book.last_generation_mb(), 1000);
}

#[test]
fn generation_budget_never_drops_below_disk_size() {
    let book = CostBook::new();
    assert_eq!(book.generation_budget_mb(2, 1500), 1500, "실측이 없으면 디스크");
    book.record_generation_load(2, 8000, 6100, true);
    assert_eq!(book.generation_budget_mb(2, 1500), 1900, "실측이 크면 실측");
    assert_eq!(book.generation_budget_mb(2, 2500), 2500, "디스크가 크면 디스크 (과소평가 금지)");
    // 이 종류의 실측이 없으면 마지막으로 잰 생성 모델 값을 씁니다.
    assert_eq!(book.generation_budget_mb(GEN_KIND_LANG4B, 700), 1900);
    book.observe_activation(6100, 5900);
    assert_eq!(book.generation_budget_mb(2, 1500), 1900 + 200);
}

#[test]
fn peak_counts_what_the_slots_hold() {
    let book = CostBook::new();
    book.record_embedding_load(5000, 4850);
    book.record_generation_load(1, 4850, 3950, true);

    assert_eq!(book.observe_residency(Residency::Idle), 0);
    assert_eq!(book.observe_residency(Residency::Embedding), 150);
    assert_eq!(book.observe_residency(Residency::Generation), 900);
    assert_eq!(book.peak_mb(), 900);
    assert_eq!(book.observe_residency(Residency::Coexist), 1050);
    assert_eq!(book.observe_residency(Residency::Idle), 0);
    assert_eq!(book.peak_mb(), 1050, "피크는 최대값을 유지합니다");
}

#[test]
fn counters_feed_the_report() {
    let book = CostBook::new();
    assert_eq!(book.mark_coexist(), 1);
    assert_eq!(book.mark_coexist(), 2);
    book.mark_swap();
    assert_eq!(book.swap_count(), 1);
    assert_eq!(book.coexist_count(), 2);
    let report = book.report();
    assert!(report.contains("스왑 1회") && report.contains("동시상주 회피 2회"), "{report}");
}

#[tokio::test]
async fn hold_gate_does_not_wait_when_free() {
    let gate = Arc::new(HoldGate::default());
    assert!(!gate.is_held());
    assert!(gate.wait_released(Duration::from_millis(1)).await);
}

#[tokio::test]
async fn hold_gate_wakes_the_waiter_on_the_last_release() {
    let gate = Arc::new(HoldGate::default());
    let first = gate.hold();
    let second = gate.hold();
    assert!(gate.is_held());

    let waiter = {
        let gate = Arc::clone(&gate);
        tokio::spawn(async move {
            let started = std::time::Instant::now();
            let released = gate.wait_released(Duration::from_secs(30)).await;
            (released, started.elapsed())
        })
    };
    tokio::time::sleep(Duration::from_millis(30)).await;
    drop(first);
    tokio::time::sleep(Duration::from_millis(30)).await;
    assert!(!waiter.is_finished(), "가드가 하나라도 남으면 계속 기다립니다");
    drop(second);

    let (released, waited) = tokio::time::timeout(Duration::from_secs(5), waiter)
        .await
        .expect("마지막 가드가 풀리면 바로 깨어나야 합니다")
        .expect("대기 작업");
    assert!(released);
    assert!(waited < Duration::from_secs(5), "{waited:?}");
    assert!(!gate.is_held());
}

#[tokio::test]
async fn hold_gate_gives_up_at_the_limit() {
    let gate = Arc::new(HoldGate::default());
    let _held = gate.hold();
    assert!(!gate.wait_released(Duration::from_millis(40)).await);
}

#[tokio::test]
async fn slot_lease_keeps_the_model_until_dropped() {
    let slot: Arc<TokioMutex<Option<u32>>> = Arc::new(TokioMutex::new(Some(7)));
    let lease = SlotLease::new(Arc::clone(&slot).lock_owned().await).expect("찬 슬롯");
    assert_eq!(*lease, 7);

    // 다른 작업의 '내리기' 는 가드가 풀릴 때까지 기다립니다.
    let unload = {
        let slot = Arc::clone(&slot);
        tokio::spawn(async move { slot.lock().await.take() })
    };
    tokio::time::sleep(Duration::from_millis(40)).await;
    assert!(!unload.is_finished(), "가드를 쥔 동안에는 슬롯을 비울 수 없습니다");
    assert_eq!(*lease, 7);

    drop(lease);
    let taken = tokio::time::timeout(Duration::from_secs(5), unload)
        .await
        .expect("가드가 풀리면 내리기가 진행됩니다")
        .expect("내리기 작업");
    assert_eq!(taken, Some(7));
    assert!(SlotLease::new(Arc::clone(&slot).lock_owned().await).is_none(), "빈 슬롯은 빌려 주지 않습니다");
}

//! 모델이 띄우는 비동기 작업 (KV 슬롯 반환 · KV 블록 굽기 · 읽기 · 지연 해제). 예전에는 `tauri::async_runtime::spawn` 이었습니다.
//!
//! 그 함수는 부른 스레드의 런타임과 상관없이 Tauri 의 전역 런타임에 띄웁니다(동기 `Drop` · rayon 스레드에서도 됨).
//! 앱은 [`set_spawner`] 로 그 함수를 등록해 예전과 같은 런타임을 쓰고, 등록이 없으면(통합 시험 · 헤드리스) 이 crate 가 처음
//! 쓸 때 만드는 멀티스레드 런타임에 띄웁니다. Tauri 가 `async_runtime::set` 없이 스스로 만드는 기본 런타임과 같은 성질입니다.

use std::future::Future;
use std::pin::Pin;
use std::sync::OnceLock;

/// 띄울 작업 (결과는 버립니다: 예전 호출부도 JoinHandle 을 쓰지 않았습니다)
pub type Task = Pin<Box<dyn Future<Output = ()> + Send + 'static>>;

static SPAWNER: OnceLock<fn(Task)> = OnceLock::new();
static FALLBACK: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

/// 앱의 전역 런타임에 띄우는 함수를 등록합니다. 처음 한 번만 받습니다.
pub fn set_spawner(spawner: fn(Task)) {
    let _ = SPAWNER.set(spawner);
}

/// 어느 스레드에서 불러도 되는 spawn: 등록된 런타임, 없으면 이 crate 의 전역 런타임.
pub fn spawn<F>(fut: F)
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let task: Task = Box::pin(async move {
        let _ = fut.await;
    });
    if let Some(spawner) = SPAWNER.get() {
        spawner(task);
        return;
    }
    FALLBACK
        .get_or_init(|| tokio::runtime::Runtime::new().expect("[LOGIS-LLM] tokio 런타임을 만들 수 없습니다"))
        .spawn(task);
}

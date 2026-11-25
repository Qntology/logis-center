//! 문서 언어 판정은 crates/logis-bias 로 옮겼습니다.
pub use logis_bias::lang_utils::*;

/// Capture current [`LangContext`](logis_llm::models::common::vocab_trim::LangContext) and
/// propagate it across `tokio::task::spawn_blocking`.
pub fn spawn_blocking_with_lang<F, R>(f: F) -> tokio::task::JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let ctx = logis_llm::models::common::vocab_trim::LangContext::capture_current();
    tokio::task::spawn_blocking(move || {
        let _guard = ctx.attach();
        f()
    })
}

/// Capture current [`LangContext`](logis_llm::models::common::vocab_trim::LangContext) and
/// propagate it across `tokio::spawn`.
pub fn spawn_async_with_lang<F>(fut: F) -> tokio::task::JoinHandle<F::Output>
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    let ctx = logis_llm::models::common::vocab_trim::LangContext::capture_current();
    tokio::spawn(ctx.scope_future(fut))
}

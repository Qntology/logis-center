use tokio::sync::Notify;
use once_cell::sync::Lazy;
use std::sync::atomic::{AtomicBool, Ordering};

// [UI-SYNC] Instant notification system to wake up the worker
pub static UI_READY_SIGNAL: Lazy<Notify> = Lazy::new(|| Notify::new());
pub static TASK_QUEUED_SIGNAL: Lazy<Notify> = Lazy::new(|| Notify::new());
pub static UI_READY_FLAG: AtomicBool = AtomicBool::new(false);

pub fn mark_ui_ready() {
    UI_READY_FLAG.store(true, Ordering::SeqCst);
    UI_READY_SIGNAL.notify_waiters(); // Wake up any sleeping tasks instantly
    println!("[Scheduler] UI signaled ready. Background worker woke up.");
}

pub fn notify_new_task() {
    TASK_QUEUED_SIGNAL.notify_waiters();
}
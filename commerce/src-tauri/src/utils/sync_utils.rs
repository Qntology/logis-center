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

// --- GLOBAL EXTRACTION CONTROL (misc_utils.rs에서 이동) ---
pub fn is_extraction_stopped() -> bool {
    crate::utils::paths::get_stop_signal_file().exists()
}

pub fn set_extraction_stop_signal(stopped: bool) {
    let file = crate::utils::paths::get_stop_signal_file();
    if stopped {
        let _ = std::fs::File::create(file);
    } else {
        let _ = std::fs::remove_file(file);
    }
}
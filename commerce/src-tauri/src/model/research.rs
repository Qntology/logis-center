use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use serde_json::json;

impl crate::model::LogisModel {

    // --- Ported from Python (search_engine.py) ---
    // --- Ported from Python (logic.py) ---
    pub async fn run_deep_research(&self, query: String, context_data: String, app_handle: &tauri::AppHandle, cancel_token: Option<Arc<AtomicBool>>) -> anyhow::Result<String> {
        let mut status_history = format!("### 🔍 Deep Research: '{}'\n\n", query);

        // 1. Context Gathering
        status_history.push_str("✅ Context gathered.\n\n");
        // [LOG-ONLY]
        crate::utils::logger::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

        // 2. Multi-step reasoning loop
        let steps = vec![
            "Analyzing relationships and implications...",
            "Evaluating cross-document consistency...",
            "Synthesizing final intelligence report..."
        ];

        for step in steps.iter() {
            status_history.push_str(&format!("**⏳ {}**\n", step));
            crate::utils::logger::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

            let prompt = format!("Given this context: {}\n\nTask: {}\nQuery: {}\n\nProvide deep insight for this specific step.", context_data, step, query);
            
            let step_result = self.run_inference_text(prompt, None, cancel_token.clone(), None, None).await?;
            
            let short_res = if step_result.len() > 200 { &step_result[..200] } else { &step_result };
            status_history.push_str(&format!("> {}...\n\n", short_res.replace("\n", " ")));
            crate::utils::logger::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

            crate::models::qwen::generate::wait_for_global_io().await;
            
            // 🌟 [신규 추가] GPU 비동기 연산 찌꺼기 강제 동기화
            if !self.is_cpu_mode {
                let dev = self.device_config.device.clone();
                let _ = tokio::task::spawn_blocking(move || {
                    if dev.is_cuda() { let _ = dev.synchronize(); }
                }).await;
            }
            
            #[cfg(target_os = "windows")]
            unsafe {
                use windows_sys::Win32::System::Threading::GetCurrentProcess;
                use windows_sys::Win32::System::Memory::{SetProcessWorkingSetSizeEx, QUOTA_LIMITS_HARDWS_MIN_DISABLE, QUOTA_LIMITS_HARDWS_MAX_DISABLE};
                let _ = SetProcessWorkingSetSizeEx(GetCurrentProcess(), usize::MAX, usize::MAX, QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_DISABLE);
            }
            #[cfg(target_os = "linux")]
            unsafe { extern "C" { fn malloc_trim(pad: usize) -> i32; } malloc_trim(0); }
            #[cfg(target_os = "macos")]
            unsafe { extern "C" { fn malloc_zone_pressure_relief(zone: *mut std::ffi::c_void, goal: usize) -> usize; } malloc_zone_pressure_relief(std::ptr::null_mut(), 0); }
            
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }

        // 3. Final Report
        status_history.push_str("### 📊 Final Research Report\n\n");
        let final_prompt = format!("CONTEXT: {}\nQUERY: {}\n\nBased on the above steps, generate a comprehensive final trade intelligence report.", context_data, query);
        
        let report = self.run_inference_text(final_prompt, None, cancel_token, None, None).await?;
        status_history.push_str(&report);
        
        // [LOG-ONLY]
        crate::utils::logger::log_task_progress(app_handle, "research", &json!({ "text": status_history }));

        Ok(report.to_string())
    }

}
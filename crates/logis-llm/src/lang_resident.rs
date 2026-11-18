//! 지금 올라 있는 언어 엔진 변형 (예전 앱 model/lang_llm.rs 의 `RESIDENT`). 앱의 언어 엔진 관리가 바꾸고,
//! Qwen3.5 모델이 KV 계획 로그에 4B · 2B 를 적을 때 읽습니다.

use once_cell::sync::Lazy;
use std::sync::Mutex;

static RESIDENT: Lazy<Mutex<Option<String>>> = Lazy::new(|| Mutex::new(None));

pub fn resident_variant() -> Option<String> {
    RESIDENT.lock().ok().and_then(|g| g.clone())
}

pub fn set_resident_variant(v: Option<String>) {
    if let Ok(mut g) = RESIDENT.lock() {
        *g = v;
    }
}

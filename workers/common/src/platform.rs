//! 런타임 의존: 시각 · 난수 · 기다림 · 로그. wasm 에서는 JS 의 `Date.now` · `Math.random` · `setTimeout` · `console.log`,
//! 호스트(단위 시험)에서는 고정값 · 표준 출력.

#[cfg(target_arch = "wasm32")]
mod imp {
    /// 패닉 내용을 console 에 남긴다 (wasm 패닉은 그대로면 "unreachable" 만 보인다)
    pub fn install_panic_hook() {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            std::panic::set_hook(Box::new(|info| {
                worker::console_error!("panic: {}", info);
            }));
        });
    }

    pub fn now() -> f64 {
        js_sys::Date::now()
    }
    pub fn random() -> f64 {
        js_sys::Math::random()
    }
    pub fn log(line: &str) {
        worker::console_log!("{}", line);
    }
    pub async fn sleep(ms: f64) {
        let ms = if ms.is_finite() && ms > 0.0 { ms } else { 0.0 };
        worker::Delay::from(std::time::Duration::from_millis(ms as u64)).await;
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use std::cell::Cell;

    thread_local! {
        static SEED: Cell<u64> = const { Cell::new(0x9E37_79B9_7F4A_7C15) };
    }

    pub fn now() -> f64 {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as f64).unwrap_or(0.0)
    }
    /// 재현되는 의사 난수 (xorshift)
    pub fn random() -> f64 {
        SEED.with(|s| {
            let mut x = s.get();
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            s.set(x);
            (x >> 11) as f64 / (1u64 << 53) as f64
        })
    }
    pub fn log(line: &str) {
        eprintln!("{line}");
    }
    pub async fn sleep(_ms: f64) {}
    pub fn install_panic_hook() {}
}

pub use imp::{install_panic_hook, log, now, random, sleep};

/// 난수 바이트 (hashId(undefined) 의 무작위 키)
pub fn random_bytes(n: usize) -> Vec<u8> {
    (0..n).map(|_| (random() * 256.0) as u8).collect()
}

/// `console.log(a, b, …)` — 객체는 JSON 으로
#[macro_export]
macro_rules! console_log {
    ($($arg:expr),+ $(,)?) => {{
        let parts: Vec<String> = vec![$($crate::platform::show(&$arg)),+];
        $crate::platform::log(&parts.join(" "));
    }};
}

/// console.log 에 찍을 꼴
pub fn show<T: Show + ?Sized>(x: &T) -> String {
    x.show()
}

pub trait Show {
    fn show(&self) -> String;
}

impl Show for str {
    fn show(&self) -> String {
        self.to_string()
    }
}
impl Show for &str {
    fn show(&self) -> String {
        self.to_string()
    }
}
impl Show for String {
    fn show(&self) -> String {
        self.clone()
    }
}
impl Show for crate::jsv::V {
    fn show(&self) -> String {
        match self {
            crate::jsv::V::Str(s) => s.clone(),
            other => crate::jsv::stringify(other).unwrap_or_else(|| "undefined".into()),
        }
    }
}
impl Show for crate::jsv::Throw {
    fn show(&self) -> String {
        self.to_string()
    }
}
impl Show for f64 {
    fn show(&self) -> String {
        crate::jsv::num_to_string(*self)
    }
}
impl Show for usize {
    fn show(&self) -> String {
        self.to_string()
    }
}
impl Show for bool {
    fn show(&self) -> String {
        self.to_string()
    }
}

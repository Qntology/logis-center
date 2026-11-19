//! logis-center Cloudflare Workers 공용 층.
//!
//! 원본 TypeScript 워커(`workers/proxy` · `workers/apac` 의 옛 `src/index.ts`)를 같은 동작으로 옮기기 위한 것이다.
//! 동적인 JS 객체를 [`jsv::V`] 로 다루고(참조 공유 · undefined/null · 던지는 속성 읽기), 바인딩은 [`js`] 가 JS 그대로 부른다.

pub mod aws;
pub mod date;
pub mod dirty_json;
pub mod eth;
pub mod gzip;
pub mod jsv;
pub mod platform;
pub mod tables;
pub mod util;
pub mod web;

#[cfg(target_arch = "wasm32")]
pub mod js;

pub use jsv::{add, loose_eq, parse, parse_str, strict_eq, stringify, stringify_str, Throw, R, V};

//! 구역 함수가 원본의 `continue` · `break` 를 바깥 반복에 알리는 값

/// 구역이 끝난 뒤 둘러싼 반복이 할 일
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flow {
    /// 다음 문장으로 (구역이 끝까지 돌았다)
    Next,
    /// 원본의 `continue` — 둘러싼 반복의 다음 회차로
    Continue,
    /// 원본의 `break` — 둘러싼 반복을 나온다
    Break,
}

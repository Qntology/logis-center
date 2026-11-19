//! 구역이 끝난 뒤 처리기가 할 일

use logis_worker_common::web::Resp;

pub enum Step {
    /// 다음 문장으로
    Next,
    /// 원본의 `return new Response(...)`
    Return(Resp),
}

//! 저장소 봉투(envelope) 물리 컬럼. 예전 store.rs 에서 옮김 (자연어 변환이 봉투 칸을 문장에서 뺄 때 씀).

pub const ENVELOPE_COLUMNS: [&str; 11] = [
    "id", "type", "flag", "from", "to", "cc", "bcc", "ref", "mode",
    "created_at", "updated_at",
];

//! PUG · 질의 · 배정 도우미는 crates/logis-pug 로 옮겼습니다. 예전 경로(crate::utils::ai_utils)를 그대로 쓰고,
//! 임베딩 모델(LogisModel)이 필요한 뱅크 함수만 여기 남습니다.
pub use logis_pug::utils::ai_utils::*;

pub async fn status_key_banks(model: &crate::model::LogisModel, page_type: &str) -> Vec<(String, Vec<Vec<f32>>)> {
    let mut key_banks: Vec<(String, Vec<Vec<f32>>)> = Vec::new();
    for k in enum_status_keys(page_type).iter() {
        let phrases = status_key_phrases(k);
        let e = model.get_embedding_batch(phrases.clone()).await
            .unwrap_or_else(|_| vec![vec![0.0; 384]; phrases.len()]);
        key_banks.push((k.to_string(), e));
    }
    key_banks
}

pub async fn status_pivot_bank(
    model: &crate::model::LogisModel,
    page_type: &str,
    doc_lang: &str,
) -> Option<StatusPivot> {
    let lang = crate::utils::bias_schema::lang_code_of(doc_lang);
    if lang == "en" {
        return None;
    }
    let src = status_bank_list(&lang, page_type);
    let dst = status_bank_list("en", page_type);
    if src.len() < 3 || src.len() != dst.len() {
        return None;
    }
    let src_embs = model.get_embedding_batch(src.clone()).await.ok()?;
    let dst_embs = model.get_embedding_batch(dst.clone()).await.ok()?;
    let align_z = bank_alignment_z(&src_embs, &dst_embs)?;
    Some(StatusPivot { lang, src, dst, align_z })
}

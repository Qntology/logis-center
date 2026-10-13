// =====================================================================
// 🌟 [EMBED-BATCHED] 진짜 배치 순전파 (CUDA / ROCm 전용)
// ---------------------------------------------------------------------
//  ── 무엇이 문제였나 ──
//   embed_batch 는 3스레드가 embed() 를 1건씩 돌리는 구조라,
//   텍스트 1건 = 배치 1 순전파 = 커널 ~730회 + 동기 D2H 1회 였습니다.
//   뱅크 구(중앙값 4토큰)는 GPU 연산보다 커널 발사 비용이 훨씬 큽니다.
//
//  ── 해결 ──
//   ① 토큰 길이가 64 이하인 텍스트는 '정확히 같은 길이' 끼리 묶습니다.
//      패딩·마스크가 없으므로 행마다 같은 연산이고, 요소별 연산과 행 단위
//      리덕션은 배치 크기와 무관하게 같은 값을 냅니다.
//   ② 더 긴 텍스트는 길이를 32 단위로 올림한 버킷에 정렬해 넣고,
//      패딩 위치에는 가산 마스크 -inf 를 더합니다(exp(-inf)=0).
//      풀링은 행마다 실제 길이만큼만 잘라서(per-text 와 같은 리덕션) 합니다.
//   ③ 한 마이크로배치의 행 수는 길이 L 만으로 정해집니다:
//        B(L) = min(max_rows, token_budget / L, attn_budget / L²)
//   ④ 각 마이크로배치의 정규화 벡터 [B, 384] 는 GPU 에 모아 두고
//      호출당 한 번만 D2H 합니다(커널 발사가 비동기라 다음 배치 준비와 겹칩니다).
//   ⑤ 행이 1개뿐인 마이크로배치는 패딩 없이 per-text 와 똑같은 연산열로 돌립니다.
//      그래서 단건 호출(get_embedding)은 기존 embed() 와 비트 단위로 같습니다.
//
//  ── 정확도 ──
//   cuBLAS GEMM 은 M = B·L 과 M = L 에서 다른 커널(또는 split-K)을 고를 수 있어
//   BF16 1ulp 수준 차이가 날 수 있습니다. 패딩 버킷은 softmax / attn·V 의 리덕션
//   길이가 달라집니다. 그래서 이 경로는 별도 캐시 공간(FORWARD_TAG_BATCHED)을 쓰고,
//   tests/it/app_embed.rs 의 GPU 검증(최소 코사인 · 최대 절대차)으로 확인합니다.
//   LOGIS_EMBED_BATCHED=0 이면 기존 per-text 경로로 돌아갑니다.
// =====================================================================
use super::{aggregate_chunk_vectors, EmbeddingModel, EMBED_CHUNK_TOKENS, EMBED_DIM};
use anyhow::Result;
use candle_core::{Device, Tensor};
use std::collections::BTreeMap;

/// 배치 순전파 크기 정책. 값은 길이 L 만의 함수라 같은 입력이면 같은 모양이 됩니다.
#[derive(Debug, Clone, Copy)]
pub struct BatchOpts {
    /// 마이크로배치 하나의 최대 토큰 수 (B·L).
    pub token_budget: usize,
    /// attention 점수 텐서 상한 (B·L²). [B,12,L,L] BF16 이 이 값 × 24 바이트입니다.
    pub attn_budget: usize,
    /// 마이크로배치 최대 행 수.
    pub max_rows: usize,
    /// 이 길이 이하는 정확한 길이끼리만 묶습니다(패딩·마스크 없음).
    pub exact_len_max: usize,
    /// 그보다 긴 텍스트의 버킷 길이 올림 단위.
    pub pad_step: usize,
}

impl Default for BatchOpts {
    fn default() -> Self {
        let token_budget = std::env::var("LOGIS_EMBED_TOKEN_BUDGET")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v >= 64)
            .unwrap_or(4096);
        Self { token_budget, attn_budget: 1 << 19, max_rows: 256, exact_len_max: 64, pad_step: 32 }
    }
}

impl BatchOpts {
    /// 길이 L 버킷의 마이크로배치 행 수.
    pub fn rows_for(&self, len: usize) -> usize {
        let l = len.max(1);
        let by_tok = self.token_budget / l;
        let by_attn = self.attn_budget / (l * l);
        by_tok.min(by_attn).min(self.max_rows).max(1)
    }

    /// 토큰 길이 → 버킷 길이.
    pub fn bucket_len(&self, len: usize) -> usize {
        if len <= self.exact_len_max {
            len
        } else {
            let step = self.pad_step.max(1);
            (((len + step - 1) / step) * step).min(EMBED_CHUNK_TOKENS.max(len))
        }
    }
}

/// 512 토큰 청크 하나.
struct Seq {
    text_idx: usize,
    ids: Vec<u32>,
}

/// embed_many 실행 통계 (로그·테스트용).
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchRunStats {
    pub texts: usize,
    pub seqs: usize,
    pub micro_batches: usize,
    pub padded_batches: usize,
    pub real_tokens: usize,
    pub computed_tokens: usize,
}

impl EmbeddingModel {
    /// 🌟 [EMBED-BATCHED] 여러 텍스트를 배치 순전파로 임베딩합니다.
    /// 결과 순서·길이는 입력과 같고, 각 벡터는 embed() 와 같은 규약
    /// (512 토큰 청크 평균 풀링 → L2 정규화 → 청크 평균 → 재정규화)입니다.
    pub fn embed_many(&self, texts: &[String], opts: &BatchOpts) -> Result<Vec<Vec<f32>>> {
        self.embed_many_with_stats(texts, opts).map(|(v, _)| v)
    }

    pub fn embed_many_with_stats(&self, texts: &[String], opts: &BatchOpts) -> Result<(Vec<Vec<f32>>, BatchRunStats)> {
        let mut stats = BatchRunStats { texts: texts.len(), ..Default::default() };
        if texts.is_empty() {
            return Ok((Vec::new(), stats));
        }
        // ⑤ 단건 호출(get_embedding)은 per-text embed() 그대로 — 512 토큰을 넘는 텍스트의
        //    두 청크가 같은 버킷에 들어가 패딩 배치가 되는 경우까지 포함해 항상 비트 단위로 같습니다.
        if texts.len() == 1 {
            let v = self.embed(&texts[0])?;
            stats.micro_batches = 1;
            return Ok((vec![v], stats));
        }

        // ── 1. 토큰화 (embed() 와 같은 encode(text, true) · 텍스트별로) ──
        let mut seqs: Vec<Seq> = Vec::new();
        let mut empty_text = vec![false; texts.len()];
        for (ti, text) in texts.iter().enumerate() {
            let enc = self.tokenizer.encode(text.as_str(), true).map_err(anyhow::Error::msg)?;
            let ids = enc.get_ids();
            if ids.is_empty() {
                empty_text[ti] = true;
                continue;
            }
            for chunk in ids.chunks(EMBED_CHUNK_TOKENS) {
                stats.real_tokens += chunk.len();
                seqs.push(Seq { text_idx: ti, ids: chunk.to_vec() });
            }
        }
        stats.seqs = seqs.len();

        // ── 2. 버킷 (버킷 길이 → 청크 번호, 입력 순서 유지) ──
        let mut buckets: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (si, s) in seqs.iter().enumerate() {
            buckets.entry(opts.bucket_len(s.ids.len())).or_default().push(si);
        }

        // ── 3. 마이크로배치 순전파 (결과는 GPU 에 모아 둠) ──
        let mut outs: Vec<Tensor> = Vec::new();
        let mut order: Vec<usize> = Vec::with_capacity(seqs.len());
        for (&blen, members) in buckets.iter() {
            let rows = opts.rows_for(blen);
            for group in members.chunks(rows) {
                let pooled = match self.forward_group(&seqs, group, blen, &mut stats) {
                    Ok(t) => t,
                    Err(e) if group.len() > 1 => {
                        // OOM 등: 이 그룹만 행 단위(per-text 와 같은 연산열)로 다시 돌립니다.
                        println!(
                            "[EMBED-BATCHED] ⚠️ 마이크로배치 {}행 × {}토큰 실패({}). 이 그룹만 1행씩 다시 계산합니다.",
                            group.len(), blen, e
                        );
                        let mut parts = Vec::with_capacity(group.len());
                        for si in group {
                            parts.push(self.forward_group(&seqs, std::slice::from_ref(si), blen, &mut stats)?);
                        }
                        Tensor::cat(&parts, 0)?
                    }
                    Err(e) => return Err(e),
                };
                outs.push(pooled);
                order.extend_from_slice(group);
            }
        }

        // ── 4. 호출당 D2H 1회 ──
        let mut seq_vecs: Vec<Vec<f32>> = vec![Vec::new(); seqs.len()];
        if !outs.is_empty() {
            let all = if outs.len() == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 0)? };
            let rows: Vec<Vec<f32>> = all
                .to_device(&Device::Cpu)?
                .to_dtype(candle_core::DType::F32)?
                .to_vec2()?;
            for (row, si) in rows.into_iter().zip(order.iter()) {
                seq_vecs[*si] = row;
            }
        }

        // ── 5. 텍스트별 청크 합산 (embed() 와 같은 f32 산술) ──
        let mut per_text: Vec<Vec<&[f32]>> = vec![Vec::new(); texts.len()];
        for (si, s) in seqs.iter().enumerate() {
            per_text[s.text_idx].push(seq_vecs[si].as_slice());
        }
        let out = per_text
            .iter()
            .enumerate()
            .map(|(ti, chunks)| {
                if empty_text[ti] || chunks.is_empty() {
                    vec![0.0f32; EMBED_DIM]
                } else {
                    aggregate_chunk_vectors(chunks)
                }
            })
            .collect();
        Ok((out, stats))
    }

    /// 마이크로배치 하나를 순전파하고 정규화된 [g, H] 텐서를 돌려줍니다(GPU 상주).
    fn forward_group(&self, seqs: &[Seq], group: &[usize], blen: usize, stats: &mut BatchRunStats) -> Result<Tensor> {
        let g = group.len();
        stats.micro_batches += 1;

        // ⑤ 1행: 패딩 없이 per-text embed() 와 같은 연산열.
        if g == 1 {
            let ids = &seqs[group[0]].ids;
            stats.computed_tokens += ids.len();
            let input = Tensor::new(ids.as_slice(), &self.device)?;
            let hidden = self.model.forward(&input, &self.device)?;
            return Self::pool_single(&hidden);
        }

        let lens: Vec<usize> = group.iter().map(|&si| seqs[si].ids.len()).collect();
        let exact = lens.iter().all(|&l| l == blen);
        let l = if exact { blen } else { blen.max(*lens.iter().max().unwrap_or(&blen)) };
        stats.computed_tokens += g * l;

        let mut flat: Vec<u32> = Vec::with_capacity(g * l);
        for &si in group {
            let ids = &seqs[si].ids;
            flat.extend_from_slice(ids);
            flat.extend(std::iter::repeat(self.pad_id).take(l - ids.len()));
        }
        let input = Tensor::from_vec(flat, g * l, &self.device)?;

        if exact {
            // ① 정확한 길이 버킷: 마스크 없음, 배치 풀링.
            let hidden = self.model.forward_batch(&input, g, l, None)?;
            let sum = hidden.sum(1)?; // [g, H]
            let mean = (sum / (l as f64))?;
            let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?; // [g, 1]
            return Ok(mean.broadcast_div(&norm)?);
        }

        // ② 패딩 버킷: 가산 마스크 [g,1,1,L] (유효 0 / 패딩 -inf).
        stats.padded_batches += 1;
        let mut mask: Vec<f32> = Vec::with_capacity(g * l);
        for &len in &lens {
            mask.extend(std::iter::repeat(0.0f32).take(len));
            mask.extend(std::iter::repeat(f32::NEG_INFINITY).take(l - len));
        }
        let mask = Tensor::from_vec(mask, (g, 1, 1, l), &self.device)?.to_dtype(self.dtype)?;
        let hidden = self.model.forward_batch(&input, g, l, Some(&mask))?;
        // 풀링은 행마다 실제 길이만큼만 (per-text 와 같은 리덕션 길이).
        let mut rows = Vec::with_capacity(g);
        for (r, &len) in lens.iter().enumerate() {
            let h = hidden.narrow(0, r, 1)?.narrow(1, 0, len)?.contiguous()?;
            rows.push(Self::pool_single(&h)?);
        }
        Ok(Tensor::cat(&rows, 0)?)
    }
}

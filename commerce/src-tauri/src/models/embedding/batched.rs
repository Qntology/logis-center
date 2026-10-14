// =====================================================================
// 🌟 [EMBED-BATCHED] 진짜 배치 순전파 (CUDA / ROCm / CPU / Vulkan / Metal)
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
//   ⑤ 🌟 [EMBED-BATCHED / FIXED-SHAPE] 모든 마이크로배치는 항상 B(L) 행 × L 토큰
//      고정 모양으로 돌립니다. 모자란 행은 첫 행의 복사본(더미)으로 채우고 출력은 버립니다.
//      단건 호출도, 1행만 남은 그룹도 같은 모양을 거칩니다. 패딩 버킷(L>64)은 구성원이
//      우연히 모두 버킷 길이여도 항상 마스크 경로로 돌립니다.
//      그래서 한 텍스트의 벡터는 그 텍스트(청크) 만의 함수이고, 같은 호출에 어떤
//      텍스트가 함께 들어왔는지, 먼저 캐시한 쪽이 어느 호출이었는지와 무관합니다
//      (고정 모양이면 GEMM 커널 선택이 같고, 행 사이 연산이 섞이지 않습니다).
//      B(L)·L ≤ token_budget(기본 1024) 이라 단건 호출의 연산량도 1024 토큰 이하입니다.
//   ⑥ 🌟 [EMBED-BATCHED / DYNAMIC] CPU 는 고정 모양을 쓰지 않고 실제 행만 묶습니다(fixed_shape=false).
//      CPU 는 연산량이 곧 시간이라 더미 행이 그대로 손해입니다(실측, Ryzen 5600X · 320건:
//      고정 모양 예산 1024 는 per-text 의 0.4배 속도, 단건 589ms vs 16.7ms). 대신 CPU gemm 은
//      행 결과가 M(=B·L) 과 행 위치에 따라 바뀌지 않아(같은 텍스트가 per-text M=L 과 배치 M=B·L 에서
//      303/303 비트 동일), 행 수가 호출마다 달라도 한 텍스트의 벡터는 그 텍스트만의 함수입니다.
//      패딩 버킷도 쓰지 않고(exact_len_max = 512) 정확한 길이끼리만 묶어, CPU 에서는 배치 결과가
//      per-text 와 320/320 비트 동일합니다(전체 1.6배 · 짧은 구 2.8배, 단건 호출은 같은 비용).
//      Vulkan 은 연산마다 GPU/CPU 경로를 크기·가중치 사용 횟수로 고르는 하이브리드라, 이 구간을
//      포크의 고정 실행 모드(shaders::scoped_exec_mode, 기본 Gpu: 가중치 행렬곱은 항상 GPU GEMM ·
//      나머지는 CPU 경로)로 묶습니다. 실측(RX 6600): 전체 1.6배 · 짧은 구 3.5배, 구성 무관 비트 동일.
//      (기존 per-text Vulkan 경로는 같은 텍스트 2회가 319/320 만 같았습니다 — 가중치 VRAM 사본이
//       2번째 사용부터 생겨 경로가 바뀌기 때문. 배치 경로는 이 문제가 없습니다)
//      tests/it/app_embed.rs 가 구성 무관성(단건 · 표본 · 역순 절반)을 비트 단위로 확인합니다.
//
//  ── 정확도 ──
//   cuBLAS GEMM 은 M = B·L 과 M = L 에서 다른 커널(또는 split-K)을 고를 수 있어
//   BF16 1ulp 수준 차이가 날 수 있습니다. 패딩 버킷은 softmax / attn·V 의 리덕션
//   길이가 달라집니다. 그래서 per-text 와는 비트 단위로 같지 않으며,
//   이 경로는 별도 캐시 공간(FORWARD_TAG_BATCHED)을 쓰고,
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
    /// true: 모든 마이크로배치를 B(L) 행 고정 모양으로 돌립니다(모자란 행은 더미, GPU 기본).
    /// false: 실제 구성원 행만 돌립니다(CPU · Vulkan, ⑥).
    pub fixed_shape: bool,
}

impl Default for BatchOpts {
    /// GPU(CUDA/ROCm) 기본값: 고정 모양. 장치별 값은 [`BatchOpts::for_device`].
    fn default() -> Self {
        let token_budget = std::env::var("LOGIS_EMBED_TOKEN_BUDGET")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|v| *v >= 64)
            .unwrap_or(1024);
        Self { token_budget, attn_budget: 1 << 19, max_rows: 256, exact_len_max: 64, pad_step: 32, fixed_shape: true }
    }
}

impl BatchOpts {
    /// 장치별 정책: CUDA/ROCm 은 고정 모양, CPU · Vulkan · Metal 은 동적 행 수(⑥).
    /// Metal 은 macos-15 러너 실측으로 동적이 비트 동일 · 구성 무관 · 2.7배, 고정 모양은 0.4배였습니다(embedding.rs 주석).
    /// `LOGIS_EMBED_FIXED_SHAPE=1|0` 이면 장치와 무관하게 강제합니다(진단용 · 캐시 공간 태그도 따라 바뀝니다).
    pub fn for_device(device: &Device) -> Self {
        let mut o = Self::default();
        o.fixed_shape = !(device.is_cpu() || device.is_vulkan() || device.is_metal());
        if let Ok(v) = std::env::var("LOGIS_EMBED_FIXED_SHAPE") {
            o.fixed_shape = v.trim() != "0";
        }
        if !o.fixed_shape {
            // 동적 모드는 패딩 버킷을 쓰지 않습니다: 긴 텍스트는 연산량이 시간을 정해 묶어도 이득이 없고
            // (실측 CPU: 65~512토큰 17건이 패딩·마스크 때문에 오히려 느림), 정확한 길이끼리만 묶으면
            // 모든 행이 per-text 와 같은 리덕션 길이가 됩니다.
            o.exact_len_max = EMBED_CHUNK_TOKENS;
        }
        o
    }

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

/// Vulkan 배치 순전파의 고정 실행 모드. 기본 Gpu(가중치 행렬곱은 항상 GPU GEMM, 나머지는 CPU 경로),
/// `LOGIS_EMBED_VK_MODE=host` 이면 전부 CPU 경로(CPU 백엔드와 같은 비트).
#[cfg(feature = "vulkan")]
fn vulkan_exec_mode() -> candle_core::vulkan_backend::shaders::ExecMode {
    use candle_core::vulkan_backend::shaders::ExecMode;
    static MODE: std::sync::OnceLock<ExecMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let host = std::env::var("LOGIS_EMBED_VK_MODE")
            .map(|v| v.trim().eq_ignore_ascii_case("host"))
            .unwrap_or(false);
        let m = if host { ExecMode::Host } else { ExecMode::Gpu };
        println!("[EMBED-BATCHED] Vulkan 배치 순전파 실행 모드: {:?} (LOGIS_EMBED_VK_MODE=host|gpu)", m);
        m
    })
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
        // ⑤ 단건 호출도 지름길 없이 아래의 고정 모양 마이크로배치를 그대로 거칩니다.

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
        // Vulkan: 연산마다 GPU/CPU 경로를 크기·가중치 사용 횟수로 고르면 같은 행이 배치 구성에 따라
        // 다른 커널을 타서 비트가 달라집니다. 이 구간은 포크의 고정 실행 모드로 경로를 묶습니다(⑥).
        #[cfg(feature = "vulkan")]
        let _vk_mode = self
            .device
            .is_vulkan()
            .then(|| candle_core::vulkan_backend::shaders::scoped_exec_mode(vulkan_exec_mode()));
        let mut outs: Vec<Tensor> = Vec::new();
        let mut order: Vec<usize> = Vec::with_capacity(seqs.len());
        for (&blen, members) in buckets.iter() {
            let rows = opts.rows_for(blen);
            let exact = blen <= opts.exact_len_max;
            for group in members.chunks(rows) {
                // 실패(OOM 등)는 더 작은 모양으로 다시 돌리지 않고 그대로 올립니다.
                // 작은 모양은 다른 수치를 내기 때문입니다. 호출부(get_embedding_batch)가
                // 그 묶음만 per-text 로 계산하고 캐시에는 넣지 않습니다.
                // ⑥ 동적 모드는 실제 구성원 행만 돌립니다(더미 없음).
                let shape_rows = if opts.fixed_shape { rows } else { group.len() };
                let pooled = self.forward_group(&seqs, group, blen, shape_rows, exact, &mut stats)?;
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

    /// 마이크로배치 하나를 고정 모양 [rows, blen] 으로 순전파하고, 실제 구성원 g 행의
    /// 정규화된 [g, H] 텐서를 돌려줍니다(GPU 상주). rows · blen · exact 는 버킷 길이만의 함수이고,
    /// g < rows 이면 첫 구성원의 복사본으로 나머지 행을 채운 뒤 그 출력은 버립니다.
    fn forward_group(
        &self,
        seqs: &[Seq],
        group: &[usize],
        blen: usize,
        rows: usize,
        exact: bool,
        stats: &mut BatchRunStats,
    ) -> Result<Tensor> {
        let g = group.len();
        debug_assert!(g >= 1 && g <= rows);
        let l = blen;
        stats.micro_batches += 1;
        stats.computed_tokens += rows * l;

        // 고정 모양의 행 목록: 실제 구성원 g 개 + 더미(첫 구성원 복사) rows-g 개.
        let row_seq = |r: usize| -> usize { if r < g { group[r] } else { group[0] } };
        let mut flat: Vec<u32> = Vec::with_capacity(rows * l);
        for r in 0..rows {
            let ids = &seqs[row_seq(r)].ids;
            debug_assert!(ids.len() <= l && (!exact || ids.len() == l));
            flat.extend_from_slice(ids);
            flat.extend(std::iter::repeat(self.pad_id).take(l - ids.len()));
        }
        let input = Tensor::from_vec(flat, rows * l, &self.device)?;

        if exact {
            // ① 정확한 길이 버킷(L ≤ exact_len_max): 마스크 없음, 배치 풀링.
            let hidden = self.model.forward_batch(&input, rows, l, None)?;
            let sum = hidden.sum(1)?; // [rows, H]
            let mean = (sum / (l as f64))?;
            let norm = mean.sqr()?.sum_keepdim(1)?.sqrt()?; // [rows, 1]
            let pooled = mean.broadcast_div(&norm)?;
            return Ok(if g == rows { pooled } else { pooled.narrow(0, 0, g)? });
        }

        // ② 패딩 버킷: 가산 마스크 [rows,1,1,L] (유효 0 / 패딩 -inf). 구성원 길이와 무관하게 항상 이 경로.
        stats.padded_batches += 1;
        let mut mask: Vec<f32> = Vec::with_capacity(rows * l);
        for r in 0..rows {
            let len = seqs[row_seq(r)].ids.len();
            mask.extend(std::iter::repeat(0.0f32).take(len));
            mask.extend(std::iter::repeat(f32::NEG_INFINITY).take(l - len));
        }
        let mask = Tensor::from_vec(mask, (rows, 1, 1, l), &self.device)?.to_dtype(self.dtype)?;
        let hidden = self.model.forward_batch(&input, rows, l, Some(&mask))?;
        // 풀링은 실제 구성원 행마다 실제 길이만큼만 (per-text 와 같은 리덕션 길이).
        let mut out = Vec::with_capacity(g);
        for (r, &si) in group.iter().enumerate() {
            let len = seqs[si].ids.len();
            let h = hidden.narrow(0, r, 1)?.narrow(1, 0, len)?.contiguous()?;
            out.push(Self::pool_single(&h)?);
        }
        Ok(Tensor::cat(&out, 0)?)
    }
}

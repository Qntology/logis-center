//! 🌟 [VISION-BATCH] SigLIP2 비전 파이프라인의 배치 경로 (CPU · 항상 / GPU · ignore)
//!
//! - 패치 × 구 채점(`score_patches_bank_neutral`)이 배치 내적 행렬로 바뀐 뒤에도 예전 스칼라 구현과 비트 동일한지.
//! - 값 접지(`verify_claims`)가 값을 한 번에 인코딩하고 코사인을 행렬로 계산해도 값별 계산과 같은지.
//! - 구 인코딩(`encode_phrase_rows`)이 묶음 구성(1건 · 40건 · 고정 모양 32행)과 무관한 비트를 내는지 —
//!   작은 무작위 SigLIP2 텍스트 인코더 + WordLevel 토크나이저(목업).

use candle_core::{DType, Device, Tensor};
use std::sync::Arc;
use tauri_app_lib::models::siglip2::legibility::{LegibilityMap, PatchLegibility};
use tauri_app_lib::models::siglip2::value_grounding::{verify_claims, GroundingClaim};
use tauri_app_lib::models::siglip2::vision_encoder::{
    encode_phrase_rows, score_patches_bank_neutral, AnchorBank, PatchGrid, PHRASE_BATCH_ROWS,
};
use tauri_app_lib::models::siglip2::{Siglip2Config, Siglip2Model};

use crate::common::TempDir;

/// 시드 고정 의사난수 (xorshift) — 플랫폼 무관.
fn det(n: usize, salt: u64) -> Vec<f32> {
    let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ salt.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            ((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

fn unit(mut v: Vec<f32>) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    for x in v.iter_mut() {
        *x /= n;
    }
    v
}

const D: usize = 1152;

fn mock_grid(rows: usize, cols: usize) -> PatchGrid {
    let n = rows * cols;
    let mut patches: Vec<Vec<f32>> = (0..n).map(|i| unit(det(D, 1000 + i as u64))).collect();
    for i in [0usize, 7, 40, n - 1] {
        patches[i] = vec![0.0; D]; // 빈 패치(전부 0)는 채점에서 빠집니다
    }
    PatchGrid {
        patches,
        pooled: unit(det(D, 7)),
        grid_rows: rows,
        grid_cols: cols,
        scale_x: 1.0,
        scale_y: 1.0,
        orig_width: (cols * 16) as u32,
        orig_height: (rows * 16) as u32,
        patch_size: 16,
    }
}

/// 서식 분류 뱅크 크기 (카테고리 4 · 키 48 · bias 구 3~6개 · 편견 그룹은 키 이름 일부와 카테고리 이름).
/// 같은 구(같은 Arc)를 여러 키 · 편견 쪽에서 함께 쓰는 경우를 넣습니다.
fn mock_bank() -> AnchorBank {
    let mut pool: Vec<Arc<Vec<f32>>> = (0..260).map(|j| Arc::new(unit(det(D, 50_000 + j)))).collect();
    pool.push(Arc::new(vec![0.0; D])); // 영 벡터 구 (인코딩 실패 자리)
    let cats = ["shipping", "customs", "finance", "parcel"];
    let mut bias = Vec::new();
    let mut prejudice = Vec::new();
    let mut next = 0usize;
    for k in 0..48usize {
        let cat = cats[k % cats.len()];
        let key = format!("key{k}");
        for _ in 0..(3 + k % 4) {
            bias.push((cat.to_string(), key.clone(), pool[next % pool.len()].clone()));
            next += 1;
        }
        if k % 5 == 0 {
            bias.push((cat.to_string(), key.clone(), pool[(k * 3) % 40].clone())); // 다른 키와 공유
        }
        if k % 3 == 0 {
            prejudice.push((cat.to_string(), key.clone(), pool[(next + 11) % pool.len()].clone()));
            prejudice.push((cat.to_string(), key.clone(), pool[k % 20].clone())); // bias 쪽과 같은 Arc
        }
    }
    for (ci, c) in cats.iter().enumerate() {
        for j in 0..4usize {
            prejudice.push((c.to_string(), c.to_string(), pool[200 + ci * 4 + j].clone()));
        }
    }
    prejudice.push(("parcel".into(), "key9".into(), pool[260].clone()));
    AnchorBank { bias, prejudice }
}

// 예전(스칼라 내적) 구현 — 이 변경 직전의 score_patches_bank_neutral 그대로(로그만 뺌).
fn reference_bank_neutral(
    grid: &PatchGrid,
    bank: &AnchorBank,
    legible: Option<&LegibilityMap>,
) -> (Vec<String>, Vec<Vec<f32>>) {
    use std::collections::HashMap;

    let n = grid.len();
    if n == 0 || bank.bias.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // ── ① (category, key) 그룹 색인 ──
    let mut order: Vec<String> = Vec::new();
    let mut bias_idx: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, (_, k, _)) in bank.bias.iter().enumerate() {
        if !bias_idx.contains_key(k) {
            order.push(k.clone());
        }
        bias_idx.entry(k.clone()).or_default().push(i);
    }
    let mut prej_idx: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, (_, k, _)) in bank.prejudice.iter().enumerate() {
        prej_idx.entry(k.clone()).or_default().push(i);
    }

    let mut key_cat: HashMap<String, String> = HashMap::new();
    for (c, k, _) in bank.bias.iter() {
        key_cat.entry(k.clone()).or_insert_with(|| c.clone());
    }

    let dot = |a: &[f32], b: &[f32]| -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    };

    let mut prej_pool: HashMap<String, Vec<f32>> = HashMap::new();
    for (gname, list) in prej_idx.iter() {
        let mut v = vec![f32::MIN; n];
        for i in 0..n {
            let p = &grid.patches[i];
            if p.iter().all(|&x| x == 0.0) { continue; }
            let mut mp = f32::MIN;
            for &j in list {
                let s = dot(p, &bank.prejudice[j].2);
                if s > mp { mp = s; }
            }
            v[i] = mp;
        }
        prej_pool.insert(gname.clone(), v);
    }

    // ── ② 원시 Max-Pool 행렬 ──
    let m = order.len();
    let mut raw_b = vec![vec![f32::MIN; n]; m];
    let mut raw_p = vec![vec![f32::MIN; n]; m];
    let mut resolved = 0usize;

    for (ki, key) in order.iter().enumerate() {
        let bi = &bias_idx[key];
        // 필드명 → 실패 시 카테고리명 순으로 편견 그룹을 찾습니다.
        let pv: Option<&Vec<f32>> = prej_pool
            .get(key)
            .or_else(|| key_cat.get(key).and_then(|c| prej_pool.get(c)));
        if pv.is_some() { resolved += 1; }
        for i in 0..n {
            let p = &grid.patches[i];
            if p.iter().all(|&v| v == 0.0) { continue; }
            let mut mb = f32::MIN;
            for &j in bi {
                let s = dot(p, &bank.bias[j].2);
                if s > mb { mb = s; }
            }
            raw_b[ki][i] = mb;
            if let Some(pv) = pv {
                raw_p[ki][i] = pv[i];
            }
        }
    }

    // ── ③ 판독 가능 패치만으로 기준선 산출 ──
    let usable: Vec<usize> = (0..n)
        .filter(|&i| legible.map_or(true, |l| l.is_legible(i)))
        .filter(|&i| !grid.patches[i].iter().all(|&v| v == 0.0))
        .collect();
    let base: &[usize] = if usable.len() >= 8 {
        &usable
    } else {
        // 판독 가능 패치가 너무 적으면 전 패치로 폴백합니다.
        // (스캔 품질이 나쁠 때 통계가 통째로 무너지는 것을 막습니다)
        &[]
    };
    let idx_all: Vec<usize> = (0..n).collect();
    let base: &[usize] = if base.is_empty() { &idx_all } else { base };

    // 전역 pooled σ : 뱅크마다 σ 를 쓰면 1구 뱅크의 z 가 폭발합니다.
    let mut pooled_var = 0.0f32;
    let mut pooled_cnt = 0usize;
    let mut mu_b = vec![0.0f32; m];
    let mut mu_p = vec![0.0f32; m];
    for ki in 0..m {
        let mut sb = 0.0f32; let mut sp = 0.0f32; let mut c = 0usize;
        for &i in base {
            if raw_b[ki][i] == f32::MIN { continue; }
            sb += raw_b[ki][i];
            if raw_p[ki][i] != f32::MIN { sp += raw_p[ki][i]; }
            c += 1;
        }
        if c == 0 { continue; }
        mu_b[ki] = sb / c as f32;
        mu_p[ki] = sp / c as f32;
        for &i in base {
            if raw_b[ki][i] == f32::MIN { continue; }
            let d = raw_b[ki][i] - mu_b[ki];
            pooled_var += d * d;
            pooled_cnt += 1;
        }
    }
    let sd = if pooled_cnt > 1 {
        (pooled_var / pooled_cnt as f32).sqrt().max(1e-6)
    } else {
        1.0
    };

    // ── ④ 행 센터링 + 편견 상쇄 ──
    let mut net = vec![vec![f32::MIN; n]; m];
    for ki in 0..m {
        for i in 0..n {
            if raw_b[ki][i] == f32::MIN { continue; }
            let zb = (raw_b[ki][i] - mu_b[ki]) / sd;
            let zp = if raw_p[ki][i] == f32::MIN {
                0.0
            } else {
                ((raw_p[ki][i] - mu_p[ki]) / sd).max(0.0)
            };
            net[ki][i] = zb - zp;
        }
    }

    // ── ⑤ 열 센터링 : '전 개념에 반응하는 잉크 패치' 공통 성분 제거 ──
    for i in 0..n {
        let mut s = 0.0f32; let mut c = 0usize;
        for ki in 0..m {
            if net[ki][i] == f32::MIN { continue; }
            s += net[ki][i]; c += 1;
        }
        if c < 2 { continue; }
        let mean = s / c as f32;
        for ki in 0..m {
            if net[ki][i] == f32::MIN { continue; }
            net[ki][i] -= mean;
        }
    }

    let _ = resolved;
    (order, net)
}


#[test]
fn vision_score_matrix_matches_scalar_reference() {
    let grid = mock_grid(18, 14);
    let bank = mock_bank();
    let t0 = std::time::Instant::now();
    let (keys_ref, net_ref) = reference_bank_neutral(&grid, &bank, None);
    let ref_s = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let (keys, net) = score_patches_bank_neutral(&grid, &bank, None);
    let new_s = t1.elapsed().as_secs_f64();
    assert_eq!(keys, keys_ref);
    assert_eq!(net.len(), net_ref.len());
    let mut cells = 0usize;
    for (a, b) in net.iter().zip(net_ref.iter()) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
            cells += 1;
        }
    }
    let phrases = bank.bias.len() + bank.prejudice.len();
    println!(
        "[VISION-SCORE] 패치 {} × 구 {} (키 {}) | 결과 {}칸 비트 동일 | 스칼라 {:.1} ms → 배치 행렬 {:.1} ms ({:.1}배, debug 테스트 빌드)",
        grid.len(), phrases, keys.len(), cells, ref_s * 1e3, new_s * 1e3, ref_s / new_s
    );
}

fn legible_map(rows: usize, cols: usize) -> LegibilityMap {
    let n = rows * cols;
    let mut verdict = vec![PatchLegibility::Legible; n];
    verdict[5] = PatchLegibility::Blank;
    LegibilityMap {
        rows,
        cols,
        ink: vec![0.5; n],
        sharpness: vec![0.5; n],
        verdict,
        split_applied: false,
        ink_gate: 0.1,
        sharp_gate: 0.1,
    }
}

fn value_emb(v: &str) -> Vec<f32> {
    if v == "ZERO" {
        return vec![0.0; D];
    }
    let salt = v.bytes().fold(1469598103934665603u64, |h, b| (h ^ b as u64).wrapping_mul(1099511628211));
    unit(det(D, salt))
}

/// 값 접지: 예전처럼 값마다 인코딩 · cosine_similarity 로 계산한 접지 점수와 비트 동일한지,
/// 인코더 호출이 1회(같은 값은 한 번)인지.
#[test]
fn value_grounding_batch_matches_per_claim_reference() {
    use tauri_app_lib::utils::ai_utils::{cosine_similarity, gumbel_expected_z};
    let grid = mock_grid(12, 10);
    let leg = legible_map(12, 10);
    let (w, h) = (grid.orig_width, grid.orig_height);
    let vals = ["ACME CORP", "12,500 KG", "ACME CORP", "", "ZERO", "BUSAN", "INV-2026-0042", "  BUSAN  "];
    let claims: Vec<GroundingClaim> = vals
        .iter()
        .enumerate()
        .map(|(i, v)| GroundingClaim {
            category: "shipper".into(),
            field: format!("f{i}"),
            value: v.to_string(),
            bbox: ((i as u32 * 13) % w, (i as u32 * 29) % h, ((i as u32 * 13) % w) + 60, ((i as u32 * 29) % h) + 50),
        })
        .collect();
    let calls = std::cell::Cell::new(0usize);
    let seen = std::cell::RefCell::new(Vec::<String>::new());
    let got = verify_claims(
        &claims,
        &grid.patches,
        grid.grid_rows,
        grid.grid_cols,
        w,
        h,
        &leg,
        |batch: &[String]| {
            calls.set(calls.get() + 1);
            seen.borrow_mut().extend(batch.iter().cloned());
            batch.iter().map(|v| value_emb(v)).collect()
        },
        &|_: &str| {},
    );
    assert_eq!(calls.get(), 1, "값 인코딩은 한 번에");
    assert_eq!(seen.borrow().len(), 5, "같은 값은 한 번만: {:?}", seen.borrow());

    // 예전 계산(값마다 전 패치 cosine_similarity)
    let n = grid.len();
    let mut checked = 0usize;
    for vd in got.iter() {
        let emb = value_emb(&vd.value);
        if emb.iter().all(|&x| x == 0.0) {
            continue; // 보류(Held) 경로
        }
        let claim = claims.iter().find(|c| c.field == vd.field).unwrap();
        let sims: Vec<f32> = grid.patches.iter().map(|p| cosine_similarity(&emb, p)).collect();
        let mean: f32 = sims.iter().sum::<f32>() / n as f32;
        let var: f32 = sims.iter().map(|s| (s - mean) * (s - mean)).sum::<f32>() / n as f32;
        let std = var.sqrt().max(1e-6);
        let (cw, ch) = (w as f32 / grid.grid_cols as f32, h as f32 / grid.grid_rows as f32);
        let inside: Vec<usize> = (0..n)
            .filter(|&i| {
                let (r, c) = (i / grid.grid_cols, i % grid.grid_cols);
                let (cx, cy) = ((c as f32 + 0.5) * cw, (r as f32 + 0.5) * ch);
                cx >= claim.bbox.0 as f32 && cx <= claim.bbox.2 as f32 && cy >= claim.bbox.1 as f32 && cy <= claim.bbox.3 as f32
            })
            .collect();
        if inside.is_empty() {
            continue;
        }
        let max_in = inside.iter().map(|&i| sims[i]).fold(f32::MIN, f32::max);
        let outs: Vec<usize> = (0..n).filter(|i| !inside.contains(i)).collect();
        let max_out = outs.iter().map(|&i| sims[i]).fold(f32::MIN, f32::max);
        let s_in = (max_in - mean) / std - gumbel_expected_z(inside.len());
        let s_out = (max_out - mean) / std - gumbel_expected_z(outs.len());
        assert_eq!(vd.surprisal_in.to_bits(), s_in.to_bits(), "{}", vd.value);
        assert_eq!(vd.surprisal_out.to_bits(), s_out.to_bits(), "{}", vd.value);
        checked += 1;
    }
    assert!(checked >= 3, "비교한 값 {checked}");
    println!("[VISION-BATCH] 값 접지: 값 {}건 → 인코더 호출 1회(고유 값 {}), 접지 점수 {}건 값별 계산과 비트 동일", claims.len(), seen.borrow().len(), checked);
}

// ── 구 인코딩 묶음 불변성 (목업 SigLIP2 텍스트 인코더) ──

const VOCAB: usize = 96;

fn mock_config() -> Siglip2Config {
    Siglip2Config {
        vision_hidden_size: 64,
        vision_intermediate_size: 128,
        vision_num_layers: 1,
        vision_num_heads: 4,
        patch_size: 16,
        max_num_patches: 256,
        vision_layer_norm_eps: 1e-6,
        // 실모델과 같은 은닉 크기: CPU gemm 은 K 가 작으면(예: 64) M < 8 에서 다른 경로를 타므로
        // 목업도 실모델 경로(K = 1152 · M ≥ 2 불변, M = 1 만 gemv)를 재야 합니다.
        text_hidden_size: 1152,
        text_intermediate_size: 1536,
        text_num_layers: 1,
        text_num_heads: 16,
        text_vocab_size: VOCAB,
        text_max_positions: 64,
        text_pad_token_id: 1,
        text_layer_norm_eps: 1e-6,
    }
}

/// WordLevel 토크나이저: "[UNK]"=0, "<pad>"=1, "w0".."w93"=2..
fn write_tokenizer(dir: &std::path::Path) {
    let mut vocab = serde_json::Map::new();
    vocab.insert("[UNK]".into(), 0.into());
    vocab.insert("<pad>".into(), 1.into());
    for i in 0..VOCAB - 2 {
        vocab.insert(format!("w{i}"), (i + 2).into());
    }
    let tok = serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    });
    std::fs::write(dir.join("tokenizer.json"), serde_json::to_vec(&tok).unwrap()).unwrap();
}

fn mock_siglip_text(dev: &Device, dir: &std::path::Path) -> anyhow::Result<Siglip2Model> {
    use candle_nn::{VarBuilder, VarMap};
    use tauri_app_lib::models::siglip2::{text::Siglip2TextModel, tokenizer::Siglip2Tokenizer};
    let cfg = mock_config();
    // 가중치는 CPU 에서 한 번 만들고(VarMap 무작위 초기화) 장치로 옮깁니다.
    let varmap = VarMap::new();
    let vb_cpu = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let _ = Siglip2TextModel::new(&cfg, vb_cpu, None)?;
    let tensors: std::collections::HashMap<String, Tensor> = varmap
        .data()
        .lock()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.clone(), v.as_tensor().to_device(dev).unwrap()))
        .collect();
    let vb = VarBuilder::from_tensors(tensors, DType::F32, dev);
    let text = Siglip2TextModel::new(&cfg, vb, None)?;
    let tok = Siglip2Tokenizer::from_dir(dir, cfg.text_pad_token_id, cfg.text_max_positions)?;
    Ok(Siglip2Model {
        vision: None,
        text: Some(text),
        tokenizer: Some(tok),
        logit_scale: 0.0,
        logit_bias: 0.0,
        device: dev.clone(),
        dtype: DType::F32,
        config: cfg,
    })
}

fn mock_phrases(n: usize) -> Vec<String> {
    (0..n)
        .map(|i| {
            let len = 1 + (i * 7) % 9;
            (0..len).map(|j| format!("w{}", (i * 31 + j * 17) % (VOCAB - 2))).collect::<Vec<_>>().join(" ")
        })
        .collect()
}

fn phrase_batch_invariance(dev: &Device) -> anyhow::Result<()> {
    let dir = TempDir::new("siglip_mock_tok");
    write_tokenizer(&dir.0);
    let model = mock_siglip_text(dev, &dir.0)?;
    let phrases = mock_phrases(PHRASE_BATCH_ROWS + 8); // 32 + 8: 묶음 2개, 마지막은 8행
    for fixed in [false, true] {
        let t0 = std::time::Instant::now();
        let batch = encode_phrase_rows(&model, &phrases, fixed)?;
        let batch_s = t0.elapsed().as_secs_f64();
        // 1건씩은 첫 묶음 · 경계 · 마지막 묶음(8행)에서 표본만 (1건 호출 하나가 고정 모양이면 32행 순전파).
        let probe = [0usize, 1, 17, PHRASE_BATCH_ROWS - 1, PHRASE_BATCH_ROWS, phrases.len() - 1];
        let t1 = std::time::Instant::now();
        let mut differ = 0usize;
        for &i in probe.iter() {
            let one = encode_phrase_rows(&model, std::slice::from_ref(&phrases[i]), fixed)?.remove(0);
            differ += one.iter().zip(batch[i].iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        }
        let single_s = t1.elapsed().as_secs_f64() / probe.len() as f64;
        println!(
            "[VISION-BATCH] 구 인코딩 {:?} fixed_shape={} | 구 {}개 묶음 vs 1건씩(표본 {}): 다른 비트 {} | 구당 1건씩 {:.1} ms → 묶음 {:.1} ms",
            dev, fixed, phrases.len(), probe.len(), differ, single_s * 1e3, batch_s * 1e3 / phrases.len() as f64
        );
        assert_eq!(differ, 0, "{dev:?} fixed_shape={fixed}: 묶음 구성에 따라 구 벡터가 달라짐");
    }
    Ok(())
}

#[test]
fn siglip2_phrase_rows_are_batch_invariant_cpu() -> anyhow::Result<()> {
    phrase_batch_invariance(&Device::Cpu)
}

/// GPU 장치(Metal · Vulkan · CUDA · ROCm)에서 같은 단언. Vulkan 실행 모드 등 장치 전역 상태를 바꾸므로
/// `--ignored --test-threads 1` 로 단독 실행합니다.
#[test]
#[ignore]
fn siglip2_phrase_rows_are_batch_invariant_gpu() -> anyhow::Result<()> {
    // gpu_count() 는 CUDA/ROCm 장치 열거라 Metal 에서는 0 입니다 — gpu_memory.rs 와 같이 장치를 직접 만듭니다.
    let dev = tauri_app_lib::utils::new_gpu_device(0).filter(|d| !d.is_cpu());
    let Some(dev) = dev else {
        println!("[VISION-BATCH] GPU 장치를 얻지 못함 — NOT RUN");
        return Ok(());
    };
    phrase_batch_invariance(&dev)
}



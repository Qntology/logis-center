//! P2b 실모델 정확도·속도 대조 (실제 GGUF, CUDA / ROCm). `LOGIS_TEST_MODELS=1` + `-- --ignored` 로만 실행합니다.
//!
//! - `p2b_qwen35_2b_fused_gdn`  : Qwen3.5-2B Q8_0 — 융합 GDN vs 기존 텐서 경로, Q8_0 forward_exact vs 기존 선형층.
//!   teacher-forcing (프리필 270 + 디코드 30 스텝): max |Δlogit|, top-1 일치율, 호출당 시간.
//! - `p2b_qwen35_4b_timing`     : Qwen3.5-4B-kor (alphaedge q4km) — 프리필 ~900 + 디코드 40 스텝 호출 시간 (융합/기존).
//! - `p2b_qwen3_q8_resident`    : Qwen3-0.6B Q8_0 — 기존 BF16 역양자화 vs Q8 상주(정확 W8A16) vs F32 기준
//!   (Q8 가중치 정확 복원 + F32 활성·matmul·logits). KLD, top-1/top-5, 상주 VRAM, 디코드 tok/s.
//! - `p2b_qwen3_kv_tier_forced` : Qwen3-0.6B Q8 상주 — 디코드 진입 시 KV 를 Vram / Ram / Ssd 로 강제, 원본 KV · Vram 대비 로짓.
//!
//! 모델 경로: `%LOCALAPPDATA%/logis-center/models/...` (없으면 건너뜀). GPU 에는 한 번에 한 모델만 올립니다.
//! 장치: CUDA 빌드는 `Device::new_cuda(0)`, ROCm 빌드(`--features rocm`)는 `Device::new_rocm(0)` (둘 다 켜면 CUDA 먼저).
#![cfg(any(feature = "cuda", feature = "rocm"))]

use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};

type R = anyhow::Result<()>;

fn enabled() -> bool {
    std::env::var("LOGIS_TEST_MODELS").map(|v| v == "1").unwrap_or(false)
}

/// `LOGIS_MODELS_DIR` 또는 OS 별 앱 데이터 경로 (tests/it/common.rs).
fn models_dir() -> PathBuf {
    crate::common::models_dir()
}

/// GPU 0번 장치. CUDA 빌드는 기존과 같이 `Device::new_cuda(0)`, ROCm 빌드는 `Device::new_rocm(0)` 을 씁니다.
/// 열 수 없으면 None → 테스트를 건너뜁니다.
fn gpu() -> Option<Device> {
    #[cfg(feature = "cuda")]
    match Device::new_cuda(0) {
        Ok(d) => return Some(d),
        Err(e) => println!("[P2B-TEST] CUDA 장치를 열 수 없어 건너뜁니다: {e}"),
    }
    #[cfg(feature = "rocm")]
    match Device::new_rocm(0) {
        Ok(d) => return Some(d),
        Err(e) => println!("[P2B-TEST] ROCm 장치를 열 수 없어 건너뜁니다: {e}"),
    }
    None
}

/// 한국어 + 영어가 섞인 고정 프롬프트 (주문·상품 추출 작업과 비슷한 문장).
fn prompt_text(repeat: usize) -> String {
    let base = "다음은 물류 센터의 주문 목록입니다. Order #20231 from 서울 강남구 테헤란로 152, 고객명 김민수, \
        상품: 무선 블루투스 이어폰 (Wireless Earbuds Pro, SKU WE-PRO-2) 수량 3개, 단가 89,000원, 총액 267,000원. \
        Shipping method: express delivery via CJ대한통운, tracking number 6301-2277-8812. \
        두 번째 주문 #20232 은 부산 해운대구 센텀중앙로 79 의 이서연 고객이 주문한 스테인리스 텀블러 500ml (Stainless Tumbler, \
        color: midnight blue) 2개이며 할인 쿠폰 SUMMER10 이 적용되어 결제 금액은 35,820원입니다. \
        The warehouse manager noted that item WE-PRO-2 is low on stock (12 units left) and requested a reorder of 200 units \
        from supplier 한빛전자(주) by next Friday. 반품 요청: 주문 #20198 의 노트북 거치대(알루미늄)는 배송 중 파손으로 \
        교환 처리 예정이며 회수 기사 방문일은 10월 15일 오후 2시입니다. Please extract every order id, customer name, \
        product name, quantity, unit price and total amount as JSON. ";
    base.repeat(repeat)
}

fn tokens_or_skip(ids: Vec<u32>, n: usize, what: &str) -> Option<Vec<u32>> {
    if ids.len() < n {
        println!("[P2B-TEST] {what}: 토큰 {} < {n}", ids.len());
        return None;
    }
    Some(ids[..n].to_vec())
}

fn row_f32(t: &Tensor) -> anyhow::Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?)
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

fn top_k(v: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.select_nth_unstable_by(k, |&a, &b| v[b].partial_cmp(&v[a]).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    idx
}

fn log_softmax(v: &[f32]) -> Vec<f64> {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = v.iter().map(|&x| ((x as f64) - m).exp()).sum::<f64>().ln() + m;
    v.iter().map(|&x| x as f64 - lse).collect()
}

/// KL(ref || x)
fn kld(reference: &[f32], x: &[f32]) -> f64 {
    let lr = log_softmax(reference);
    let lx = log_softmax(x);
    lr.iter().zip(lx.iter()).map(|(&a, &b)| a.exp() * (a - b)).sum()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

struct Cmp {
    max_abs: f32,
    top1: f64,
    top5: f64,
    kld: f64,
}

fn compare(reference: &[Vec<f32>], x: &[Vec<f32>]) -> Cmp {
    let n = reference.len().min(x.len()).max(1);
    let (mut ma, mut t1, mut t5, mut kl) = (0f32, 0f64, 0f64, 0f64);
    for (r, y) in reference.iter().zip(x.iter()) {
        ma = ma.max(max_abs(r, y));
        t1 += (argmax(r) == argmax(y)) as u32 as f64;
        let (tr, ty) = (top_k(r, 5), top_k(y, 5));
        t5 += tr.iter().filter(|i| ty.contains(i)).count() as f64 / 5.0;
        kl += kld(r, y);
    }
    let n = n as f64;
    Cmp { max_abs: ma, top1: t1 / n * 100.0, top5: t5 / n * 100.0, kld: kl / n }
}

// ─────────────────────────────── Qwen3.5 ───────────────────────────────

struct Q35Run {
    rows: Vec<Vec<f32>>,
    prefill_s: f64,
    decode_ms: f64,
}

/// teacher-forcing: 앞쪽 `ids.len() - decode` 토큰을 한 번에 프리필, 나머지를 한 토큰씩 디코드.
fn q35_run(
    rt: &tokio::runtime::Runtime,
    g: &mut tauri_app_lib::models::qwen3_5::generate::Qwen3_5GenerateModel,
    dev: &Device,
    ids: &[u32],
    decode: usize,
) -> anyhow::Result<Q35Run> {
    // 앱의 중지 신호 파일(사용자 상태)은 건드리지 않고, 이 테스트 프로세스에서만 프리필 중지 검사를 끕니다.
    tauri_app_lib::models::qwen3_5::model::IGNORE_STOP_SIGNAL_FOR_TESTS
        .store(true, std::sync::atomic::Ordering::Relaxed);
    g.clear_kv_cache();
    let n = ids.len();
    let full = Tensor::from_slice(ids, (1, n), dev)?;
    let _ = g.qwen3_5.compute_and_set_rope_deltas(&full, None, None);
    let p = n - decode;
    let mut rows = Vec::with_capacity(decode + 1);
    let t0 = Instant::now();
    let pre = Tensor::from_slice(&ids[..p], (1, p), dev)?;
    let logits = rt.block_on(g.qwen3_5.forward(&pre, None, None, None, None, None, 0, None, None))?;
    rows.push(row_f32(&logits)?);
    let prefill_s = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    for j in p..n - 1 {
        let one = Tensor::from_slice(&ids[j..j + 1], (1, 1), dev)?;
        let logits = rt.block_on(g.qwen3_5.forward(&one, None, None, None, None, None, j, None, None))?;
        rows.push(row_f32(&logits)?);
    }
    let decode_ms = t1.elapsed().as_secs_f64() * 1000.0 / (n - 1 - p).max(1) as f64;
    g.clear_kv_cache();
    Ok(Q35Run { rows, prefill_s, decode_ms })
}

fn set_paths(fused: bool, exact: bool) {
    tauri_app_lib::models::common::set_gdn_fused_override(Some(fused));
    tauri_app_lib::models::common::gguf::set_q8_exact_override(Some(exact));
}

#[test]
#[ignore = "실모델 (LOGIS_TEST_MODELS=1)"]
fn p2b_qwen35_2b_fused_gdn() -> R {
    if !enabled() {
        return Ok(());
    }
    let Some(dev) = gpu() else { return Ok(()) };
    let file = models_dir().join("Qwen3.5-2B-Instruct-gguf").join("Qwen3.5-2B-Q8_0.gguf");
    if !file.exists() {
        println!("[P2B-TEST] {file:?} 없음 → 건너뜀");
        return Ok(());
    }
    let rt = tokio::runtime::Runtime::new()?;
    let mut g = tauri_app_lib::models::qwen3_5::generate::Qwen3_5GenerateModel::init_from_gguf(
        file.to_str().unwrap(),
        None,
        Some(&dev),
    )?;
    let ids = g.tokenizer.text_encode_vec(prompt_text(3), false)?;
    let Some(ids) = tokens_or_skip(ids, 300, "2B") else { return Ok(()) };
    let decode = 30;

    // 1) 기존(텐서 GDN + 기존 선형층, 콜드) 2) 융합 GDN 만 3) 기존(웜, 시간 기준) 4) 융합 GDN + forward_exact (새 기본값)
    set_paths(false, false);
    let old_cold = q35_run(&rt, &mut g, &dev, &ids, decode)?;
    set_paths(true, false);
    let fused = q35_run(&rt, &mut g, &dev, &ids, decode)?;
    set_paths(false, false);
    let old = q35_run(&rt, &mut g, &dev, &ids, decode)?;
    set_paths(true, true);
    let new = q35_run(&rt, &mut g, &dev, &ids, decode)?;
    tauri_app_lib::models::common::set_gdn_fused_override(None);
    tauri_app_lib::models::common::gguf::set_q8_exact_override(None);

    let det = compare(&old_cold.rows, &old.rows);
    let gdn = compare(&old.rows, &fused.rows);
    let lin = compare(&fused.rows, &new.rows);
    let all = compare(&old.rows, &new.rows);
    println!("[P2B-RESULT] Qwen3.5-2B Q8_0 · 토큰 {} (프리필 {} + 디코드 {}) · 비교 행 {}", ids.len(), ids.len() - decode, decode, old.rows.len());
    println!("[P2B-RESULT]  기존 콜드 vs 기존 웜 (결정성): max|Δ| {:.3e} · top-1 {:.1}%", det.max_abs, det.top1);
    println!("[P2B-RESULT]  융합 GDN vs 기존 GDN      : max|Δ| {:.3e} · top-1 {:.1}% · top-5 {:.1}% · KLD {:.3e}", gdn.max_abs, gdn.top1, gdn.top5, gdn.kld);
    println!("[P2B-RESULT]  forward_exact vs 기존 선형 : max|Δ| {:.3e} · top-1 {:.1}% · top-5 {:.1}% · KLD {:.3e}", lin.max_abs, lin.top1, lin.top5, lin.kld);
    println!("[P2B-RESULT]  새 기본값 vs 기존 전체    : max|Δ| {:.3e} · top-1 {:.1}% · top-5 {:.1}% · KLD {:.3e}", all.max_abs, all.top1, all.top5, all.kld);
    for (name, r) in [("기존(콜드)", &old_cold), ("융합 GDN", &fused), ("기존(웜)", &old), ("융합+exact", &new)] {
        println!("[P2B-RESULT]  시간 {name:<10}: 프리필 {:.3} s · 디코드 {:.1} ms/토큰", r.prefill_s, r.decode_ms);
    }
    assert!(gdn.top1 >= 90.0, "융합 GDN top-1 {:.1}%", gdn.top1);
    Ok(())
}

#[test]
#[ignore = "실모델 (LOGIS_TEST_MODELS=1)"]
fn p2b_qwen35_4b_timing() -> R {
    if !enabled() {
        return Ok(());
    }
    let Some(dev) = gpu() else { return Ok(()) };
    let dir = models_dir().join("Qwen3.5-4B-kor-16384");
    if !dir.join("runtime-q4km").join("READY").exists() {
        println!("[P2B-TEST] {dir:?} 런타임 GGUF 없음 → 건너뜀");
        return Ok(());
    }
    let rt = tokio::runtime::Runtime::new()?;
    let mut g = tauri_app_lib::model::lang_gguf::load_runtime(&dir, &dev)?;
    let ids = g.tokenizer.text_encode_vec(prompt_text(8), false)?;
    let Some(ids) = tokens_or_skip(ids, 940, "4B") else { return Ok(()) };
    let decode = 40;
    set_paths(true, true);
    let fused_cold = q35_run(&rt, &mut g, &dev, &ids, decode)?;
    set_paths(false, true);
    let old = q35_run(&rt, &mut g, &dev, &ids, decode)?;
    set_paths(true, true);
    let fused = q35_run(&rt, &mut g, &dev, &ids, decode)?;
    tauri_app_lib::models::common::set_gdn_fused_override(None);
    tauri_app_lib::models::common::gguf::set_q8_exact_override(None);
    let c = compare(&old.rows, &fused.rows);
    println!("[P2B-RESULT] Qwen3.5-4B-kor · 프리필 {} + 디코드 {}", ids.len() - decode, decode);
    for (name, r) in [("융합(콜드)", &fused_cold), ("기존 GDN", &old), ("융합(웜)", &fused)] {
        println!(
            "[P2B-RESULT]  {name:<10}: 프리필 {:.2} s · 디코드 {:.1} ms/토큰 · 호출 합계(프리필+디코드 {decode}) {:.2} s",
            r.prefill_s,
            r.decode_ms,
            r.prefill_s + r.decode_ms * decode as f64 / 1000.0
        );
    }
    println!("[P2B-RESULT]  융합 vs 기존 GDN: max|Δ| {:.3e} · top-1 {:.1}% · KLD {:.3e}", c.max_abs, c.top1, c.kld);
    Ok(())
}

// ─────────────────────────────── Qwen3-0.6B ───────────────────────────────

struct Q3Run {
    prefill_rows: Vec<Vec<f32>>,
    decode_rows: Vec<Vec<f32>>,
    resident_mb: f64,
    decode_tps: f64,
}

fn free_bytes(dev: &Device) -> f64 {
    let _ = dev.synchronize_all();
    dev.effective_free_memory().ok().flatten().unwrap_or(0) as f64
}

/// teacher-forcing: 앞 200 토큰을 40 토큰 청크 5개로 프리필(청크마다 마지막 행), 나머지 100 토큰을 한 토큰씩 디코드.
fn q3_run(dev: &Device, path: &str, q8: bool, dtype: DType, ids: &[u32]) -> anyhow::Result<Q3Run> {
    let _ = dev.trim_memory_pool(0);
    let before = free_bytes(dev);
    let mut g = tauri_app_lib::models::qwen3::generate::Qwen3GenerateModel::init_from_gguf_with(path, Some(dev), Some(dtype), q8)?;
    let after = free_bytes(dev);
    let m = g.model_mut();
    m.clear_kv_cache();
    m.set_kv_exact(true);
    let (chunk, prefill) = (40usize, 200usize);
    let mut prefill_rows = Vec::new();
    let mut off = 0;
    while off < prefill {
        let x = Tensor::from_slice(&ids[off..off + chunk], (1, chunk), dev)?;
        prefill_rows.push(row_f32(&m.forward(Some(&x), None, off)?)?);
        off += chunk;
    }
    let mut decode_rows = Vec::new();
    let t0 = Instant::now();
    for j in prefill..ids.len() - 1 {
        let x = Tensor::from_slice(&ids[j..j + 1], (1, 1), dev)?;
        decode_rows.push(row_f32(&m.forward(Some(&x), None, j)?)?);
    }
    let decode_tps = decode_rows.len() as f64 / t0.elapsed().as_secs_f64();
    m.clear_kv_cache();
    drop(g);
    let _ = dev.synchronize_all();
    let _ = dev.trim_memory_pool(0);
    Ok(Q3Run { prefill_rows, decode_rows, resident_mb: (before - after) / 1e6, decode_tps })
}

#[test]
#[ignore = "실모델 (LOGIS_TEST_MODELS=1)"]
fn p2b_qwen3_q8_resident() -> R {
    if !enabled() {
        return Ok(());
    }
    let Some(dev) = gpu() else { return Ok(()) };
    let dir = models_dir().join("Qwen3-0.6B-Instruct-gguf");
    if !dir.exists() {
        println!("[P2B-TEST] {dir:?} 없음 → 건너뜀");
        return Ok(());
    }
    let path = dir.to_str().unwrap().to_string();
    let tok = tauri_app_lib::tokenizer::TokenizerModel::init(&path)?;
    let ids = tok.text_encode_vec(prompt_text(3), false)?;
    let Some(ids) = tokens_or_skip(ids, 301, "0.6B") else { return Ok(()) };

    let old = q3_run(&dev, &path, false, DType::BF16, &ids)?;
    let new = q3_run(&dev, &path, true, DType::BF16, &ids)?;
    let reference = q3_run(&dev, &path, true, DType::F32, &ids)?;

    let p_old = compare(&reference.prefill_rows, &old.prefill_rows);
    let p_new = compare(&reference.prefill_rows, &new.prefill_rows);
    let d_old = compare(&reference.decode_rows, &old.decode_rows);
    let d_new = compare(&reference.decode_rows, &new.decode_rows);
    let d_on = compare(&old.decode_rows, &new.decode_rows);
    println!(
        "[P2B-RESULT] Qwen3-0.6B Q8_0 · 기준 = Q8_0 정확 복원 가중치 + F32 활성/matmul/logits (어텐션은 두 경로와 같은 BF16 flash-attn)"
    );
    println!("[P2B-RESULT]  프리필 행 {} · 디코드 행 {}", old.prefill_rows.len(), old.decode_rows.len());
    for (name, c) in [("프리필 기존 BF16", &p_old), ("프리필 Q8 상주 ", &p_new), ("디코드 기존 BF16", &d_old), ("디코드 Q8 상주 ", &d_new)] {
        println!(
            "[P2B-RESULT]  {name} vs F32 기준: KLD {:.3e} · top-1 {:.1}% · top-5 {:.1}% · max|Δ| {:.3}",
            c.kld, c.top1, c.top5, c.max_abs
        );
    }
    println!("[P2B-RESULT]  디코드 Q8 상주 vs 기존 BF16: top-1 {:.1}% · max|Δ| {:.3}", d_on.top1, d_on.max_abs);
    println!(
        "[P2B-RESULT]  상주 VRAM: 기존 BF16 {:.0} MB → Q8 상주 {:.0} MB (Δ {:+.0} MB) · F32 기준 {:.0} MB",
        old.resident_mb,
        new.resident_mb,
        new.resident_mb - old.resident_mb,
        reference.resident_mb
    );
    println!("[P2B-RESULT]  디코드 속도: 기존 BF16 {:.1} tok/s → Q8 상주 {:.1} tok/s", old.decode_tps, new.decode_tps);
    let kld_old = (p_old.kld * p_old_n(&old) + d_old.kld * old.decode_rows.len() as f64) / total_n(&old);
    let kld_new = (p_new.kld * p_old_n(&new) + d_new.kld * new.decode_rows.len() as f64) / total_n(&new);
    println!("[P2B-RESULT]  전체 KLD vs F32 기준: 기존 {:.3e} · Q8 상주 {:.3e}", kld_old, kld_new);
    assert!(kld_new <= kld_old * 1.05, "Q8 상주 KLD {kld_new:.3e} > 기존 {kld_old:.3e}");
    assert!(d_new.top1 >= d_old.top1 - 1.0, "디코드 top-1 Q8 {:.1}% < 기존 {:.1}%", d_new.top1, d_old.top1);
    Ok(())
}

fn p_old_n(r: &Q3Run) -> f64 {
    r.prefill_rows.len() as f64
}

fn total_n(r: &Q3Run) -> f64 {
    (r.prefill_rows.len() + r.decode_rows.len()).max(1) as f64
}

// ─────────────────────────── KV 계층 강제 (체크리스트 §4) ───────────────────────────

struct TierRun {
    rows: Vec<Vec<f32>>,
    decode_ms: f64,
    free_before_mb: f64,
    free_after_mb: f64,
    /// 디코드 직후(동기화 전) 드라이버 free 와 메모리 풀 유휴분 (MB).
    raw_free_mb: f64,
    pool_idle_mb: f64,
}

/// teacher-forcing: 앞 200 토큰을 한 번에 프리필(KV 는 VRAM), 디코드 진입 시 KV 계층을 `tier` 로 강제한 뒤
/// 나머지를 한 토큰씩 디코드합니다 (generate.rs 의 `set_kv_residency(plan)` 과 같은 위치).
fn q3_tier_run(
    g: &mut tauri_app_lib::models::qwen3::generate::Qwen3GenerateModel,
    dev: &Device,
    ids: &[u32],
    exact: bool,
    tier: tauri_app_lib::utils::resources::KvResidency,
) -> anyhow::Result<TierRun> {
    use tauri_app_lib::utils::resources::KvResidency;
    let prefill = 200usize;
    let m = g.model_mut();
    m.clear_kv_cache();
    m.set_kv_residency(KvResidency::Vram);
    m.set_kv_exact(exact);
    let x = Tensor::from_slice(&ids[..prefill], (1, prefill), dev)?;
    let mut rows = vec![row_f32(&m.forward(Some(&x), None, 0)?)?];
    let free_before_mb = free_bytes(dev) / 1e6;
    m.set_kv_residency(tier);
    let t0 = Instant::now();
    for j in prefill..ids.len() - 1 {
        let x = Tensor::from_slice(&ids[j..j + 1], (1, 1), dev)?;
        rows.push(row_f32(&m.forward(Some(&x), None, j)?)?);
    }
    let decode_ms = t0.elapsed().as_secs_f64() * 1000.0 / (ids.len() - 1 - prefill).max(1) as f64;
    // 동기화 전에 잽니다: 디코드 동안 스트림 순서 메모리 풀이 쥐고 있는 양 (free_bytes 는 먼저 동기화합니다).
    let raw_free_mb = dev.mem_info().ok().flatten().map(|(f, _)| f as f64 / 1e6).unwrap_or(0.0);
    let pool_idle_mb = dev.pool_idle_bytes().unwrap_or(0) as f64 / 1e6;
    let free_after_mb = free_bytes(dev) / 1e6;
    m.clear_kv_cache();
    m.set_kv_residency(KvResidency::Vram);
    Ok(TierRun { rows, decode_ms, free_before_mb, free_after_mb, raw_free_mb, pool_idle_mb })
}

/// Qwen3-0.6B Q8 상주(BF16 활성): 디코드 진입 시 KV 를 Vram / Ram / Ssd 로 강제하고 같은 토큰 열의 로짓을
/// '원본 KV · Vram' 과 비교합니다. Qwen3 경로는 Ssd 를 Ram 으로 강등하고(model.rs), Ram 대피본은 FP8 이라
/// 비트 동일은 기대하지 않습니다 — top-1 · KLD 로 봅니다. 같은 모드 재실행은 비트 동일이어야 합니다.
#[test]
#[ignore = "실모델 (LOGIS_TEST_MODELS=1)"]
fn p2b_qwen3_kv_tier_forced() -> R {
    use tauri_app_lib::utils::resources::KvResidency;
    if !enabled() {
        return Ok(());
    }
    let Some(dev) = gpu() else { return Ok(()) };
    let dir = models_dir().join("Qwen3-0.6B-Instruct-gguf");
    if !dir.exists() {
        println!("[P2B-TEST] {dir:?} 없음 → 건너뜀");
        return Ok(());
    }
    let path = dir.to_str().unwrap().to_string();
    let tok = tauri_app_lib::tokenizer::TokenizerModel::init(&path)?;
    let ids = tok.text_encode_vec(prompt_text(3), false)?;
    let Some(ids) = tokens_or_skip(ids, 301, "0.6B") else { return Ok(()) };
    let _ = dev.trim_memory_pool(0);
    let start = free_bytes(&dev);
    let mut g = tauri_app_lib::models::qwen3::generate::Qwen3GenerateModel::init_from_gguf_with(
        &path,
        Some(&dev),
        Some(DType::BF16),
        true,
    )?;
    let modes = [
        ("원본 KV · Vram", true, KvResidency::Vram),
        ("원본 KV · Ram", true, KvResidency::Ram),
        ("원본 KV · Ssd", true, KvResidency::Ssd),
        ("압축 KV · Vram", false, KvResidency::Vram),
        ("압축 KV · Ram", false, KvResidency::Ram),
        ("원본 KV · Vram 재실행", true, KvResidency::Vram),
    ];
    let mut runs = Vec::new();
    for (name, exact, tier) in modes {
        runs.push((name, q3_tier_run(&mut g, &dev, &ids, exact, tier)?));
    }
    drop(g);
    let _ = dev.synchronize_all();
    let _ = dev.trim_memory_pool(0);
    let end = free_bytes(&dev);
    let base = &runs[0].1.rows;
    println!("[P2B-RESULT] Qwen3-0.6B Q8 상주 · KV 계층 강제 (프리필 200 + 디코드 {} 행, 기준 = 원본 KV · Vram)", base.len());
    let mut worst_top1 = 100f64;
    for (name, r) in &runs {
        let c = compare(base, &r.rows);
        worst_top1 = worst_top1.min(c.top1);
        println!(
            "[P2B-RESULT]  {name:<20}: max|Δ| {:.3e} · top-1 {:.1}% · top-5 {:.1}% · KLD {:.3e} · 디코드 {:.1} ms/토큰 · 실효 free {:.0} → {:.0} MB · 디코드 직후(동기화 전) 드라이버 free {:.0} MB + 풀 유휴 {:.0} MB",
            c.max_abs, c.top1, c.top5, c.kld, r.decode_ms, r.free_before_mb, r.free_after_mb, r.raw_free_mb, r.pool_idle_mb
        );
    }
    println!("[P2B-RESULT]  실효 free: 시작 {:.0} MB → 끝 {:.0} MB ({:+.0} MB)", start / 1e6, end / 1e6, (end - start) / 1e6);
    let det = compare(base, &runs[5].1.rows);
    assert_eq!(det.max_abs, 0.0, "같은 모드(원본 KV · Vram) 재실행이 비트 단위로 다릅니다");
    assert!(worst_top1 >= 90.0, "KV 계층 강제 결과의 top-1 이 {worst_top1:.1}% 로 낮습니다");
    Ok(())
}

//! 🌟 [VOCAB TRIM] 생성 모델 셋(Qwen3 0.6B · Qwen3.5 2B · Qwen3.5 4B)의 출력 투영 잘라 내기 실측 (`models::common::vocab_trim`).
//! `LOGIS_TEST_MODELS=1` + `-- --ignored` 로만 실행합니다. GPU 에는 한 번에 한 모델만 올립니다.
//!
//! - `vocab_trim_text`  : 텍스트 맥락. 같은 모델을 `off`(전체 어휘)와 잘라 낸 갈래(규칙 full × 언어 any · 12 개 · 문서 언어 자동)로
//!   읽어 견줍니다. 교사 강제(프리필 + 한 토큰씩 디코드): top-1 일치 · 잘린 확률 질량(off 가 잘린 행에 준 확률) · 다음 토큰이
//!   잘린 비율 · 남긴 행 위 KLD · 디코드 ms/토큰. 탐욕 생성 48 토큰이 off 와 같은 길이. 출력 투영 VRAM(적재 뒤 여유 메모리 차이).
//!   앱 전체 경로의 대조(N:N 자연어 검색 · 전처리)는 `synthetic_eval::run_plan` 에 `LOGIS_VOCAB_TRIM=off|full` 과
//!   `LOGIS_EVAL_NOW_MS`(같은 현재 시각)를 주어 두 번 돌려 견줍니다 (`SyntheticData/eval/README.md`).
//!   글은 한국어 무역 문서(+ 영어) · 영어 · 일본어(+ 영어) 셋이고, 자동 언어는 앱처럼 글로 정합니다 (`DocLangScope::text`).
//!   `LOGIS_TRIM_MODELS` (`q3,q35,4b`, 기본 모두) 로 고릅니다.
//! - `vocab_trim_image` : 이미지 맥락 (Qwen3.5 2B VLM). 합성 무역 서류 쪽 이미지(`LOGIS_TRIM_IMAGES` 폴더의 PNG)를 같은 지시문으로
//!   off 와 갈래마다 읽습니다. 자동 언어는 앱처럼 첫 판독 글로 정한 뒤(`DocLangScope::image` → `observe_read`) 다시 읽습니다.
//!   off 의 글과 같은가 · 글자 편집 거리 · 호출 시간.
//! - `vocab_trim_fetch_4b` : 원본 Qwen/Qwen3.5-4B 를 앱의 다운로드 로직(`lang_llm::request_download`)으로 받고 실행용 GGUF 로 바꿉니다.
#![cfg(any(feature = "cuda", feature = "rocm"))]

use std::path::PathBuf;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use logis_app::models::common::vocab_trim::{self, DocLangScope};

type R = anyhow::Result<()>;

fn enabled() -> bool {
    std::env::var("LOGIS_TEST_MODELS").map(|v| v == "1").unwrap_or(false)
}

fn models_dir() -> PathBuf {
    crate::common::models_dir()
}

fn gpu() -> Option<Device> {
    #[cfg(feature = "cuda")]
    if let Ok(d) = Device::new_cuda(0) {
        return Some(d);
    }
    #[cfg(feature = "rocm")]
    if let Some(d) = logis_app::utils::new_gpu_device(0) {
        return Some(d);
    }
    if std::env::var("LOGIS_REQUIRE_GPU").map(|v| v == "1").unwrap_or(false) {
        panic!("[TRIM-TEST] LOGIS_REQUIRE_GPU=1 이지만 GPU 를 열 수 없습니다!");
    }
    println!("[TRIM-TEST] GPU 를 열 수 없어 건너뜁니다");
    None
}

fn finish_gpu_test() {
    logis_app::flush_reflection_logs();
    if std::env::var("LOGIS_TEST_FAST_EXIT").map(|v| v == "1").unwrap_or(false) {
        logis_app::utils::device_utils::rocm_exit_now_if_used(0);
    }
}

fn free_mb(dev: &Device) -> f64 {
    let _ = dev.synchronize_all();
    let _ = dev.trim_memory_pool(0);
    dev.effective_free_memory().ok().flatten().unwrap_or(0) as f64 / 1e6
}

fn set_rule(trim: &str, langs: &str) {
    std::env::set_var(vocab_trim::TRIM_ENV, trim);
    std::env::set_var(vocab_trim::LANGS_ENV, langs);
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

/// 유한한 자리만의 log-softmax (잘린 행 −∞ 는 건너뜀)
fn log_softmax_on(v: &[f32], mask: &[bool]) -> Vec<f64> {
    let m = v.iter().zip(mask).filter(|(_, &k)| k).map(|(&x, _)| x as f64).fold(f64::NEG_INFINITY, f64::max);
    let s: f64 = v.iter().zip(mask).filter(|(_, &k)| k).map(|(&x, _)| (x as f64 - m).exp()).sum();
    v.iter().map(|&x| x as f64 - m - s.ln()).collect()
}

/// 한 모델의 생성기 (테스트가 쓰는 만큼만)
enum Gen {
    Q3(logis_app::models::qwen3::generate::Qwen3GenerateModel),
    Q35(Box<logis_app::models::qwen3_5::generate::Qwen3_5GenerateModel>),
}

impl Gen {
    fn load(kind: &str, dev: &Device) -> anyhow::Result<Self> {
        Ok(match kind {
            "q3" => {
                let dir = models_dir().join("Qwen3-0.6B-Instruct-gguf");
                Self::Q3(logis_app::models::qwen3::generate::Qwen3GenerateModel::init_from_gguf(dir.to_str().unwrap(), Some(dev), Some(DType::BF16))?)
            }
            "q35" => {
                let f = models_dir().join("Qwen3.5-2B-Instruct-gguf").join("Qwen3.5-2B-Q8_0.gguf");
                Self::Q35(Box::new(logis_app::models::qwen3_5::generate::Qwen3_5GenerateModel::init_from_gguf(f.to_str().unwrap(), None, Some(dev))?))
            }
            _ => {
                let dir = logis_app::model::lang_llm::model_dir(logis_app::model::lang_llm::LANG_ENGINE);
                Self::Q35(Box::new(logis_app::model::lang_gguf::load_runtime(&dir, dev)?))
            }
        })
    }

    fn available(kind: &str) -> bool {
        match kind {
            "q3" => models_dir().join("Qwen3-0.6B-Instruct-gguf").join("Qwen3-0.6B-Q8_0.gguf").exists(),
            "q35" => models_dir().join("Qwen3.5-2B-Instruct-gguf").join("Qwen3.5-2B-Q8_0.gguf").exists(),
            _ => logis_app::model::lang_llm::is_ready(logis_app::model::lang_llm::LANG_ENGINE),
        }
    }

    fn label(kind: &str) -> &'static str {
        match kind {
            "q3" => "Qwen3 0.6B (Q8_0)",
            "q35" => "Qwen3.5 2B (Q8_0)",
            _ => "Qwen3.5 4B (원본 → Q4_K_M)",
        }
    }

    fn tokenizer(&self, kind: &str) -> anyhow::Result<tokenizers::Tokenizer> {
        Ok(match self {
            Self::Q35(g) => g.tokenizer.tokenizer.clone(),
            Self::Q3(_) => {
                let _ = kind;
                logis_app::tokenizer::TokenizerModel::init(models_dir().join("Qwen3-0.6B-Instruct-gguf").to_str().unwrap())?.tokenizer
            }
        })
    }

    fn refresh(&mut self) -> anyhow::Result<()> {
        match self {
            Self::Q3(g) => g.refresh_vocab(),
            Self::Q35(g) => g.refresh_vocab(),
        }
    }

    fn head_rows(&mut self) -> Option<usize> {
        match self {
            Self::Q3(g) => g.model_mut().trim_head.as_ref().map(|h| h.kept.len()),
            Self::Q35(g) => g.qwen3_5.trim_head.as_ref().map(|h| h.kept.len()),
        }
    }

    fn begin(&mut self, ids: &[u32], dev: &Device) -> anyhow::Result<()> {
        match self {
            Self::Q3(g) => {
                g.model_mut().clear_kv_cache();
            }
            Self::Q35(g) => {
                logis_app::models::qwen3_5::model::IGNORE_STOP_SIGNAL_FOR_TESTS.store(true, std::sync::atomic::Ordering::Relaxed);
                g.clear_kv_cache();
                let full = Tensor::from_slice(ids, (1, ids.len()), dev)?;
                let _ = g.qwen3_5.compute_and_set_rope_deltas(&full, None, None);
            }
        }
        Ok(())
    }

    /// `ids` 를 `offset` 부터 넣고 마지막 자리의 로짓
    fn step(&mut self, rt: &tokio::runtime::Runtime, ids: &[u32], offset: usize, dev: &Device) -> anyhow::Result<Vec<f32>> {
        let x = Tensor::from_slice(ids, (1, ids.len()), dev)?;
        match self {
            Self::Q3(g) => row_f32(&g.model_mut().forward(Some(&x), None, offset)?),
            Self::Q35(g) => row_f32(&rt.block_on(g.qwen3_5.forward(&x, None, None, None, None, None, offset, None, None))?),
        }
    }

    /// Production fast greedy step (`forward_last_hidden` + `greedy_from_hidden_with_stats`, zero full-vocab logits expansion).
    fn step_greedy_fast(
        &mut self,
        rt: &tokio::runtime::Runtime,
        ids: &[u32],
        offset: usize,
        dev: &Device,
    ) -> anyhow::Result<(u32, vocab_trim::StepGuardStats)> {
        let x = Tensor::from_slice(ids, (1, ids.len()), dev)?;
        match self {
            Self::Q3(g) => {
                let h = g.model_mut().forward_last_hidden(Some(&x), None, offset)?;
                g.model_mut().greedy_from_hidden_with_stats(&h)
            }
            Self::Q35(g) => {
                let h = rt.block_on(g.qwen3_5.forward_last_hidden(&x, None, None, None, None, None, offset, None, None))?;
                g.qwen3_5.greedy_from_hidden_with_stats(&h)
            }
        }
    }

    fn resident_vram_bytes(&self) -> u64 {
        match self {
            Self::Q3(g) => g.resident_vram_bytes(),
            Self::Q35(g) => g.resident_vram_bytes(),
        }
    }

    fn refresh_for(&mut self, prompt: &str) -> anyhow::Result<()> {
        match self {
            Self::Q3(g) => g.refresh_vocab_for(prompt, false),
            Self::Q35(g) => g.refresh_vocab_for(prompt, false),
        }
    }

    fn eos_ids(&self) -> Vec<u32> {
        match self {
            Self::Q3(g) => {
                let mut v = Vec::new();
                if let Some(id) = g.tokenizer.tokenizer.token_to_id("<|im_end|>") { v.push(id); }
                if let Some(id) = g.tokenizer.tokenizer.token_to_id("<|endoftext|>") { v.push(id); }
                v
            }
            Self::Q35(g) => {
                let mut v = vec![g.eos_token_id];
                if let Some(id) = g.tokenizer.tokenizer.token_to_id("<|im_end|>") { if !v.contains(&id) { v.push(id); } }
                v
            }
        }
    }

    fn load_from_dir(kind: &str, dir: &std::path::Path, dev: &Device) -> anyhow::Result<Self> {
        Ok(match kind {
            "q3" => Self::Q3(logis_app::models::qwen3::generate::Qwen3GenerateModel::init_from_gguf(
                dir.to_str().unwrap(),
                Some(dev),
                Some(DType::BF16),
            )?),
            _ => {
                // Find .gguf directly in dir or runtime-q4km/model-q4km.gguf
                let direct_gguf = std::fs::read_dir(dir)?
                    .flatten()
                    .map(|e| e.path())
                    .find(|p| p.extension().is_some_and(|e| e == "gguf") && !p.file_name().unwrap_or_default().to_string_lossy().contains("mmproj"));
                if let Some(gf) = direct_gguf {
                    Self::Q35(Box::new(logis_app::models::qwen3_5::generate::Qwen3_5GenerateModel::init_from_gguf(
                        gf.to_str().unwrap(),
                        None,
                        Some(dev),
                    )?))
                } else {
                    Self::Q35(Box::new(logis_app::model::lang_gguf::load_runtime(dir, dev)?))
                }
            }
        })
    }
}

struct Teacher {
    rows: Vec<Vec<f32>>,
    decode_ms: f64,
}

/// 교사 강제: 앞 `warm` 토큰을 프리필하고 나머지를 한 토큰씩 디코드하며 다음 토큰의 로짓을 남깁니다.
fn teacher(g: &mut Gen, rt: &tokio::runtime::Runtime, dev: &Device, ids: &[u32], warm: usize) -> anyhow::Result<Teacher> {
    g.begin(ids, dev)?;
    let mut rows = vec![g.step(rt, &ids[..warm], 0, dev)?];
    let t0 = Instant::now();
    for j in warm..ids.len() - 1 {
        rows.push(g.step(rt, &ids[j..j + 1], j, dev)?);
    }
    let decode_ms = t0.elapsed().as_secs_f64() * 1000.0 / (ids.len() - 1 - warm).max(1) as f64;
    Ok(Teacher { rows, decode_ms })
}

fn greedy(g: &mut Gen, rt: &tokio::runtime::Runtime, dev: &Device, prompt: &[u32], steps: usize) -> anyhow::Result<Vec<u32>> {
    g.begin(prompt, dev)?;
    let mut last = g.step(rt, prompt, 0, dev)?;
    let mut out = Vec::with_capacity(steps);
    for j in 0..steps {
        let next = argmax(&last) as u32;
        out.push(next);
        last = g.step(rt, &[next], prompt.len() + j, dev)?;
    }
    Ok(out)
}

struct Versus {
    top1: f64,
    cut_mass: f64,
    target_cut: f64,
    kept_kld: f64,
}

fn versus(off: &[Vec<f32>], arm: &[Vec<f32>], next_ids: &[u32]) -> Versus {
    let n = off.len().min(arm.len()).max(1);
    let (mut t1, mut mass, mut tcut, mut kl) = (0f64, 0f64, 0f64, 0f64);
    for (i, (r, y)) in off.iter().zip(arm).enumerate() {
        t1 += (argmax(r) == argmax(y)) as u32 as f64;
        let all = vec![true; r.len()];
        let lr = log_softmax_on(r, &all);
        let kept: Vec<bool> = y.iter().map(|v| v.is_finite()).collect();
        mass += lr.iter().zip(&kept).filter(|(_, &k)| !k).map(|(l, _)| l.exp()).sum::<f64>();
        if let Some(&t) = next_ids.get(i) {
            tcut += !kept[t as usize] as u32 as f64;
        }
        let (a, b) = (log_softmax_on(r, &kept), log_softmax_on(y, &kept));
        kl += a.iter().zip(&b).zip(&kept).filter(|(_, &k)| k).map(|((&p, &q), _)| p.exp() * (p - q)).sum::<f64>();
    }
    let n = n as f64;
    Versus { top1: t1 / n * 100.0, cut_mass: mass / n, target_cut: tcut / n * 100.0, kept_kld: kl / n }
}

fn same_prefix(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// 텍스트 맥락의 글 셋 (이름, 글)
fn texts() -> Vec<(&'static str, String)> {
    let ko = "선하증권 BL-2026-08814 에 따르면 화주 글로벌테크로지스틱스(주)는 부산항에서 로스앤젤레스항으로 산업용 마이크로컨트롤러 \
        1,200 상자를 선적했으며, 신용장 LC-88492011 의 조건에 따라 상업송장 CI-2026-08001 과 포장명세서, 원산지증명서를 함께 제출해야 합니다. \
        수입자 노바임포터스는 도착통지서를 받은 뒤 관세사를 통해 통관을 진행하고, 적하보험증권의 보험금액은 송장 금액의 110% 로 정했습니다. \
        운임은 선불(Freight Prepaid)이며 컨테이너 번호는 MSKU 7731025, 봉인 번호는 SL-559102 입니다. 결제 조건은 일람불 신용장이고 \
        선적 기한은 2026년 9월 30일, 분할 선적은 허용되지 않습니다. 다음 내용을 JSON 으로 정리하세요: 송하인, 수하인, 선적항, 도착항, 품명, 수량.";
    let en = "According to bill of lading BL-2026-08814, the shipper Global Tech Logistics Co., Ltd. loaded 1,200 cartons of industrial \
        microcontrollers at the port of Busan bound for Los Angeles. Under letter of credit LC-88492011 the commercial invoice CI-2026-08001, \
        the packing list and the certificate of origin must be presented together. The importer Nova Importers will clear customs through a \
        licensed broker after receiving the arrival notice, and the cargo insurance amount is 110% of the invoice value. Freight is prepaid, \
        the container number is MSKU 7731025 and the seal number is SL-559102. Extract shipper, consignee, ports, goods and quantity as JSON.";
    let ja = "船荷証券 BL-2026-08814 によると、荷送人グローバルテックロジスティクス株式会社は釜山港からロサンゼルス港へ産業用マイクロコントローラ \
        1,200 箱を船積みしました。信用状 LC-88492011 の条件に従い、商業送り状 CI-2026-08001、梱包明細書、原産地証明書を併せて提出する必要があります。\
        輸入者ノバインポーターズは到着通知を受け取った後、通関業者を通じて通関手続きを行い、貨物保険の保険金額は送り状金額の110%とします。\
        運賃は前払い(Freight Prepaid)で、コンテナ番号は MSKU 7731025、シール番号は SL-559102 です。荷送人、荷受人、港、品名、数量を JSON で抽出してください。";
    vec![("ko", ko.repeat(2)), ("en", en.repeat(2)), ("ja", ja.repeat(2))]
}

/// 갈래 (이름, 규칙, 언어). `auto` 는 앱처럼 글로 정한 문서 언어
const ARMS: [(&str, &str, &str); 3] = [
    ("full · any", "full", "any"),
    ("full · 12개 언어", "full", "all"),
    ("full · 문서 언어", "full", "auto"),
];

#[test]
#[ignore = "실모델 (LOGIS_TEST_MODELS=1)"]
fn vocab_trim_text() -> R {
    if !enabled() {
        return Ok(());
    }
    let Some(dev) = gpu() else { return Ok(()) };
    let pick = std::env::var("LOGIS_TRIM_MODELS").unwrap_or_else(|_| "q3,q35,4b".into());
    let rt = tokio::runtime::Runtime::new()?;
    for kind in pick.split(',').map(str::trim).filter(|k| !k.is_empty()) {
        if !Gen::available(kind) {
            println!("[TRIM-RESULT] {} 없음 → 건너뜀", Gen::label(kind));
            continue;
        }
        let (warm, total, steps) = (160usize, 260usize, 48usize);
        // ── off: 전체 어휘 (따로 올림)
        set_rule("off", "all");
        let base_free = free_mb(&dev);
        let mut g = Gen::load(kind, &dev)?;
        let off_mb = base_free - free_mb(&dev);
        let tok = g.tokenizer(kind)?;
        let mut cases = Vec::new();
        for (name, text) in texts() {
            let ids = tok.encode(text.as_str(), false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
            if ids.len() < total {
                println!("[TRIM-RESULT] {name}: 토큰 {} < {total} → 건너뜀", ids.len());
                continue;
            }
            let ids = ids[..total].to_vec();
            let t = teacher(&mut g, &rt, &dev, &ids, warm)?;
            let gr = greedy(&mut g, &rt, &dev, &ids[..warm], steps)?;
            cases.push((name, text, ids, t, gr));
        }
        drop(g);
        let vocab = tok.get_vocab_size(true);
        println!("[TRIM-RESULT] ===== {} · 어휘 {vocab} · off 적재 {off_mb:.0} MB =====", Gen::label(kind));
        for (name, _, _, t, _) in &cases {
            println!("[TRIM-RESULT]  [{name}] off: 디코드 {:.2} ms/토큰", t.decode_ms);
        }
        // ── 잘라 낸 갈래: 한 번 올리고(full 로 매핑) 규칙 · 언어만 바꿔 출력 투영을 다시 모읍니다
        set_rule("full", "any");
        let base_free = free_mb(&dev);
        let mut g = Gen::load(kind, &dev)?;
        for (arm, rule, langs) in ARMS {
            for (name, text, ids, off, off_greedy) in &cases {
                set_rule(rule, langs);
                let doc_lang = logis_app::utils::lang_utils::detect_document_language(text);
                let _scope = (langs == "auto").then(|| DocLangScope::text(&doc_lang, text));
                g.refresh()?;
                let rows = g.head_rows().unwrap_or(vocab);
                let used_mb = base_free - free_mb(&dev);
                let t = teacher(&mut g, &rt, &dev, ids, warm)?;
                let gr = greedy(&mut g, &rt, &dev, &ids[..warm], steps)?;
                let v = versus(&off.rows, &t.rows, &ids[warm..]);
                let label = if langs == "auto" { format!("{arm}({})", vocab_trim::current_langs().label()) } else { arm.to_string() };
                println!(
                    "[TRIM-RESULT]  [{name}] {label}: 행 {rows} ({:.1}%) · 적재 {used_mb:.0} MB (off 대비 {:+.0}) · top-1 {:.1}% · 잘린 질량 {:.2e} · 다음 토큰 잘림 {:.1}% · 남긴 행 KLD {:.2e} · 탐욕 {}/{steps} · 디코드 {:.2} ms/토큰 ({:+.1}%)",
                    rows as f64 / vocab as f64 * 100.0,
                    used_mb - off_mb,
                    v.top1,
                    v.cut_mass,
                    v.target_cut,
                    v.kept_kld,
                    same_prefix(off_greedy, &gr),
                    t.decode_ms,
                    (t.decode_ms / off.decode_ms - 1.0) * 100.0
                );
            }
        }
        drop(g);
        set_rule("full", "auto");
    }
    finish_gpu_test();
    Ok(())
}

fn edit_distance(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1];
        for (j, cb) in b.iter().enumerate() {
            cur.push((prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + (ca != cb) as usize));
        }
        prev = cur;
    }
    prev[b.len()]
}

fn read_image(
    rt: &tokio::runtime::Runtime,
    g: &mut logis_app::models::qwen3_5::generate::Qwen3_5GenerateModel,
    img: &image::DynamicImage,
) -> anyhow::Result<(String, f64)> {
    use logis_app::openai_types::*;
    let (url, _guard) = logis_app::utils::img_utils::register_mem_image(img.clone());
    let user = ChatCompletionRequestUserMessage {
        content: ChatCompletionRequestUserMessageContent::Array(vec![
            ChatCompletionRequestMessageContentPart::ImageURL(ChatCompletionRequestMessageContentPartImage { image_url: ImageURL { url, detail: None } }),
            ChatCompletionRequestMessageContentPart::Text(ChatCompletionRequestMessageContentPartText {
                text: "Transcribe every line of text in this document image exactly as printed, top to bottom. Output only the text.".into(),
            }),
        ]),
        name: None,
    };
    let params = ChatCompletionParameters {
        messages: vec![
            ChatCompletionRequestMessage::System(ChatCompletionRequestSystemMessage { content: "You are an OCR engine.".into(), name: None }),
            ChatCompletionRequestMessage::User(user),
        ],
        model: "qwen3.5".into(),
        max_tokens: Some(200),
        temperature: Some(0.0),
        top_p: Some(0.95),
        ..Default::default()
    };
    let t0 = Instant::now();
    let out = rt.block_on(g.generate(params, None, None, Some("inference".into()), None, None))?;
    Ok((out, t0.elapsed().as_secs_f64()))
}

#[test]
#[ignore = "실모델 (LOGIS_TEST_MODELS=1, LOGIS_TRIM_IMAGES)"]
fn vocab_trim_image() -> R {
    if !enabled() {
        return Ok(());
    }
    let Ok(dir) = std::env::var("LOGIS_TRIM_IMAGES") else {
        println!("[TRIM-TEST] LOGIS_TRIM_IMAGES 없음 → 건너뜀");
        return Ok(());
    };
    let Some(dev) = gpu() else { return Ok(()) };
    let base = models_dir().join("Qwen3.5-2B-Instruct-gguf");
    let (model, mmproj) = (base.join("Qwen3.5-2B-Q8_0.gguf"), base.join("mmproj-BF16.gguf"));
    if !model.exists() || !mmproj.exists() {
        println!("[TRIM-TEST] {base:?} 없음 → 건너뜀");
        return Ok(());
    }
    let mut pages: Vec<(String, image::DynamicImage)> = Vec::new();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)?.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "png")).collect();
    files.sort();
    for p in files {
        pages.push((p.file_stem().unwrap().to_string_lossy().to_string(), image::open(&p)?));
    }
    let rt = tokio::runtime::Runtime::new()?;
    let load = |trim: &str| -> anyhow::Result<logis_app::models::qwen3_5::generate::Qwen3_5GenerateModel> {
        set_rule(trim, "any");
        logis_app::models::qwen3_5::generate::Qwen3_5GenerateModel::init_from_gguf(model.to_str().unwrap(), Some(mmproj.to_str().unwrap()), Some(&dev))
    };
    // ── off
    let mut g = load("off")?;
    let mut off = Vec::new();
    for (name, img) in &pages {
        let (text, s) = read_image(&rt, &mut g, img)?;
        println!("[TRIM-RESULT] [{name}] off: {s:.1} s · {} 자", text.chars().count());
        off.push((text, s));
    }
    drop(g);
    // ── 잘라 낸 갈래
    let mut g = load("full")?;
    for (arm, rule, langs) in ARMS {
        let (mut same, mut edits, mut chars, mut secs) = (0usize, 0usize, 0usize, 0f64);
        for ((name, img), (off_text, _)) in pages.iter().zip(&off) {
            set_rule(rule, langs);
            let (text, s, decided) = if langs == "auto" {
                // 앱의 이미지 맥락: 아무 문자 체계도 빼지 않고 첫 판독 → 그 글로 문서 언어를 정해 다시 읽음
                let _scope = DocLangScope::image();
                let (first, s1) = read_image(&rt, &mut g, img)?;
                let decided = vocab_trim::observe_read(&first).map(|l| l.label()).unwrap_or_else(|| "정하지 못함(any)".into());
                let (text, s2) = read_image(&rt, &mut g, img)?;
                (text, s1 + s2, decided)
            } else {
                let (text, s) = read_image(&rt, &mut g, img)?;
                (text, s, langs.to_string())
            };
            let (a, b): (Vec<char>, Vec<char>) = (off_text.chars().collect(), text.chars().collect());
            let e = edit_distance(&a, &b);
            same += (e == 0) as usize;
            edits += e;
            chars += a.len();
            secs += s;
            println!("[TRIM-RESULT] [{name}] {arm}: 언어 {decided} · off 와 편집 {e} / {} 자 · {s:.1} s", a.len());
        }
        println!(
            "[TRIM-RESULT] == {arm}: off 와 같음 {same}/{} · 편집 비율 {:.4} · 합계 {secs:.1} s (off {:.1} s)",
            pages.len(),
            edits as f64 / chars.max(1) as f64,
            off.iter().map(|(_, s)| s).sum::<f64>()
        );
    }
    drop(g);
    set_rule("full", "auto");
    finish_gpu_test();
    Ok(())
}

#[test]
#[ignore = "실모델 · 네트워크 (LOGIS_TEST_MODELS=1): 원본 Qwen/Qwen3.5-4B 를 받고 실행용 GGUF 로 바꿉니다"]
fn vocab_trim_fetch_4b() -> R {
    if !enabled() {
        return Ok(());
    }
    use logis_app::model::lang_llm;
    let code = lang_llm::LANG_ENGINE;
    if lang_llm::is_ready(code) {
        println!("[TRIM-TEST] {} 준비됨: {:?}", lang_llm::repo_id(code), lang_llm::model_dir(code));
        return Ok(());
    }
    // Tauri 앱 없이: 엔진 초기화만 하고 진행 이벤트는 버립니다 (예전 build_headless_app 과 같은 초기화).
    logis_app::init_headless();
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let phase = lang_llm::request_download(code, &logis_app::sink::noop(), "trim_fetch_4b");
        println!("[TRIM-TEST] 다운로드 요청: {phase:?}");
        let t0 = Instant::now();
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(30)).await;
            let st = lang_llm::state(code);
            println!("[TRIM-TEST] {:.0} 분 · {:?} {}", t0.elapsed().as_secs_f64() / 60.0, st.phase, st.error);
            match st.phase {
                lang_llm::DlPhase::Ready => break,
                lang_llm::DlPhase::Failed | lang_llm::DlPhase::Unavailable => anyhow::bail!("받지 못했습니다: {}", st.error),
                _ => {}
            }
        }
        Ok(())
    })?;
    assert!(lang_llm::is_ready(code));
    println!("[TRIM-RESULT] {} 준비 완료: {:?} · 실행용 {:.0} MB", lang_llm::repo_id(code), lang_llm::model_dir(code), logis_app::model::lang_gguf::runtime_bytes(&lang_llm::model_dir(code)) as f64 / 1e6);
    Ok(())
}

/// 12 Core languages + English identifier/JSON extraction suite for AlphaEdge head-to-head benchmarking.
fn bench_12lang_prompts() -> Vec<(&'static str, &'static str, &'static str)> {
    vec![
        ("ko", "ko", "선하증권 BL-2026-08814 화주 글로벌테크(주) 품목 산업용 마이크로컨트롤러 1,200상자 컨테이너 MSKU 7731025 운임 선불. 아래 항목을 JSON으로 정리하세요: {\"goods\": \"...\", \"condition\": \"...\", \"container\": \"...\"}\n{"),
        ("en", "en", "Commercial Invoice CI-2026-08001, Bill of Lading BL-2026-08814, Container MSKU 7731025. Extract fields goods, condition, transcription as JSON:\n{\"goods\": \"industrial microcontrollers\", \"condition\": \"new\", \"transcription\": \""),
        ("ja", "ja", "商業送り状 CI-2026-08001 荷送人 グローバルテック株式会社 貨物 マイクロコントローラ 1,200箱 コンテナ MSKU 7731025。JSON形式で抽出してください:\n{"),
        ("zh", "zh-hans", "商业发票 CI-2026-08001 发货人 环球科技物流有限公司 货物 工业微控制器 1,200箱 集装箱号 MSKU 7731025。请以JSON格式提取:\n{"),
        ("ar", "ar", "فاتورة تجارية رقم CI-2026-08001 بوليصة شحن BL-2026-08814 الشاحن شركة جلوبال تك للوجستيات البضاعة متحكمات دقيقة صناعية 1200 صندوق الحاوية MSKU 7731025. استخرج البيانات بصيغة JSON:\n{"),
        ("de", "de", "Handelsrechnung CI-2026-08001, Konnossement BL-2026-08814, Container MSKU 7731025, Ware: Industrielle Mikrocontroller, 1.200 Kartons, Fracht vorausbezahlt. Extrahieren Sie als JSON:\n{"),
        ("fr", "fr", "Facture commerciale CI-2026-08001, Connaissement BL-2026-08814, Conteneur MSKU 7731025, Marchandise: Microcontrôleurs industriels, 1 200 cartons, Port payé. Extrayez en JSON:\n{"),
        ("es", "es", "Factura comercial CI-2026-08001, Conocimiento de embarque BL-2026-08814, Contenedor MSKU 7731025, Mercancía: Microcontroladores industriales, 1.200 cajas. Extraiga en formato JSON:\n{"),
        ("it", "it", "Fattura commerciale CI-2026-08001, Polizza di carico BL-2026-08814, Container MSKU 7731025, Merce: Microcontrollori industriali, 1.200 cartoni. Estrai in formato JSON:\n{"),
        ("pt", "pt", "Fatura comercial CI-2026-08001, Conhecimento de embarque BL-2026-08814, Contêiner MSKU 7731025, Mercadoria: Microcontroladores industriais, 1.200 caixas. Extraia em JSON:\n{"),
        ("nl", "nl", "Handelsfactuur CI-2026-08001, Cognossement BL-2026-08814, Container MSKU 7731025, Goederen: Industriële microcontrollers, 1.200 dozen. Extraheer als JSON:\n{"),
        ("cs", "cs", "Obchodní faktura CI-2026-08001, Konosament BL-2026-08814, Kontejner MSKU 7731025, Zboží: Průmyslové mikrokontroléry, 1 200 kartonů. Extrahujte jako JSON:\n{"),
        ("id_en", "en", "Copy these exact keys and values verbatim into JSON without changing a single character: goods=microcontroller, condition=new, transcription=BL-2026-08814, container=MSKU 7731025, invoice=CI-2026-08001, lc=LC-88492011.\n{\"goods\": \""),
        ("tr_ko", "ko", "다음 영문 물류 상품명을 자연스러운 한국어 외래어 표기로 음차하세요: Microcontroller, Logistics, Container, Commercial Invoice, Express Shipping -> "),
    ]
}

fn count_orphan_bpe_tokens(tok_path: &std::path::Path) -> usize {
    let Ok(bytes) = std::fs::read(tok_path) else { return 0 };
    let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else { return 0 };
    let Some(model) = v.get("model") else { return 0 };
    let Some(vocab) = model.get("vocab").and_then(|x| x.as_object()) else { return 0 };
    let Some(merges) = model.get("merges").and_then(|x| x.as_array()) else { return 0 };
    let mut reachable: std::collections::HashSet<String> = std::collections::HashSet::new();
    for k in vocab.keys() {
        if k.chars().count() == 1 {
            reachable.insert(k.clone());
        }
    }
    let mut changed = true;
    while changed {
        changed = false;
        for m in merges {
            let pair = if let Some(s) = m.as_str() {
                s.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string()))
            } else if let Some(arr) = m.as_array() {
                if arr.len() == 2 {
                    Some((arr[0].as_str().unwrap_or("").to_string(), arr[1].as_str().unwrap_or("").to_string()))
                } else { None }
            } else { None };
            if let Some((a, b)) = pair {
                if reachable.contains(&a) && reachable.contains(&b) {
                    let ab = format!("{a}{b}");
                    if vocab.contains_key(&ab) && reachable.insert(ab) {
                        changed = true;
                    }
                }
            }
        }
    }
    let mut orphans = 0usize;
    for k in vocab.keys() {
        if k.chars().count() > 1 && !reachable.contains(k) {
            orphans += 1;
        }
    }
    orphans
}

#[derive(serde::Serialize)]
struct ArmBenchResult {
    arm: String,
    dir: String,
    vocab_size: usize,
    active_k_mean: f64,
    vram_mb: f64,
    resident_model_mb: f64,
    vram_delta_mb: f64,
    decode_ms_per_tok: f64,
    total_gen_ms: f64,
    prefill_ms_total: f64,
    total_tokens: usize,
    fertility_tok_per_char: f64,
    orphan_bpe_tokens: usize,
    oracle_char_sim: f64,
    per_lang_oracle_sim: std::collections::BTreeMap<String, f64>,
    identifier_exact_rate: f64,
    precision_score: f64,
    guard_rescue_rate: f64,
    guard_rescue_ms: f64,
    guard_rest_wins: usize,
    quant_mode_a: bool,
    outputs: std::collections::BTreeMap<String, String>,
}

fn run_bench_arm(
    arm_name: &str,
    kind: &str,
    dir: &std::path::Path,
    trim: &str,
    langs_env: &str,
    guard: &str,
    active_k: Option<&str>,
    exec_mode: Option<&str>,
    dev: &Device,
    rt: &tokio::runtime::Runtime,
) -> anyhow::Result<ArmBenchResult> {
    std::env::set_var(vocab_trim::TRIM_ENV, trim);
    std::env::set_var(vocab_trim::LANGS_ENV, langs_env);
    std::env::set_var(vocab_trim::GUARD_ENV, guard);
    match active_k {
        Some(k) if !k.is_empty() => std::env::set_var(vocab_trim::ACTIVE_K_ENV, k),
        _ => std::env::remove_var(vocab_trim::ACTIVE_K_ENV),
    }
    match exec_mode {
        Some(m) if !m.is_empty() => std::env::set_var("LOGIS_VOCAB_EXEC_MODE", m),
        _ => std::env::remove_var("LOGIS_VOCAB_EXEC_MODE"),
    }

    let base_free = free_mb(dev);
    let mut g = Gen::load_from_dir(kind, dir, dev)?;
    let tok_path = dir.join("tokenizer.json");
    let tok = if tok_path.exists() {
        tokenizers::Tokenizer::from_file(&tok_path).map_err(anyhow::Error::msg)?
    } else {
        g.tokenizer(kind)?
    };
    let vocab_size = tok.get_vocab_size(true);
    let orphan_bpe_tokens = if tok_path.exists() { count_orphan_bpe_tokens(&tok_path) } else { 0 };
    let eos_ids = g.eos_ids();

    // Warmup 1 prompt (not timed)
    {
        let warm_p = "Warmup: BL-2026-08814\n{";
        let _sc = DocLangScope::text("en", warm_p);
        g.refresh_for(warm_p)?;
        let wids = tok.encode(warm_p, false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        g.begin(&wids, dev)?;
        let (mut next, _) = g.step_greedy_fast(rt, &wids, 0, dev)?;
        for s in 0..4usize {
            let (n2, _) = g.step_greedy_fast(rt, &[next], wids.len() + s, dev)?;
            next = n2;
        }
    }

    let mut max_resident_model_mb = g.resident_vram_bytes() as f64 / 1e6;
    let post_warm_free = free_mb(dev);
    let mut min_free = post_warm_free;

    let prompts = bench_12lang_prompts();
    let max_steps: usize = std::env::var("LOGIS_BENCH_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let target_chars: usize = std::env::var("LOGIS_BENCH_TARGET_CHARS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    let mut total_prefill_ms = 0.0f64;
    let mut total_decode_ms = 0.0f64;
    let mut total_tokens = 0usize;
    let mut total_prompt_tokens = 0usize;
    let mut total_prompt_chars = 0usize;
    let mut active_k_sum = 0usize;
    let mut cum_guard = vocab_trim::StepGuardStats::default();
    let mut outputs = std::collections::BTreeMap::new();

    for (pid, lang_hint, prompt) in &prompts {
        let _sc = DocLangScope::text(lang_hint, prompt);
        g.refresh_for(prompt)?;
        max_resident_model_mb = max_resident_model_mb.max(g.resident_vram_bytes() as f64 / 1e6);
        active_k_sum += g.head_rows().unwrap_or(vocab_size);
        let ids = tok.encode(*prompt, false).map_err(anyhow::Error::msg)?.get_ids().to_vec();
        total_prompt_tokens += ids.len();
        total_prompt_chars += prompt.chars().count();

        g.begin(&ids, dev)?;
        let tp0 = Instant::now();
        let (mut next, st0) = g.step_greedy_fast(rt, &ids, 0, dev)?;
        let _ = dev.synchronize_all();
        total_prefill_ms += tp0.elapsed().as_secs_f64() * 1e3;
        cum_guard.merge(&st0);

        let mut gen_ids = vec![next];
        let td0 = Instant::now();
        // Generate until target_chars reached (or max_steps*2 hard cap) so fertility differences show in total_gen_ms!
        let cap = (max_steps * 2).min(96);
        for s in 0..cap {
            if eos_ids.contains(&next) {
                break;
            }
            if s + 1 >= max_steps {
                let dec_so_far = tok.decode(&gen_ids, true).unwrap_or_default();
                if dec_so_far.chars().count() >= target_chars {
                    break;
                }
            }
            let (n2, st) = g.step_greedy_fast(rt, &[next], ids.len() + s, dev)?;
            cum_guard.merge(&st);
            gen_ids.push(n2);
            next = n2;
        }
        let _ = dev.synchronize_all();
        total_decode_ms += td0.elapsed().as_secs_f64() * 1e3;
        total_tokens += gen_ids.len().saturating_sub(1).max(1);

        let cur_free = free_mb(dev);
        if cur_free < min_free {
            min_free = cur_free;
        }
        let decoded = tok.decode(&gen_ids, true).unwrap_or_default();
        outputs.insert((*pid).to_string(), decoded);
    }

    let resident_model_mb = max_resident_model_mb;
    let vram_delta_mb = (base_free - min_free).max(0.0);
    // Use resident_model_mb as exact weight+head footprint (and verify vram_delta_mb alongside it)
    let vram_mb = if resident_model_mb > 10.0 { resident_model_mb } else { vram_delta_mb };
    let decode_ms_per_tok = total_decode_ms / total_tokens.max(1) as f64;
    let fertility = total_prompt_tokens as f64 / total_prompt_chars.max(1) as f64;
    let active_k_mean = active_k_sum as f64 / prompts.len().max(1) as f64;

    drop(g);
    let _ = free_mb(dev);

    Ok(ArmBenchResult {
        arm: arm_name.to_string(),
        dir: dir.to_string_lossy().to_string(),
        vocab_size,
        active_k_mean,
        vram_mb,
        resident_model_mb,
        vram_delta_mb,
        decode_ms_per_tok,
        total_gen_ms: total_decode_ms,
        prefill_ms_total: total_prefill_ms,
        total_tokens,
        fertility_tok_per_char: fertility,
        orphan_bpe_tokens,
        oracle_char_sim: 1.0,
        per_lang_oracle_sim: std::collections::BTreeMap::new(),
        identifier_exact_rate: 1.0,
        precision_score: 1.0,
        guard_rescue_rate: cum_guard.rescue_rate(),
        guard_rescue_ms: cum_guard.rescue_ms(),
        guard_rest_wins: cum_guard.rest_wins,
        quant_mode_a: cum_guard.quant_mode_a,
        outputs,
    })
}

fn score_arm_vs_oracle(arm: &mut ArmBenchResult, oracle: &ArmBenchResult) {
    let mut sims = Vec::new();
    for (k, ref_out) in &oracle.outputs {
        let cand_out = arm.outputs.get(k).map(|s| s.as_str()).unwrap_or("");
        let a: Vec<char> = ref_out.chars().take(64).collect();
        let b: Vec<char> = cand_out.chars().take(64).collect();
        let ed = edit_distance(&a, &b);
        let denom = a.len().max(b.len()).max(1);
        let sim = (1.0 - ed as f64 / denom as f64).clamp(0.0, 1.0);
        arm.per_lang_oracle_sim.insert(k.clone(), sim);
        sims.push(sim);
    }
    arm.oracle_char_sim = sims.iter().sum::<f64>() / sims.len().max(1) as f64;

    // Identifier exact preservation check across `en` and `id_en`
    let id_text = format!(
        "{}\n{}",
        arm.outputs.get("en").map(|s| s.as_str()).unwrap_or(""),
        arm.outputs.get("id_en").map(|s| s.as_str()).unwrap_or("")
    );
    let must_ids = ["microcontroller", "condition", "BL-2026-08814", "MSKU 7731025", "CI-2026-08001"];
    let hits = must_ids.iter().filter(|m| id_text.contains(**m)).count();
    arm.identifier_exact_rate = hits as f64 / must_ids.len() as f64;

    let orphan_health = if arm.orphan_bpe_tokens == 0 {
        1.0
    } else {
        (1.0 - arm.orphan_bpe_tokens as f64 / 3000.0).clamp(0.0, 0.9)
    };
    arm.precision_score = 0.50 * arm.oracle_char_sim + 0.25 * arm.identifier_exact_rate + 0.25 * orphan_health;
}

/// Real RX 6600 Head-to-Head Benchmark: Ours vs Matched AlphaEdge vs Full Untrimmed Oracle.
/// Run with:
///   LOGIS_TEST_MODELS=1 LOGIS_REQUIRE_GPU=1 LOGIS_BENCH_MODEL_KIND=q3|q35|4b
///   LOGIS_BENCH_OURS_DIR=<path> LOGIS_BENCH_ALPHAEDGE_DIR=<path> LOGIS_BENCH_OUT_JSON=<out.json>
#[test]
#[ignore = "실모델 AlphaEdge 대조 벤치마크 (LOGIS_TEST_MODELS=1)"]
fn alphaedge_head_to_head() -> R {
    if !enabled() {
        return Ok(());
    }
    let Some(dev) = gpu() else { return Ok(()) };
    let kind = std::env::var("LOGIS_BENCH_MODEL_KIND").unwrap_or_else(|_| "q3".into());
    let default_ref = match kind.as_str() {
        "q3" => models_dir().join("Qwen3-0.6B-Instruct-gguf"),
        "q35" => models_dir().join("Qwen3.5-2B-Instruct-gguf"),
        _ => logis_app::model::lang_llm::model_dir(logis_app::model::lang_llm::LANG_ENGINE),
    };
    let ref_dir = std::env::var("LOGIS_BENCH_REF_DIR").map(PathBuf::from).unwrap_or(default_ref);
    let ours_dir = std::env::var("LOGIS_BENCH_OURS_DIR").map(PathBuf::from).unwrap_or_else(|_| ref_dir.clone());
    let ae_dir = std::env::var("LOGIS_BENCH_ALPHAEDGE_DIR").map(PathBuf::from)?;

    let ours_trim = std::env::var("LOGIS_BENCH_OURS_TRIM").unwrap_or_else(|_| "compact".into());
    let ours_langs = std::env::var("LOGIS_BENCH_OURS_LANGS").unwrap_or_else(|_| "auto".into());
    let ours_guard = std::env::var("LOGIS_BENCH_OURS_GUARD").unwrap_or_else(|_| "off".into());
    let ours_active_k = std::env::var("LOGIS_BENCH_OURS_ACTIVE_K").ok();
    let ours_exec_mode = std::env::var("LOGIS_BENCH_OURS_EXEC_MODE").ok();

    let rt = tokio::runtime::Runtime::new()?;

    // 1. Reference full untrimmed oracle
    let oracle = run_bench_arm("ref_full", &kind, &ref_dir, "off", "all", "off", None, Some("full_untrimmed"), &dev, &rt)?;
    // 2. AlphaEdge matched model (runs natively as artifact_direct, all of its 16384/32768 rows)
    let mut ae = run_bench_arm("alphaedge", &kind, &ae_dir, "full", "all", "off", None, Some("artifact_direct"), &dev, &rt)?;
    // 3. Ours (improved model/runtime)
    let mut ours = run_bench_arm(
        "ours",
        &kind,
        &ours_dir,
        &ours_trim,
        &ours_langs,
        &ours_guard,
        ours_active_k.as_deref(),
        ours_exec_mode.as_deref(),
        &dev,
        &rt,
    )?;

    score_arm_vs_oracle(&mut ae, &oracle);
    score_arm_vs_oracle(&mut ours, &oracle);

    println!("\n========== [ALPHAEDGE HEAD-TO-HEAD: {kind}] ==========");
    println!(
        "  REF_FULL  : vocab={:6} | vram={:6.1}MB (delta {:6.1}MB) | {:6.2}ms/tok | total={:7.1}ms | fert={:.3} | orphans={}",
        oracle.vocab_size, oracle.vram_mb, oracle.vram_delta_mb, oracle.decode_ms_per_tok, oracle.total_gen_ms, oracle.fertility_tok_per_char, oracle.orphan_bpe_tokens
    );
    println!(
        "  ALPHAEDGE : vocab={:6} | vram={:6.1}MB (delta {:6.1}MB) | {:6.2}ms/tok | total={:7.1}ms | fert={:.3} | orphans={:4} | sim={:.4} | id={:.2} | prec={:.4}",
        ae.vocab_size, ae.vram_mb, ae.vram_delta_mb, ae.decode_ms_per_tok, ae.total_gen_ms, ae.fertility_tok_per_char, ae.orphan_bpe_tokens, ae.oracle_char_sim, ae.identifier_exact_rate, ae.precision_score
    );
    println!(
        "  OURS      : vocab={:6} (k={:.0}) | vram={:6.1}MB (delta {:6.1}MB) | {:6.2}ms/tok | total={:7.1}ms | fert={:.3} | orphans={:4} | sim={:.4} | id={:.2} | prec={:.4} | rescue={:.1}% ({:.1}ms, wins={})",
        ours.vocab_size, ours.active_k_mean, ours.vram_mb, ours.vram_delta_mb, ours.decode_ms_per_tok, ours.total_gen_ms, ours.fertility_tok_per_char, ours.orphan_bpe_tokens, ours.oracle_char_sim, ours.identifier_exact_rate, ours.precision_score, ours.guard_rescue_rate * 100.0, ours.guard_rescue_ms, ours.guard_rest_wins
    );

    let report = serde_json::json!({
        "model": kind,
        "oracle": oracle,
        "alphaedge": ae,
        "ours": ours,
        "beat_vram": ours.vram_mb < ae.vram_mb,
        "beat_ms_per_tok": ours.decode_ms_per_tok < ae.decode_ms_per_tok,
        "beat_total_gen_ms": ours.total_gen_ms < ae.total_gen_ms,
        "beat_precision": ours.precision_score > ae.precision_score,
    });
    if let Ok(out_path) = std::env::var("LOGIS_BENCH_OUT_JSON") {
        if let Some(p) = PathBuf::from(&out_path).parent() {
            let _ = std::fs::create_dir_all(p);
        }
        std::fs::write(&out_path, serde_json::to_string_pretty(&report)?)?;
        println!("  Saved report -> {out_path}");
    }
    finish_gpu_test();
    Ok(())
}

/// Real GPU verification for Granite-97m-r2 `MappedTable` E0 parity & VRAM reduction + GPU fused SwiGLU throughput.
#[test]
#[ignore = "실모델 임베딩 MappedTable + GPU SwiGLU 검증 (LOGIS_TEST_MODELS=1)"]
fn embedding_mapped_and_glu_bench() -> R {
    if !enabled() {
        return Ok(());
    }
    let Some(dev) = gpu() else { return Ok(()) };
    let embed_dir = models_dir().join("granite-embedding-97m-multilingual-r2");
    if !embed_dir.join("model.safetensors").exists() {
        anyhow::bail!("granite-embedding-97m-multilingual-r2 not found at {embed_dir:?}");
    }
    let samples: Vec<String> = bench_12lang_prompts().into_iter().map(|(_, _, p)| p.to_string()).collect();

    // 1. Resident baseline
    std::env::set_var("LOGIS_EMBED_TABLE", "resident");
    let f0 = free_mb(&dev);
    let m_res = logis_app::models::embedding::EmbeddingModel::new_with_device(&embed_dir, &dev)?;
    let _ = m_res.embed(&samples[0])?; // warmup
    let vram_res = f0 - free_mb(&dev);
    let batch_opts = logis_app::models::embedding::BatchOpts::default();
    let t0 = Instant::now();
    let v_res = m_res.embed_many(&samples, &batch_opts)?;
    let _ = dev.synchronize_all();
    let ms_res = t0.elapsed().as_secs_f64() * 1e3;
    drop(m_res);
    let _ = free_mb(&dev);

    // 2. MappedTable (default)
    std::env::set_var("LOGIS_EMBED_TABLE", "mapped");
    let f1 = free_mb(&dev);
    let m_map = logis_app::models::embedding::EmbeddingModel::new_with_device(&embed_dir, &dev)?;
    let _ = m_map.embed(&samples[0])?; // warmup
    let vram_map = f1 - free_mb(&dev);
    let t1 = Instant::now();
    let v_map = m_map.embed_many(&samples, &batch_opts)?;
    let _ = dev.synchronize_all();
    let ms_map = t1.elapsed().as_secs_f64() * 1e3;
    drop(m_map);

    let mut max_abs = 0.0f32;
    for (a, b) in v_res.iter().zip(&v_map) {
        for (x, y) in a.iter().zip(b) {
            max_abs = max_abs.max((x - y).abs());
        }
    }
    println!(
        "[EMBED-BENCH] resident: vram={vram_res:.1}MB, batch14={ms_res:.1}ms | mapped: vram={vram_map:.1}MB, batch14={ms_map:.1}ms | max_abs_err={max_abs:.2e}"
    );
    assert!(max_abs <= 1e-5, "MappedTable must match resident embeddings (E0/E1): max_abs={max_abs}");
    finish_gpu_test();
    Ok(())
}

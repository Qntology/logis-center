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
use tauri_app_lib::models::common::vocab_trim::{self, DocLangScope};

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
    if let Some(d) = tauri_app_lib::utils::new_gpu_device(0) {
        return Some(d);
    }
    println!("[TRIM-TEST] GPU 를 열 수 없어 건너뜁니다");
    None
}

fn finish_gpu_test() {
    if std::env::var("LOGIS_TEST_FAST_EXIT").map(|v| v == "1").unwrap_or(false) {
        tauri_app_lib::utils::device_utils::rocm_exit_now_if_used(0);
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
    Q3(tauri_app_lib::models::qwen3::generate::Qwen3GenerateModel),
    Q35(Box<tauri_app_lib::models::qwen3_5::generate::Qwen3_5GenerateModel>),
}

impl Gen {
    fn load(kind: &str, dev: &Device) -> anyhow::Result<Self> {
        Ok(match kind {
            "q3" => {
                let dir = models_dir().join("Qwen3-0.6B-Instruct-gguf");
                Self::Q3(tauri_app_lib::models::qwen3::generate::Qwen3GenerateModel::init_from_gguf(dir.to_str().unwrap(), Some(dev), Some(DType::BF16))?)
            }
            "q35" => {
                let f = models_dir().join("Qwen3.5-2B-Instruct-gguf").join("Qwen3.5-2B-Q8_0.gguf");
                Self::Q35(Box::new(tauri_app_lib::models::qwen3_5::generate::Qwen3_5GenerateModel::init_from_gguf(f.to_str().unwrap(), None, Some(dev))?))
            }
            _ => {
                let dir = tauri_app_lib::model::lang_llm::model_dir(tauri_app_lib::model::lang_llm::LANG_ENGINE);
                Self::Q35(Box::new(tauri_app_lib::model::lang_gguf::load_runtime(&dir, dev)?))
            }
        })
    }

    fn available(kind: &str) -> bool {
        match kind {
            "q3" => models_dir().join("Qwen3-0.6B-Instruct-gguf").join("Qwen3-0.6B-Q8_0.gguf").exists(),
            "q35" => models_dir().join("Qwen3.5-2B-Instruct-gguf").join("Qwen3.5-2B-Q8_0.gguf").exists(),
            _ => tauri_app_lib::model::lang_llm::is_ready(tauri_app_lib::model::lang_llm::LANG_ENGINE),
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
                tauri_app_lib::tokenizer::TokenizerModel::init(models_dir().join("Qwen3-0.6B-Instruct-gguf").to_str().unwrap())?.tokenizer
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
                tauri_app_lib::models::qwen3_5::model::IGNORE_STOP_SIGNAL_FOR_TESTS.store(true, std::sync::atomic::Ordering::Relaxed);
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
                let doc_lang = tauri_app_lib::utils::lang_utils::detect_document_language(text);
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
    g: &mut tauri_app_lib::models::qwen3_5::generate::Qwen3_5GenerateModel,
    img: &image::DynamicImage,
) -> anyhow::Result<(String, f64)> {
    use tauri_app_lib::openai_types::*;
    let (url, _guard) = tauri_app_lib::utils::img_utils::register_mem_image(img.clone());
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
    let load = |trim: &str| -> anyhow::Result<tauri_app_lib::models::qwen3_5::generate::Qwen3_5GenerateModel> {
        set_rule(trim, "any");
        tauri_app_lib::models::qwen3_5::generate::Qwen3_5GenerateModel::init_from_gguf(model.to_str().unwrap(), Some(mmproj.to_str().unwrap()), Some(&dev))
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
    use tauri_app_lib::model::lang_llm;
    let code = lang_llm::LANG_ENGINE;
    if lang_llm::is_ready(code) {
        println!("[TRIM-TEST] {} 준비됨: {:?}", lang_llm::repo_id(code), lang_llm::model_dir(code));
        return Ok(());
    }
    let state = tauri_app_lib::AppState {
        model: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
        store: std::sync::Arc::new(tokio::sync::Mutex::new(None)),
        cancellation_token: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    };
    let app = tauri_app_lib::build_headless_app(state)?;
    let handle = app.handle().clone();
    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        let phase = lang_llm::request_download(code, &tauri_app_lib::sink::tauri(&handle), "trim_fetch_4b");
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
    println!("[TRIM-RESULT] {} 준비 완료: {:?} · 실행용 {:.0} MB", lang_llm::repo_id(code), lang_llm::model_dir(code), tauri_app_lib::model::lang_gguf::runtime_bytes(&lang_llm::model_dir(code)) as f64 / 1e6);
    Ok(())
}

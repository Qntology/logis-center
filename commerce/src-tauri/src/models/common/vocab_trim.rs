//! 🌟 [VOCAB TRIM] 어휘 잘라 내기: 생성 모델(Qwen3 0.6B · Qwen3.5 2B · 4B)의 출력 투영에서 남길 행 고르기와 그 행만의 출력 투영.
//!
//! 입력 임베딩은 원본 전체 어휘 그대로 두고(프롬프트 토큰은 어떤 것이든 그대로), 매 디코드 스텝 행렬-벡터 곱의 대부분을 차지하는 출력 투영만
//! 남긴 K 행으로 곱합니다. 잘린 행은 어디에도 남기지 않습니다. 묶인 표라 입력 임베딩 행은 남긴 행의 출력 투영(장치)에서 꺼내고, 남긴 행 밖의
//! 토큰만 매핑에서 읽습니다 (`EmbedTable::forward_with_head` · `rows_f32_with_head`). 남길 행 (blueprint-sandbox `core/src/ai/trim.rs` 와 같은 규칙):
//!
//! - **공통 코어**: 추가 · 특수 토큰, 한 바이트 토큰 256 개(바이트 수준 BPE 의 바이트 폴백), 혼자서는 UTF-8 이 아닌 바이트 조각 전부,
//!   기호 · 단위 · 그리스 문자 토큰, ASCII 전부 (영어는 어느 문서에서나 전부 디코딩됩니다: JSON 키 · 영어 값 · 숫자 · 코드).
//! - **문서 언어의 문자 체계 전부**: 한글(ko) · 라틴 확장(de · fr · es · it · pt · nl · cs) · 가나(ja) · 아랍 문자(ar) · 한자(ja · zh).
//!   한자를 쓰지 않는 문서에 섞인 한자(한국어 문서의 한자 등, [`Langs::hanja`])는 도메인 글에 나온 것 + 병합 순 상위 [`CJK_COMMON`] 개.
//!   (일본어 · 중국어 문서도 상위 4,096 개만 남기면 blueprint 의 Zoro-1(같은 Qwen3.5 어휘)에서 일본어 한자 設 · 場 · 階 가 간체로 바뀌고
//!   중국어 여러 글자 토큰이 빠져 글자가 사라졌습니다.) 12 개 언어 밖(키릴 · 타이 · 데바나가리 문자 등, `other`)은
//!   그 언어의 문서이거나 글에 나왔을 때만 남깁니다.
//! - **글에 실제로 나온 문자 체계**: 판정 언어와 합칩니다 ([`langs_of_document`]). `detect_document_language` 는 라틴 언어를
//!   fr · de · es · it · pt · nl · cs 밖이면 `en` 으로 돌려주므로(폴란드어 ł · ś 등), 판정만 따르면 문서에 있는 글자를 잘라 냅니다.
//! - **도메인 글**: 내장 도메인 사전(`bias.json` · `full_bias.json` 의 모든 글)과 JSON · HTML 표 태그가 쓰는 토큰 전부.
//! - 남긴 수는 64 의 배수.
//!
//! 문서 언어는 `utils::lang_utils::detect_document_language` 가 정합니다 (현지 언어 + 영어 두 코드). 텍스트 문서는 생성 전에
//! (스케줄러의 `[DOC LANG]`), 이미지 문서는 첫 판독 글로 정합니다 ([`DocLangScope`] · [`observe_read`], 실행 흐름마다). 정하지 않았으면 문자 체계를 빼지 않습니다.
//!
//! 출력은 장치에서 전체 어휘 길이로 다시 펼칩니다(역 번호표 `index_select`): 남긴 행의 로짓은 그 자리에, 잘린 행은 −∞. 그래서 생성 루프의
//! 토큰 번호 기반 처리(끝 토큰 · `{` 강제 · 무시 목록 · 반복 벌점 · 선입견 벡터)는 그대로이고, 잘린 토큰은 절대 뽑히지 않습니다.
//! 로짓 수치는 슬롯의 원래 출력 투영과 같은 dtype 입니다 (Qwen3 · Qwen3.5 는 F32, QwenVL 의 `QLinear` 는 BF16 출력).
//! 샘플링(온도 > 0)은 전체 행으로 곱합니다 ([`refresh_for`]).

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Result};
use candle_core::quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul};
use candle_core::{Device, Module, Tensor};
use once_cell::sync::Lazy;
use rayon::prelude::*;

// 🌟 [PD-QUANT TRIM] 토큰 종류(kind_of) · 언어 묶음(Langs) · 규칙(TrimRule) · 남길 행 고르기(select_from) · CJK_COMMON · ALIGN 은
//    premium-discount-quant::trim 을 씁니다 (이 파일에서 옮겨 간 코드, 주석 · 줄바꿈을 빼면 토큰 단위로 같음). 여기에는 logis 쪽 연결만
//    남습니다: 환경 변수 · 문서 언어 판정 · 토크나이저 · 출력 투영(TrimmedHead) · 실행 흐름별 언어 범위(DocLangScope).
pub use premium_discount_quant::trim::{kind_of, select_from, Kind, Langs, TrimRule, ALIGN, CJK_COMMON};

/// 문서 글에서: `detect_document_language` 의 현지 언어 + 영어, 그리고 글에 실제로 나온 문자 체계 (예전 `Langs::of_document`)
pub fn langs_of_document(text: &str) -> Langs {
    Langs::of_doc_lang(&crate::utils::lang_utils::detect_document_language(text)).union(Langs::present(text))
}

/// `LOGIS_VOCAB_TRIM` (기본 `full`, 예전 `TrimRule::from_env`). 읽지 못한 값은 알리고 full 로 둡니다.
pub fn rule_from_env() -> TrimRule {
    match std::env::var(TRIM_ENV) {
        Ok(v) => TrimRule::parse(&v).unwrap_or_else(|e| {
            println!("[VOCAB-TRIM] {TRIM_ENV}={v:?}: {e} → full");
            TrimRule::Full
        }),
        Err(_) => TrimRule::Full,
    }
}

/// 규칙 (`full` · `off`)
pub const TRIM_ENV: &str = "LOGIS_VOCAB_TRIM";
/// 언어 (`auto` = 문서 언어, `all`, 또는 `ko,en` 처럼)
pub const LANGS_ENV: &str = "LOGIS_VOCAB_LANGS";

// ---------------------------------------------------------------------------------------------
// 토크나이저에서
// ---------------------------------------------------------------------------------------------

/// GPT-2 바이트 수준 BPE 의 글자 → 바이트 표
fn byte_decoder() -> &'static HashMap<char, u8> {
    static TABLE: Lazy<HashMap<char, u8>> = Lazy::new(|| {
        let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32).chain(0xA1..=0xAC).chain(0xAE..=0xFF).collect();
        let mut cs = bs.clone();
        let mut n = 0u32;
        for b in 0..256u32 {
            if !bs.contains(&b) {
                bs.push(b);
                cs.push(256 + n);
                n += 1;
            }
        }
        bs.into_iter().zip(cs).map(|(b, c)| (char::from_u32(c).unwrap(), b as u8)).collect()
    });
    &TABLE
}

/// 토큰의 원래 바이트 (바이트 수준 BPE 를 되돌림, 표에 없는 글자가 있으면 글자 그대로)
pub fn token_bytes(tok: &tokenizers::Tokenizer, id: u32) -> Vec<u8> {
    let Some(t) = tok.id_to_token(id) else { return Vec::new() };
    let table = byte_decoder();
    let mut out = Vec::with_capacity(t.len());
    for c in t.chars() {
        match table.get(&c) {
            Some(b) => out.push(*b),
            None => return t.into_bytes(),
        }
    }
    out
}

/// 도메인 글: 내장 도메인 사전의 모든 글(키 · 값)과 JSON · HTML 표 태그
pub fn corpus() -> &'static Vec<String> {
    static CORPUS: Lazy<Vec<String>> = Lazy::new(|| {
        fn walk(v: &serde_json::Value, out: &mut Vec<String>) {
            match v {
                serde_json::Value::String(s) => out.push(s.clone()),
                serde_json::Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
                serde_json::Value::Object(o) => o.iter().for_each(|(k, x)| {
                    out.push(k.clone());
                    walk(x, out)
                }),
                serde_json::Value::Number(n) => out.push(n.to_string()),
                _ => {}
            }
        }
        let mut out = Vec::new();
        walk(&crate::utils::bias_schema::BIAS_DICT, &mut out);
        for t in ["<table>", "</table>", "<tr>", "</tr>", "<td>", "</td>", "<th>", "</th>", "<thead>", "<tbody>", "<br>", "{\"", "\":", "\": \"", "\", \"", "\"}", "[{", "}]", "null", "true", "false"] {
            out.push(t.to_string());
        }
        out
    });
    &CORPUS
}

/// 토큰 종류와 도메인 글의 토큰 번호. 언어와 상관없으므로 토크나이저(어휘 크기)마다 한 번만 셉니다
/// (언어마다 다시 세면 2B 어휘에서 한 번에 5 ~ 6 s 라, N:N 검색 96 질의에서 12 번 72 s 가 들었습니다).
fn kinds_and_domain(tok: &tokenizers::Tokenizer, vocab: usize) -> Arc<(Vec<Kind>, BTreeSet<u32>)> {
    static BASE: Lazy<Mutex<HashMap<(usize, usize), Arc<(Vec<Kind>, BTreeSet<u32>)>>>> = Lazy::new(|| Mutex::new(HashMap::new()));
    let key = (vocab, tok.get_vocab_size(true));
    if let Some(b) = BASE.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return Arc::clone(b);
    }
    let t0 = std::time::Instant::now();
    let added: std::collections::HashSet<u32> = tok.get_added_tokens_decoder().keys().copied().collect();
    let kinds: Vec<Kind> = (0..vocab as u32).into_par_iter().map(|i| kind_of(&token_bytes(tok, i), added.contains(&i))).collect();
    let domain: BTreeSet<u32> = corpus()
        .par_iter()
        .filter_map(|text| tok.encode(text.as_str(), false).ok().map(|e| e.get_ids().to_vec()))
        .flatten_iter()
        .collect::<Vec<u32>>()
        .into_iter()
        .collect();
    println!("[VOCAB-TRIM] 어휘 {vocab}: 토큰 종류 · 도메인 글 토큰 {} 개 · {:.2} s (어휘마다 한 번)", domain.len(), t0.elapsed().as_secs_f64());
    let b = Arc::new((kinds, domain));
    BASE.lock().unwrap_or_else(|p| p.into_inner()).insert(key, Arc::clone(&b));
    b
}

/// 토크나이저 · 규칙 · 언어로 남길 행 (오름차순). 어휘 크기 · 규칙 · 언어마다 한 번 계산하고 기억합니다.
pub fn ids_for(tok: &tokenizers::Tokenizer, vocab: usize, rule: TrimRule, langs: Langs) -> Result<Arc<Vec<u32>>> {
    static CHOSEN: Lazy<Mutex<HashMap<(usize, usize, String), Arc<Vec<u32>>>>> = Lazy::new(|| Mutex::new(HashMap::new()));
    let key = (vocab, tok.get_vocab_size(true), format!("{}:{}", rule.label(), langs.label()));
    if let Some(ids) = CHOSEN.lock().unwrap_or_else(|p| p.into_inner()).get(&key) {
        return Ok(Arc::clone(ids));
    }
    let base = kinds_and_domain(tok, vocab);
    let t0 = std::time::Instant::now();
    let (ids, tiers) = select_from(&base.0, &base.1, rule, langs);
    let t: Vec<String> = tiers.iter().map(|(n, k)| format!("{n} {k}")).collect();
    println!("[VOCAB-TRIM] {} · {}: 어휘 {vocab} → {} 행 ({}) · {:.2} s", rule.label(), langs.label(), ids.len(), t.join(" · "), t0.elapsed().as_secs_f64());
    let ids = Arc::new(ids);
    CHOSEN.lock().unwrap_or_else(|p| p.into_inner()).insert(key, Arc::clone(&ids));
    Ok(ids)
}

// ---------------------------------------------------------------------------------------------
// 문서 언어 (실행 흐름마다)
// ---------------------------------------------------------------------------------------------
//
// 예전에는 프로세스 전체에 값 하나(`DOC_LANGS`)를 두고 범위를 열 때 덮어쓰고 닫을 때 되돌렸습니다. 검색(질의 언어 + 앱 언어)과
// 스케줄러의 문서 추출(그 문서의 언어)이 동시에 돌면 나중에 연 쪽이 값을 덮어, 다른 쪽의 생성이 남의 언어로 잘린 행을 썼습니다
// (예: 독일어 문서 추출이 범위를 연 사이 한국어 앱의 검색 답변 → 한글 행이 빠짐). 닫는 순서가 엇갈리면 이미 끝난 범위의
// 값이 되살아나기도 했습니다. 이제 범위는 연 실행 흐름(tokio 작업, 작업 밖이면 스레드)에 묶입니다:
// - 생성은 자기 흐름이 연 가장 안쪽 범위를 씁니다 (한 흐름 안에서 겹쳐 연 범위는 예전처럼 안쪽이 이김).
// - 자기 흐름이 연 범위가 없으면(다른 작업으로 넘겨진 생성) 열린 모든 흐름의 가장 안쪽 범위의 합집합을 씁니다. 행이 더 많을 뿐
//   어느 흐름의 글도 잘리지 않습니다 (off 에 더 가깝기만 함). 열린 범위가 없으면 아무 문자 체계도 빼지 않습니다.

/// 범위를 연 실행 흐름
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flow {
    Task(tokio::task::Id),
    Thread(std::thread::ThreadId),
}

fn flow() -> Flow {
    match tokio::task::try_id() {
        Some(id) => Flow::Task(id),
        None => Flow::Thread(std::thread::current().id()),
    }
}

/// 열린 범위 하나 (연 순서대로 쌓임)
struct ScopeEntry {
    id: u64,
    flow: Flow,
    langs: Langs,
    /// 이미지 문서가 아직 언어를 정하지 않았는가 ([`DocLangScope::image`] → [`observe_read`])
    pending: bool,
}

static SCOPES: Lazy<Mutex<Vec<ScopeEntry>>> = Lazy::new(|| Mutex::new(Vec::new()));
static NEXT_SCOPE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn scopes() -> std::sync::MutexGuard<'static, Vec<ScopeEntry>> {
    SCOPES.lock().unwrap_or_else(|p| p.into_inner())
}

/// 지금 흐름의 문서 언어 (위 규칙)
fn scoped_langs() -> Langs {
    let g = scopes();
    let me = flow();
    if let Some(e) = g.iter().rev().find(|e| e.flow == me) {
        return e.langs;
    }
    let mut seen: Vec<Flow> = Vec::new();
    let mut acc: Option<Langs> = None;
    for e in g.iter().rev() {
        if !seen.contains(&e.flow) {
            seen.push(e.flow);
            acc = Some(acc.map_or(e.langs, |a| a.union(e.langs)));
        }
    }
    acc.unwrap_or_else(Langs::any)
}

/// 문서 하나의 언어 범위. 연 실행 흐름에 묶이고 닫힐 때(드롭) 사라집니다. 문서 처리 함수의 맨 위에 두면
/// 그 문서가 끝난 뒤의 생성(검색 질의 등)은 다시 바깥 범위(없으면 아무 문자 체계도 빼지 않음)로 돌아갑니다.
pub struct DocLangScope {
    id: u64,
}

impl DocLangScope {
    /// 텍스트 맥락: 생성 전에 판정한 문서 언어(`detect_document_language` 의 현지 언어 + 영어)와 판정에 쓴 글에 나온 문자 체계
    pub fn text(doc_lang: &str, text: &str) -> Self {
        Self::open(Langs::of_doc_lang(doc_lang).union(Langs::present(text)), false)
    }

    /// 이미지 맥락: 언어를 미리 모르므로 아무 문자 체계도 빼지 않고 시작하고, 판독 글이 문자 체계를 가리키면 [`observe_read`] 가 정합니다.
    pub fn image() -> Self {
        Self::open(Langs::any(), true)
    }

    fn open(langs: Langs, pending: bool) -> Self {
        let id = NEXT_SCOPE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        scopes().push(ScopeEntry { id, flow: flow(), langs, pending });
        Self { id }
    }

    fn update(&self, f: impl FnOnce(&mut ScopeEntry)) {
        if let Some(e) = scopes().iter_mut().find(|e| e.id == self.id) {
            f(e);
        }
    }

    /// 같은 문서에서 언어를 다시 판정했을 때 (노이즈 제거 뒤 재확정)
    pub fn set(&self, doc_lang: &str, text: &str) {
        let langs = Langs::of_doc_lang(doc_lang).union(Langs::present(text));
        self.update(|e| {
            e.langs = langs;
            e.pending = false;
        });
    }

    /// 앱이 출력하라고 지시하는 언어를 더합니다 (검색 답변의 "Write in {LANG}" = 앱 언어 등). 언어 이름(`korean`)도 코드(`ko`)도 받습니다.
    pub fn also(&self, lang: &str) {
        let more = Langs::of_doc_lang(&crate::utils::bias_schema::lang_code_of(lang));
        self.update(|e| e.langs = e.langs.union(more));
    }
}

/// 생성 요청의 메시지 글 전부 (문자 체계만 보므로 JSON 그대로)
pub fn prompt_text(mes: &crate::openai_types::ChatCompletionParameters) -> String {
    serde_json::to_string(&mes.messages).unwrap_or_default()
}

impl Drop for DocLangScope {
    fn drop(&mut self) {
        scopes().retain(|e| e.id != self.id);
    }
}

/// 판독 글에서 값만 (JSON 이면 문자열 값만 모읍니다: 키 이름은 영어라 언어 판정을 흐립니다)
fn read_values(text: &str) -> String {
    fn walk(v: &serde_json::Value, out: &mut String) {
        match v {
            serde_json::Value::String(s) => {
                out.push_str(s);
                out.push('\n');
            }
            serde_json::Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            serde_json::Value::Object(m) => m.values().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    let t = text.trim();
    let body = t.find(|c| c == '{' || c == '[').map_or(t, |s| &t[s..]);
    let body = body.rfind(|c| c == '}' || c == ']').map_or(body, |e| &body[..=e]);
    match serde_json::from_str::<serde_json::Value>(body) {
        Ok(v) => {
            let mut out = String::new();
            walk(&v, &mut out);
            out
        }
        Err(_) => text.to_string(),
    }
}

/// 이미지 문서의 판독 글 하나를 봅니다. 아직 언어를 정하지 않았고, 값에 라틴 밖 문자 체계(한글 · 가나 · 한자 · 아랍 문자 등)가
/// 3 자 이상 나오면 `detect_document_language` 로 문서 언어(현지 언어 + 영어)를 정해 다음 생성부터 그 행만 남깁니다.
/// 라틴 문자만 나온 판독으로는 정하지 않습니다: 한국어 무역 서류도 첫 값(번호 · 영문 상호)은 라틴 문자뿐인 일이 흔해서,
/// 영어로 정하면 뒤의 한글 값을 쓸 수 없게 됩니다. 정했으면 그 언어를 돌려줍니다.
/// 정할 이미지 범위는 지금 흐름이 연 가장 안쪽의 언어 미정 범위이고, 지금 흐름에 없으면 열린 언어 미정 범위가 하나뿐일 때 그것입니다
/// (둘 이상이면 어느 문서의 판독인지 모르므로 정하지 않습니다: 아무 문자 체계도 빼지 않는 채로 둠).
pub fn observe_read(text: &str) -> Option<Langs> {
    let mut g = scopes();
    let me = flow();
    let at = match g.iter().rposition(|e| e.flow == me && e.pending) {
        Some(i) => i,
        None => {
            let pending: Vec<usize> = g.iter().enumerate().filter(|(_, e)| e.pending).map(|(i, _)| i).collect();
            match pending.as_slice() {
                [i] if !g.iter().any(|e| e.flow == me) => *i,
                _ => return None,
            }
        }
    };
    let values = read_values(text);
    let decisive = values.chars().filter(|&c| c.is_alphabetic() && !c.is_ascii() && !matches!(c as u32, 0x00C0..=0x024F | 0x1E00..=0x1EFF)).count();
    if decisive < 3 {
        return None;
    }
    let langs = langs_of_document(&values);
    g[at].langs = langs;
    g[at].pending = false;
    Some(langs)
}

/// 지금 쓸 언어: `LOGIS_VOCAB_LANGS` 가 `all`(12 개 언어) · `any` · 코드면 그것, 아니면(`auto`, 기본) 지금 흐름의 문서 언어
pub fn current_langs() -> Langs {
    match std::env::var(LANGS_ENV) {
        Ok(v) if !v.trim().is_empty() && !v.trim().eq_ignore_ascii_case("auto") => {
            Langs::from_codes(&v.split(',').collect::<Vec<_>>()).unwrap_or_else(Langs::any)
        }
        _ => scoped_langs(),
    }
}

// ---------------------------------------------------------------------------------------------
// 남긴 행만의 출력 투영
// ---------------------------------------------------------------------------------------------

/// 출력 투영 표(묶인 `token_embd.weight`)의 저장된 바이트를 빌려 줄 곳: 매핑된 GGUF 와 텐서 위치
#[derive(Clone)]
pub struct TableSource {
    pub map: Arc<memmap2::Mmap>,
    pub start: usize,
    pub rows: usize,
    pub cols: usize,
    pub dtype: GgmlDType,
}

impl TableSource {
    /// GGUF 머리(`content`)와 매핑에서 텐서 `name` 의 자리
    pub fn from_gguf(map: Arc<memmap2::Mmap>, content: &candle_core::quantized::gguf_file::Content, name: &str) -> Result<Self> {
        let info = content.tensor_infos.get(name).ok_or_else(|| anyhow!("GGUF 에 {name} 이 없습니다"))?;
        let (rows, cols) = info.shape.dims2()?;
        let dtype = info.ggml_dtype;
        if cols % dtype.block_size() != 0 {
            bail!("{name}: 열 {cols} 이 블록 {} 로 나뉘지 않습니다", dtype.block_size());
        }
        let start = (content.tensor_data_offset + info.offset) as usize;
        let row_bytes = cols / dtype.block_size() * dtype.type_size();
        if start + rows * row_bytes > map.len() {
            bail!("{name}: 파일이 잘렸습니다");
        }
        Ok(Self { map, start, rows, cols, dtype })
    }

    fn row_bytes(&self) -> usize {
        self.cols / self.dtype.block_size() * self.dtype.type_size()
    }

    /// `ids` 번 행만 모은 양자화 텐서 (그 순서대로)
    pub fn gather(&self, ids: &[u32], device: &Device) -> Result<candle_core::quantized::QTensor> {
        let rb = self.row_bytes();
        let mut words = vec![0u64; (ids.len() * rb).div_ceil(8)];
        let total = ids.len() * rb;
        let contiguous = ids.first().is_some_and(|&f| ids.iter().enumerate().all(|(k, &r)| r as usize == f as usize + k));
        if contiguous {
            // 이어진 행(전체 행 등)은 한 번에 복사합니다
            let f = ids[0] as usize;
            if f + ids.len() > self.rows {
                bail!("행 {} 이 표 크기 {} 를 넘습니다", f + ids.len() - 1, self.rows);
            }
            let (_, raw, _) = unsafe { words.align_to_mut::<u8>() };
            raw[..total].copy_from_slice(&self.map[self.start + f * rb..self.start + f * rb + total]);
        } else {
            let (_, raw, _) = unsafe { words.align_to_mut::<u8>() };
            for (k, &r) in ids.iter().enumerate() {
                let r = r as usize;
                if r >= self.rows {
                    bail!("행 {r} 이 표 크기 {} 를 넘습니다", self.rows);
                }
                raw[k * rb..(k + 1) * rb].copy_from_slice(&self.map[self.start + r * rb..self.start + (r + 1) * rb]);
            }
        }
        let (_, raw, _) = unsafe { words.align_to::<u8>() };
        Ok(qtensor_from_ggml(self.dtype, &raw[..total], vec![ids.len(), self.cols], device)?)
    }
}

/// 남긴 행만의 출력 투영. 로짓은 전체 어휘 길이로 펼쳐 CPU 에 돌려줍니다 (잘린 행은 −∞).
/// 복제는 장치의 행렬을 함께 씁니다 (`QMatMul` 의 Arc).
#[derive(Clone)]
pub struct TrimmedHead {
    w: QMatMul,
    pub kept: Arc<Vec<u32>>,
    pub vocab: usize,
    /// 남긴 행을 고른 규칙 · 언어 (같으면 다시 만들지 않음)
    pub label: String,
    /// 장치의 역표 [vocab] (u32): 어휘 번호 → 남긴 K 행 안의 자리, 잘린 번호는 K (덧붙인 −∞ 칸)
    inv: Tensor,
    /// 같은 역표의 호스트 사본 (입력 임베딩을 이 행에서 꺼낼 때, [`Self::embed`])
    host_inv: Arc<Vec<u32>>,
    /// 덧붙일 −∞ 한 칸 [1, 1]
    pad: Tensor,
}

impl TrimmedHead {
    pub fn build(src: &TableSource, kept: Arc<Vec<u32>>, label: String, device: &Device) -> Result<Self> {
        let t0 = std::time::Instant::now();
        let q = src.gather(&kept, device)?;
        let mut inv = vec![kept.len() as u32; src.rows];
        for (i, &id) in kept.iter().enumerate() {
            inv[id as usize] = i as u32;
        }
        let host_inv = Arc::new(inv.clone());
        let inv = Tensor::from_vec(inv, src.rows, device)?;
        let pad = Tensor::full(f32::NEG_INFINITY, (1, 1), device)?;
        let mb = (kept.len() * src.row_bytes()) as f64 / 1e6;
        println!("[VOCAB-TRIM] 출력 투영 {} → {} 행 ({mb:.0} MB, {label}) · {:.0} ms", src.rows, kept.len(), t0.elapsed().as_secs_f64() * 1e3);
        Ok(Self { w: QMatMul::from_qtensor(q)?, kept, vocab: src.rows, label, inv, host_inv, pad })
    }

    /// 입력 임베딩 행: 묶인 표라 남긴 행의 출력 투영과 같은 바이트입니다. `ids` 가 모두 남긴 행에 있을 때만 (아니면 `None`) 장치의
    /// 양자화 임베딩(`qembed`)으로 꺼내므로, 표 전체를 장치에 둔 off 의 입력 임베딩과 비트까지 같습니다. 프롬프트에 나온 문자 체계는
    /// 남긴 행에 들어가므로([`refresh_for`]) 프롬프트 토큰은 거의 언제나 여기서 나옵니다. 매핑에서 행을 읽으면 프리필마다 작업 집합에서
    /// 밀려난 표 페이지를 다시 들여와(0.6B 목록 추출 프리필 토큰당 +0.2 ms) 느렸습니다.
    pub fn embed(&self, ids: &[u32], device: &Device) -> Option<candle_core::Result<Tensor>> {
        let rows: Option<Vec<u32>> = ids.iter().map(|&id| self.row_of(id)).collect();
        rows.map(|r| self.embed_rows(&r, device))
    }

    /// 어휘 번호 `id` 의 남긴 행 자리 (잘린 번호면 `None`)
    pub fn row_of(&self, id: u32) -> Option<u32> {
        self.host_inv.get(id as usize).copied().filter(|&j| (j as usize) < self.kept.len())
    }

    /// 남긴 행 자리 `rows` 의 입력 임베딩 [n, hidden] F32 (장치의 `qembed`)
    pub fn embed_rows(&self, rows: &[u32], device: &Device) -> candle_core::Result<Tensor> {
        self.w.embedding(&Tensor::from_vec(rows.to_vec(), rows.len(), device)?)
    }

    /// `h` `[.., hidden]` (F32) → `[.., vocab]` F32, `h` 와 같은 장치. 남긴 행 밖은 −∞.
    /// 펼치기는 장치의 모으기 하나입니다: 남긴 K 행의 로짓 뒤에 −∞ 한 칸을 붙이고 역표로 `index_select` (출력 원소마다 스레드 하나).
    /// (CPU 에서 어휘 길이의 배열을 채우면 dev 빌드 0.6B 디코드가 전체 어휘보다 1 ~ 8 % 느렸고, 포크의 `index_add` 는 이 모양에서
    /// 스레드 하나가 K 행을 모두 돌아 0.6B +138 % · 2B +129 % 였습니다.)
    pub fn logits(&self, h: &Tensor) -> candle_core::Result<Tensor> {
        let mut dims = h.dims().to_vec();
        let k = self.w.forward(h)?.to_dtype(candle_core::DType::F32)?;
        if self.kept.len() == self.vocab {
            // 전체 행(번호 순)은 펼칠 것이 없습니다
            return Ok(k);
        }
        let n = k.elem_count() / self.kept.len().max(1);
        let k = k.reshape((n, self.kept.len()))?;
        let pad = if n == 1 { self.pad.clone() } else { self.pad.broadcast_as((n, 1))?.contiguous()? };
        let full = Tensor::cat(&[&k, &pad], 1)?.index_select(&self.inv, 1)?;
        if let Some(last) = dims.last_mut() {
            *last = self.vocab;
        }
        full.reshape(dims)
    }
}

/// 잘라 내기를 켤 수 있으면(규칙이 `off` 가 아니고 GPU · 매핑 · Q8_0 류 블록 표) 입력 임베딩 · 출력 투영 표의 자리.
/// 이때 입력 임베딩은 이 매핑에서 쓰는 행만 읽고(장치 0), 출력 투영은 남긴 행만 장치에 둡니다.
pub fn mapped_table(device: &Device, map: Option<&Arc<memmap2::Mmap>>, content: Option<&candle_core::quantized::gguf_file::Content>) -> Option<TableSource> {
    if rule_from_env() == TrimRule::Off || !(device.is_cuda() || device.is_rocm()) {
        return None;
    }
    let (map, content) = (map?, content?);
    // 묶이지 않은 출력 투영(`output.weight`)이 따로 있으면 남긴 행을 임베딩 표에서 뽑을 수 없어 자르지 않습니다
    // (Qwen3 0.6B · Qwen3.5 2B · 4B 는 모두 묶여 있습니다).
    if content.tensor_infos.contains_key("output.weight") {
        println!("[VOCAB-TRIM] 출력 투영이 임베딩 표와 묶여 있지 않아(output.weight) 자르지 않습니다.");
        return None;
    }
    match TableSource::from_gguf(Arc::clone(map), content, "token_embd.weight") {
        Ok(src) if src.dtype.block_size() > 1 => Some(src),
        Ok(_) => None,
        Err(e) => {
            println!("[VOCAB-TRIM] 출력 투영 표를 매핑하지 못해 자르지 않습니다: {e}");
            None
        }
    }
}

/// 어휘 길이의 의미 선입견 벡터(로짓에서 빼는 값)에서 잘린 행 자리를 +∞ 로 정합니다. 빼면 그 토큰은 −∞ 라, 잘린 토큰은 선입견을
/// 거쳐서도 절대 뽑히지 않습니다 (잘라 내기의 약속을 이 벡터에도 직접 걸어 둠). 선입견 블록은 잘린 행 자리를 남긴 행 하나로 채워
/// 계산하므로(`EmbedTable::rows_f32_with_head`) 그 자리 값은 의미가 없는데, 지금은 잘린 토큰의 로짓이 이미 −∞ 라 쓰이지 않을 뿐이었습니다.
/// 나중에 선입견을 빼기 전에 로짓을 고치는 코드가 생겨도 채움값이 새어 나가지 않습니다. 남긴 행의 값은 그대로입니다.
pub fn exclude_dropped(head: Option<&TrimmedHead>, prejudice: Tensor) -> candle_core::Result<Tensor> {
    let Some(h) = head else { return Ok(prejudice) };
    if h.kept.len() == h.vocab || prejudice.elem_count() != h.vocab {
        return Ok(prejudice);
    }
    let kept: Vec<u8> = h.host_inv.iter().map(|&j| ((j as usize) < h.kept.len()) as u8).collect();
    let mask = Tensor::from_vec(kept, prejudice.shape(), prejudice.device())?;
    let inf = Tensor::full(f32::INFINITY, prejudice.shape(), prejudice.device())?.to_dtype(prejudice.dtype())?;
    mask.where_cond(&prejudice, &inf)
}

/// 오름차순 `a` 가 오름차순 `b` 안에 모두 있는가
fn sorted_subset(a: &[u32], b: &[u32]) -> bool {
    let mut j = 0;
    for &x in a {
        while j < b.len() && b[j] < x {
            j += 1;
        }
        if j == b.len() || b[j] != x {
            return false;
        }
        j += 1;
    }
    true
}

/// 전체 행 출력 투영의 이름표
pub fn all_rows_label() -> String {
    format!("{}:all-rows", rule_from_env().label())
}

/// 전체 행 출력 투영 (off 와 같은 로짓). 샘플링과, 생성기가 언어를 정하기 전에 불린 계산에 씁니다.
/// 모델을 올릴 때 미리 문서 언어의 행을 모으지 않습니다: 첫 생성이 샘플링이면 곧바로 전체 행으로 바꿔 모으기만 두 번이 되고,
/// 탐욕 생성이면 [`refresh_for`] 가 그때 그 언어로 모읍니다.
pub fn all_rows(src: &TableSource, device: &Device) -> Result<TrimmedHead> {
    TrimmedHead::build(src, Arc::new((0..src.rows as u32).collect()), all_rows_label(), device)
}

/// 지금의 문서 언어로 남긴 행을 맞춥니다 (규칙 · 언어가 바뀌었으면 다시 모음). 규칙이 `off` 면 비웁니다.
pub fn refresh(slot: &mut Option<TrimmedHead>, src: &TableSource, tok: &tokenizers::Tokenizer, device: &Device) -> Result<()> {
    refresh_for(slot, src, tok, device, "", false)
}

/// [`refresh`] + 이번 생성의 프롬프트 글에 나온 문자 체계. 읽기는 다국어 전부라 프롬프트(질의 · 검색된 기록 · 문서)에 어떤 글자든
/// 올 수 있고, 모델은 그 값을 그대로 베껴 씁니다 (예: 한국어 질의에 일본 쇼핑몰 기록 → 상품 이름은 가나). 그래서 문서 언어 + 영어에
/// 프롬프트에 나온 문자 체계를 더해야 베낀 값이 잘리지 않습니다. `LOGIS_VOCAB_LANGS` 로 언어를 고정했으면 더하지 않습니다.
///
/// `sampling` (온도 > 0) 이면 전체 행으로 곱합니다. 잘린 행을 −∞ 로 두면 남은 토큰의 확률이 다시 정규화되어, top-p 의 누적 확률 경계에
/// 걸린 후보가 들어가거나 빠지고 같은 난수로도 다른 토큰을 뽑습니다 (N:N 검색에서 0.6B 가 `"percent_total": "20.0"` 대신 `"20.5"`).
/// 탐욕 디코딩은 off 가 고르는 토큰이 남긴 행에 있으면 결과가 같으므로 잘라 냅니다.
pub fn refresh_for(slot: &mut Option<TrimmedHead>, src: &TableSource, tok: &tokenizers::Tokenizer, device: &Device, prompt: &str, sampling: bool) -> Result<()> {
    let rule = rule_from_env();
    if rule == TrimRule::Off {
        *slot = None;
        return Ok(());
    }
    if sampling {
        if slot.as_ref().is_some_and(|h| h.label == all_rows_label()) {
            return Ok(());
        }
        *slot = None;
        *slot = Some(all_rows(src, device)?);
        return Ok(());
    }
    let fixed = std::env::var(LANGS_ENV).is_ok_and(|v| !v.trim().is_empty() && !v.trim().eq_ignore_ascii_case("auto"));
    let langs = if fixed { current_langs() } else { current_langs().union(Langs::present(prompt)) };
    let label = format!("{}:{}", rule.label(), langs.label());
    if slot.as_ref().is_some_and(|h| h.label == label) {
        return Ok(());
    }
    let ids = ids_for(tok, src.rows, rule, langs)?;
    // 지금 올린 행이 이번에 남길 행을 모두 담고 많아야 10 % 더 크면 그대로 씁니다. 같은 문서 안에서 프롬프트의 문자 체계에 따라
    // 이름표가 오가며(이미지 판독의 hangul+kana+cjk ↔ kana+cjk) 쪽마다 8 ~ 10 번(번에 0.3 s) 다시 모으던 것을 없앱니다.
    // 행이 더 많으면 off 에 더 가깝기만 합니다: off 가 고르는 토큰이 작은 쪽에 있으면 큰 쪽에도 있습니다.
    if let Some(h) = slot.as_ref() {
        if h.kept.len() <= ids.len() + ids.len() / 10 && sorted_subset(&ids, &h.kept) {
            return Ok(());
        }
    }
    *slot = None;
    *slot = Some(TrimmedHead::build(src, ids, label, device)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_and_languages() {
        assert_eq!(kind_of(b"a", false), Kind::Byte);
        assert_eq!(kind_of(&[0xEC, 0x9E], false), Kind::Partial);
        assert_eq!(kind_of(" ORDER".as_bytes(), false), Kind::Ascii);
        assert!(matches!(kind_of("주문".as_bytes(), false), Kind::Script { hangul: true, .. }));
        assert_eq!(kind_of("заказ".as_bytes(), false), Kind::Other);
        assert_eq!(Langs::of_doc_lang("ko"), Langs::from_codes(&["ko", "en"]).unwrap());
        let ru = Langs::of_doc_lang("ru");
        assert!(ru.other && ru.latin && !ru.hangul && !ru.cjk, "12 개 밖 언어는 라틴 확장 + 그 밖 문자: {ru:?}");
        assert_eq!(langs_of_document("주문 번호 20231"), Langs::from_codes(&["ko"]).unwrap());
        // 판정은 en(whatlang 의 폴란드어 → en)이어도 글의 라틴 확장 글자는 남김
        let pl = langs_of_document("Zamówienie nr 20231, łączna wartość");
        assert!(pl.latin && !pl.hangul, "{pl:?}");
        assert_eq!(Langs::from_codes(&["all"]).unwrap(), Langs::all());
        let ko_hanja = langs_of_document("급수 펌프(給水) 압력 주문");
        assert!(ko_hanja.hangul && ko_hanja.hanja && !ko_hanja.cjk, "{ko_hanja:?}");
        let zh = langs_of_document("给水泵压力");
        assert!(zh.cjk && !zh.hanja, "{zh:?}");
        // keeps_script(Kind::Other) 는 premium-discount-quant 안의 비공개 함수라 그쪽 kinds_and_languages 시험이 봅니다.
    }

    #[test]
    fn image_reads_decide_on_non_latin_values_only() {
        let _scope = DocLangScope::image();
        assert_eq!(current_langs_raw(), Langs::any());
        assert_eq!(observe_read(r#"{"doc_type": "commercial_invoice", "number": "INV-2023-001"}"#), None, "라틴 값만으로는 정하지 않음");
        let got = observe_read(r#"{"seller": "주식회사 한빛", "number": "INV-2023-001"}"#).expect("한글 값으로 정함");
        assert_eq!(got, Langs::from_codes(&["ko", "en"]).unwrap());
        assert_eq!(observe_read(r#"{"seller": "株式会社"}"#), None, "한 번 정하면 그대로");
        let id = _scope.id;
        drop(_scope);
        assert!(!scopes().iter().any(|e| e.id == id), "문서가 끝나면 그 범위는 사라짐");
    }

    fn current_langs_raw() -> Langs {
        scoped_langs()
    }

    /// 두 흐름(검색 · 문서 추출)이 동시에 범위를 열어도 각자의 언어를 쓰고, 범위가 없는 흐름은 둘의 합집합을 씁니다
    #[test]
    fn concurrent_flows_keep_their_own_language() {
        let rt = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build().unwrap();
        rt.block_on(async {
            let opened = Arc::new(tokio::sync::Barrier::new(3));
            let checked = Arc::new(tokio::sync::Barrier::new(3));
            let run = |lang: &'static str, text: &'static str| {
                let (opened, checked) = (Arc::clone(&opened), Arc::clone(&checked));
                tokio::spawn(async move {
                    let scope = DocLangScope::text(lang, text);
                    opened.wait().await;
                    let mine = current_langs_raw();
                    checked.wait().await;
                    let id = scope.id;
                    drop(scope);
                    (mine, id)
                })
            };
            let ko = run("ko", "주문 번호");
            let de = run("de", "Bestellung Größe");
            opened.wait().await;
            let outside = current_langs_raw();
            checked.wait().await;
            let ((ko, ko_id), (de, de_id)) = (ko.await.unwrap(), de.await.unwrap());
            assert!(ko.hangul && !ko.latin, "검색 흐름은 한국어: {ko:?}");
            assert!(de.latin && !de.hangul, "추출 흐름은 독일어: {de:?}");
            assert!(outside.hangul && outside.latin, "범위 없는 흐름은 합집합: {outside:?}");
            assert!(!scopes().iter().any(|e| e.id == ko_id || e.id == de_id), "닫힌 범위는 사라짐");
        });
    }

    #[test]
    fn selection_keeps_bytes_scripts_and_ascii_and_drops_other_scripts() {
        let pieces: Vec<Vec<u8>> = vec![b"<s>".to_vec(), b"a".to_vec(), vec![0xEC, 0x9E], b" ORDER".to_vec(), b" the".to_vec(), "주문".as_bytes().to_vec(), "заказ".as_bytes().to_vec(), "Größe".as_bytes().to_vec()];
        let kinds: Vec<Kind> = pieces.iter().enumerate().map(|(i, p)| kind_of(p, i == 0)).collect();
        let (ids, tiers) = select_from(&kinds, &BTreeSet::new(), TrimRule::Full, Langs::from_codes(&["ko", "en"]).unwrap());
        let before_pad: usize = tiers.iter().filter(|(n, _)| *n != "pad").map(|(_, k)| k).sum();
        assert_eq!(before_pad, 6, "ko,en: 특수 · 바이트 · 조각 · ASCII 둘 · 한글 (키릴 · 라틴 확장 빠짐) {tiers:?}");
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
        assert!(sorted_subset(&[1, 3], &[0, 1, 2, 3]) && sorted_subset(&[], &[5]) && !sorted_subset(&[1, 4], &[0, 1, 2, 3]) && !sorted_subset(&[2], &[]));
        assert_eq!(TrimRule::parse("full").unwrap(), TrimRule::Full);
        assert_eq!(TrimRule::parse("off").unwrap(), TrimRule::Off);
        assert!(TrimRule::parse("compact").is_err(), "예산을 둔 규칙은 없습니다 (영어 전부 디코딩)");
    }

    #[test]
    fn trimmed_logits_spread_back_to_the_vocabulary_with_minus_infinity() -> Result<()> {
        // 어휘 4 × 열 32 Q8_0 표에서 1 · 3 행만
        let dev = Device::Cpu;
        let w: Vec<f32> = (0..4 * 32).map(|i| ((i as f32) * 0.37).sin()).collect();
        let q = candle_core::quantized::QTensor::quantize(&Tensor::from_vec(w, (4, 32), &dev)?, GgmlDType::Q8_0)?;
        let bytes = q.data()?.to_vec();
        let dir = std::env::temp_dir().join(format!("logis-vt-{}", std::process::id()));
        std::fs::write(&dir, &bytes)?;
        let map = Arc::new(unsafe { memmap2::Mmap::map(&std::fs::File::open(&dir)?)? });
        let src = TableSource { map, start: 0, rows: 4, cols: 32, dtype: GgmlDType::Q8_0 };
        let head = TrimmedHead::build(&src, Arc::new(vec![1, 3]), "t".into(), &dev)?;
        let h = Tensor::from_vec((0..32).map(|i| (i as f32) * 0.01).collect::<Vec<f32>>(), (1, 32), &dev)?;
        let got: Vec<f32> = head.logits(&h)?.flatten_all()?.to_vec1()?;
        // 입력 임베딩: 남긴 행이면 표 전체의 임베딩과 비트까지 같고, 잘린 행이 섞이면 꺼내지 않음
        let want: Vec<f32> = src.gather(&[3, 1], &dev)?.dequantize(&dev)?.flatten_all()?.to_vec1()?;
        let emb: Vec<f32> = head.embed(&[3, 1], &dev).expect("남긴 행")?.flatten_all()?.to_vec1()?;
        assert_eq!(emb, want);
        assert!(head.embed(&[1, 2], &dev).is_none() && head.embed(&[9], &dev).is_none());
        // 남긴 행 · 잘린 행이 섞인 입력: 남긴 행은 투영에서, 잘린 행은 매핑에서 꺼내 원래 순서로 → 매핑만 쓴 것과 같음
        let table = crate::models::qwen3_5::model::EmbedTable::Mapped { src: src.clone(), device: dev.clone() };
        let ids = Tensor::new(&[[3u32, 2, 1, 0]], &dev)?;
        let mixed: Vec<f32> = table.forward_with_head(&ids, Some(&head))?.flatten_all()?.to_vec1()?;
        let plain: Vec<f32> = table.forward(&ids)?.flatten_all()?.to_vec1()?;
        assert_eq!(mixed, plain);
        // 어휘 블록(의미 선입견): 남긴 행은 같은 값, 잘린 행 자리는 남긴 행 0 번(= 어휘 1)으로 채움 (노름 0 → NaN 방지)
        let via_head = table.rows_f32_with_head(0, 4, Some(&head))?.to_vec2::<f32>()?;
        let mapped = table.rows_f32(0, 4)?.to_vec2::<f32>()?;
        assert_eq!((&via_head[1], &via_head[3]), (&mapped[1], &mapped[3]));
        assert_eq!((&via_head[0], &via_head[2]), (&mapped[1], &mapped[1]));
        // 선입견의 잘린 행 자리는 +∞: 그 자리 로짓이 (앞으로 어떤 코드로) 유한값이 되어 있어도 빼면 −∞, 남긴 행은 그대로
        let prej = exclude_dropped(Some(&head), Tensor::new(&[0.5f32, 0.25, 0.75, 1.0], &dev)?)?;
        let out: Vec<f32> = Tensor::new(&[5.0f32, 2.0, f32::NEG_INFINITY, 3.0], &dev)?.broadcast_sub(&prej)?.to_vec1()?;
        assert!(out[0] == f32::NEG_INFINITY && out[2] == f32::NEG_INFINITY, "{out:?}");
        assert_eq!((out[1], out[3]), (1.75, 2.0));
        let every = TrimmedHead::build(&src, Arc::new((0..4).collect()), "every".into(), &dev)?;
        assert_eq!(exclude_dropped(Some(&every), Tensor::new(&[0.5f32, 0.25, 0.75, 1.0], &dev)?)?.to_vec1::<f32>()?, vec![0.5, 0.25, 0.75, 1.0], "전체 행이면 그대로");
        let full: Vec<f32> = QMatMul::from_qtensor(q)?.forward(&h)?.flatten_all()?.to_vec1()?;
        assert_eq!(got.len(), 4);
        assert!(got[0].is_infinite() && got[2].is_infinite());
        assert!((got[1] - full[1]).abs() < 1e-5 && (got[3] - full[3]).abs() < 1e-5);
        // 전체 행(이어진 행을 한 번에 복사, 펼치기 없음)은 원래 투영과 비트까지 같음
        let all = TrimmedHead::build(&src, Arc::new((0..4).collect()), "all".into(), &dev)?;
        let got_all: Vec<f32> = all.logits(&h)?.flatten_all()?.to_vec1()?;
        assert_eq!(got_all, full);
        // 이어진 일부 행도 한 번에 복사
        let mid = TrimmedHead::build(&src, Arc::new(vec![1, 2]), "mid".into(), &dev)?;
        let got_mid: Vec<f32> = mid.logits(&h)?.flatten_all()?.to_vec1()?;
        assert_eq!((got_mid[1], got_mid[2]), (full[1], full[2]));
        assert!(got_mid[0].is_infinite() && got_mid[3].is_infinite());
        drop(head);
        let _ = std::fs::remove_file(&dir);
        Ok(())
    }
}

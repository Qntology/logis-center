//! 질의 글의 언어 판정 · 따옴표 구간 · 앱 언어 코드 (whatlang + 도메인 사전, Stanza 모델 없이). 예전 stanza.rs 에서 옮김.

use std::collections::{HashMap, HashSet};

use once_cell::sync::Lazy;
use serde_json::Value;

/// 앱 언어 이름 · ISO 코드 → 변환본 언어 코드. 12개 언어가 아니면 None.
pub fn lang_code(language: &str) -> Option<&'static str> {
    let l = language.trim().to_lowercase();
    Some(match l.as_str() {
        "korean" | "ko" | "kor" => "ko",
        "english" | "en" | "eng" => "en",
        "japanese" | "ja" | "jpn" => "ja",
        "chinese" | "zh" | "zh-hans" | "zh-cn" | "cmn" | "zh-hant" | "zh-tw" | "zh-hk" => "zh-hans",
        "french" | "fr" | "fra" => "fr",
        "german" | "de" | "deu" => "de",
        "spanish" | "es" | "spa" => "es",
        "italian" | "it" | "ita" => "it",
        "portuguese" | "pt" | "por" => "pt",
        "dutch" | "nl" | "nld" => "nl",
        "arabic" | "ar" | "ara" => "ar",
        "czech" | "cs" | "ces" => "cs",
        _ => return None,
    })
}

/// 라틴 문자 변환본 언어 (whatlang 후보를 이 여덟으로 제한)
pub const LATIN_LANGS: [(whatlang::Lang, &str); 8] = [
    (whatlang::Lang::Eng, "en"),
    (whatlang::Lang::Deu, "de"),
    (whatlang::Lang::Spa, "es"),
    (whatlang::Lang::Fra, "fr"),
    (whatlang::Lang::Ita, "it"),
    (whatlang::Lang::Por, "pt"),
    (whatlang::Lang::Nld, "nl"),
    (whatlang::Lang::Ces, "cs"),
];

pub static LATIN_DETECTOR: Lazy<whatlang::Detector> =
    Lazy::new(|| whatlang::Detector::with_allowlist(LATIN_LANGS.iter().map(|(l, _)| *l).collect()));

/// 도메인 사전 언어 구역(필드 라벨 · 뜻 · 편견 문구)의 낱말 → 그 낱말이 든 라틴 언어 구역들.
/// 8개 라틴 언어 구역이 모두 있는 사전(배포 사전 · full-bias)에서만 만듭니다. 개발 사전(ko · en 구역만)에서는 None 이라 투표하지 않습니다.
pub static LATIN_VOCAB: Lazy<Option<HashMap<String, Vec<&'static str>>>> = Lazy::new(|| {
    let dict: &Value = &crate::bias_schema::BIAS_DICT;
    let mut map: HashMap<String, Vec<&'static str>> = HashMap::new();
    for (_, code) in LATIN_LANGS.iter() {
        let mut words: HashSet<String> = HashSet::new();
        collect_vocab(dict.get(*code)?, &mut words);
        for w in words {
            map.entry(w).or_default().push(*code);
        }
    }
    Some(map)
});

/// 사전 투표 낱말: 세 글자 이상, 숫자 · 밑줄 없는 낱말 (소문자)
pub fn vocab_words(s: &str) -> impl Iterator<Item = String> + '_ {
    s.split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| w.chars().count() >= 3 && !w.chars().any(|c| c.is_numeric() || c == '_'))
        .map(|w| w.to_lowercase())
}

pub fn collect_vocab(v: &Value, out: &mut HashSet<String>) {
    match v {
        Value::String(s) => out.extend(vocab_words(s)),
        Value::Array(a) => a.iter().for_each(|x| collect_vocab(x, out)),
        Value::Object(o) => logis_schema::json_order::sorted_values(o).for_each(|x| collect_vocab(x, out)),
        _ => {}
    }
}

/// 라틴 문자 질의의 사전 투표: 질의 낱말이 든 언어 구역마다 1 / (그 낱말이 든 구역 수)를 더해, 1위 언어의 낱말이 둘 이상이고
/// 2위보다 1 이상 앞설 때만 그 언어입니다. whatlang 은 짧은 질의에서 cs → pt · nl → fr · de → en · fr → es 로 자주 틀렸습니다
/// (v2 질의 라틴 8개 언어 184개 중 16개). 투표가 그중 9개를 바로잡고 새로 틀린 것은 없었습니다.
///
/// 이름 줄(문장 처음이 아닌 곳에서 대문자로 시작하는 낱말이 둘 이상 이어진 것: 고객 Felix Wagner · 쇼핑몰 Copperleaf Goods)은 세지 않습니다.
/// 사전 구역에는 예시 이름도 들어 있어 'wagner' 가 독일어 표를 받았습니다 (productos que pidió Felix Wagner → whatlang 이 pt).
/// 표가 한 언어에만 하나 있을 때는, 그 낱말이 그 언어 구역에만 그대로 있는 여섯 글자 이상 소문자 낱말이면 그 언어입니다
/// (prodotti recensiti da 이예준 의 prodotti → it, whatlang 은 pt). 굴절 꼴 맞추기는 이 사전에서 로망스어끼리 엇갈려
/// (incluían → 포르투갈어 incluído, contenant → 영어 contents) 넣지 않았습니다 — v2 질의 468개로 잼 (blueprint-sandbox 의 작은 도메인 사전과 다름).
pub fn latin_vocab_vote(text: &str) -> Option<&'static str> {
    let vocab = LATIN_VOCAB.as_ref()?;
    let names = name_words(text);
    let mut score: Vec<(&'static str, f32, usize)> = LATIN_LANGS.iter().map(|(_, c)| (*c, 0.0, 0)).collect();
    let mut seen: HashSet<String> = HashSet::new();
    let mut sole: Option<&'static str> = None;
    for raw in text.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if raw.chars().count() < 3 || raw.chars().any(|c| c.is_numeric() || c == '_') || names.contains(&raw) {
            continue;
        }
        let w = raw.to_lowercase();
        if !seen.insert(w.clone()) {
            continue;
        }
        if let Some(owners) = vocab.get(&w) {
            if owners.len() == 1 && raw == w && w.chars().count() >= 6 && sole.is_none() {
                sole = Some(owners[0]);
            }
            for s in score.iter_mut().filter(|s| owners.contains(&s.0)) {
                s.1 += 1.0 / owners.len() as f32;
                s.2 += 1;
            }
        }
    }
    score.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let (top, first, hits) = score[0];
    if hits >= 2 && first - score[1].1 >= 0.999 {
        return Some(top);
    }
    (hits == 1 && score[1].1 == 0.0 && sole == Some(top)).then_some(top)
}

/// 이름 줄의 낱말 (원문 꼴): 문장 처음이 아닌 곳에서 대문자로 시작하는 낱말(전부 대문자인 약어는 아님)이 둘 이상 이어진 것.
/// 독일어 명사처럼 따로 선 대문자 낱말(Artikel der Aktion 의 Aktion)은 이름이 아닙니다.
pub fn name_words(text: &str) -> HashSet<&str> {
    let capital = |t: &str| {
        let mut cs = t.chars();
        cs.next().is_some_and(|c| c.is_uppercase()) && (t.chars().count() == 1 || cs.any(|c| c.is_lowercase()))
    };
    let tokens: Vec<&str> = text.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty() && t.chars().all(|c| c.is_alphabetic())).collect();
    let mut out = HashSet::new();
    let mut i = 0;
    while i < tokens.len() {
        let mut j = i;
        while j < tokens.len() && capital(tokens[j]) {
            j += 1;
        }
        if j - i >= 2 && i > 0 {
            out.extend(&tokens[i..j]);
        }
        i = j.max(i + 1);
    }
    out
}

pub fn is_latin_letter(c: char) -> bool {
    c.is_alphabetic() && matches!(c as u32, 0x0041..=0x024F | 0x1E00..=0x1EFF)
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Script {
    Hangul,
    Kana,
    Han,
    Arabic,
    Latin,
    Other,
}

impl Script {
    fn of(c: char) -> Option<Script> {
        Some(match c as u32 {
            0xAC00..=0xD7A3 | 0x1100..=0x11FF | 0x3130..=0x318F => Script::Hangul,
            0x3040..=0x30FF | 0x31F0..=0x31FF | 0xFF66..=0xFF9F => Script::Kana,
            0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF => Script::Han,
            0x0600..=0x06FF | 0x0750..=0x077F | 0xFB50..=0xFDFF | 0xFE70..=0xFEFF => Script::Arabic,
            _ if is_latin_letter(c) => Script::Latin,
            _ if c.is_alphabetic() => Script::Other,
            _ => return None,
        })
    }

    fn cjk(self) -> bool {
        matches!(self, Script::Hangul | Script::Kana | Script::Han)
    }
}

/// 따옴표 짝: 여는 글자 → 닫는 글자들. `'` · `"` 는 낱말 경계에서만 짝으로 봅니다 (l'opération · dell'ordine 의 아포스트로피 제외).
pub const QUOTE_PAIRS: [(char, &str); 10] = [
    ('\'', "'"),
    ('"', "\""),
    ('‘', "’"),
    ('“', "”"),
    ('„', "“”"),
    ('‚', "‘’"),
    ('«', "»"),
    ('‹', "›"),
    ('「', "」"),
    ('『', "』"),
];

/// 따옴표로 묶은 구의 바이트 범위 (따옴표 글자 포함). 행사 · 쿠폰 · 상품 이름을 그대로 적은 값입니다.
pub fn quoted_ranges(query: &str) -> Vec<std::ops::Range<usize>> {
    let chars: Vec<(usize, char)> = query.char_indices().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let (start, c) = chars[i];
        if let Some((_, closers)) = QUOTE_PAIRS.iter().find(|(o, _)| *o == c) {
            let ascii = c == '\'' || c == '"';
            if !ascii || i == 0 || !chars[i - 1].1.is_alphanumeric() {
                let close = (i + 2..chars.len()).find(|&j| {
                    closers.contains(chars[j].1) && (!ascii || j + 1 == chars.len() || !chars[j + 1].1.is_alphanumeric())
                });
                if let Some(j) = close {
                    out.push(start..chars[j].0 + chars[j].1.len_utf8());
                    i = j + 1;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

/// 따옴표로 묶은 구를 공백으로 바꿉니다. 질의 문장의 언어 · 기간은 따옴표 밖 글로 정합니다.
/// 따옴표 밖에 글자가 하나도 없으면(이름만 친 질의) 원문을 그대로 냅니다.
pub fn strip_quoted(query: &str) -> String {
    let mut out = String::with_capacity(query.len());
    let mut at = 0;
    for r in quoted_ranges(query) {
        out.push_str(&query[at..r.start]);
        out.push(' ');
        at = r.end;
    }
    out.push_str(&query[at..]);
    if out.chars().any(|c| c.is_alphabetic()) {
        out
    } else {
        query.to_string()
    }
}

/// 질의를 나눈 낱말(Stanza 토큰 · 공백 어절)마다 따옴표 안에 있는지. 낱말을 원문에서 차례로 찾아 맞추고,
/// 원문에 그대로 없는 낱말(MWT 풀이 du → de le 등)은 따옴표 밖으로 봅니다.
pub fn quoted_token_flags(query: &str, tokens: &[String]) -> Vec<bool> {
    let ranges = quoted_ranges(query);
    let mut at = 0;
    tokens
        .iter()
        .map(|t| match (!t.is_empty()).then(|| query[at..].find(t.as_str())).flatten() {
            Some(off) => {
                let start = at + off;
                at = start + t.len();
                ranges.iter().any(|r| r.start <= start && at <= r.end)
            }
            None => false,
        })
        .collect()
}

/// 간체에만 쓰는 한자 (일본어 신자체와 모양이 다른 것만: 単/单 · 発/发 · 記/记 …). 가나가 섞인 글이
/// 일본어 이름(こもれび商店 · 株式会社ひなた工房)이 든 중국어 문장인지 가립니다.
pub const SIMPLIFIED_ONLY: &str = "单发运购订货车录页选结进领优费详记关实东们这么吗呢给让说话时间门问题对应开买卖务产业场报价钱银账码张网认证书动华汇过还达递邮输转统计设备线级类样历现为处见观视规则质从项阳伞";

/// 일본어 신자체에만 쓰는 한자: 간체와도 번체와도 모양이 다른 것 (気 ↔ 气 · 氣, 圧 ↔ 压 · 壓, 図 ↔ 图 · 圖, 発 ↔ 发 · 發 …).
/// 가나 없이 한자만 쓴 질의가 일본어인지 정할 때 간체 전용 한자 수와 비교합니다. 번체와 같은 글자(風 · 機 · 熱)는 넣지 않습니다 —
/// 번체 중국어 질의(앱은 zh-hans 로 처리)가 일본어로 판정되지 않게.
pub const KANJI_NOT_SIMPLIFIED: &str = "気圧弁変発図転関検験続価売読様鉄総録駅険済戦単囲伝両歓観証豊軽辺銭帰広拡処歯恵団児浄渋猟獣稲穂穏粋継絵緑縁縄繊聴臓舎薬蔵覧訳謡譲賛酔醸釈鋳錬隠雑霊頼顕駆騒髄髪鶏黒黙齢";

/// 라틴 · 아랍 문장에 든 한중일 이름 어절을 뺀 글. 문장 낱말(숫자 · 대문자 약어 · 코드를 뺀 라틴 · 아랍 낱말)이
/// 셋 이상이고 한중일 글자가 든 어절의 두 배 이상일 때만 뺍니다 (products 이예준 wrote a review for →
/// products wrote a review for). 한국어 · 일본어 문장에 든 영어 이름(Felix Wagner 고객이 …)은 그대로 둡니다.
pub fn drop_cjk_names(text: &str) -> String {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let has_cjk = |t: &str| t.chars().filter_map(Script::of).any(Script::cjk);
    let (mut latin_words, mut arabic_words, mut cjk_tokens) = (0usize, 0usize, 0usize);
    for t in &tokens {
        let scripts: Vec<Script> = t.chars().filter_map(Script::of).collect();
        if scripts.is_empty() {
            continue;
        }
        if scripts.iter().any(|s| s.cjk()) {
            cjk_tokens += 1;
        } else if t.chars().any(|c| c.is_ascii_digit()) {
            continue;
        } else if scripts.iter().all(|s| *s == Script::Latin) {
            let letters: Vec<char> = t.chars().filter(|c| c.is_alphabetic()).collect();
            if !(letters.len() >= 2 && letters.iter().all(|c| c.is_uppercase())) {
                latin_words += 1;
            }
        } else if scripts.iter().all(|s| *s == Script::Arabic) {
            arabic_words += 1;
        }
    }
    let sentence_words = latin_words.max(arabic_words);
    // 문장 낱말이 셋 이상이고 한중일 어절의 두 배 이상, 또는 셋 이상이고 한중일 어절보다 많을 때. 뒤 조건은 짧은 로망스어 문장 속 두 어절
    // 이름(productos que 加藤 芽依 reseñó · prodotti recensiti da 加藤 芽依: 문장 낱말 3 · 이름 어절 2)이 중국어로 판정되던 것 (blueprint-sandbox
    // 다국어 검색에서 같은 규칙을 다듬으며 찾음). 한국어 · 일본어 문장 속 영어 이름(Felix Wagner 고객이 주문한 …)은 한중일 어절이 더 많아 그대로입니다.
    if cjk_tokens > 0 && sentence_words >= 3 && (sentence_words >= 2 * cjk_tokens || sentence_words > cjk_tokens) {
        tokens.into_iter().filter(|t| !has_cjk(t)).collect::<Vec<_>>().join(" ")
    } else {
        text.to_string()
    }
}

/// 질의 글에서 변환본 언어를 고릅니다. 12개 언어가 아니면(키릴 · 태국 문자 …) None.
///  · 따옴표로 묶은 이름은 빼고, 라틴 · 아랍 문장에 든 한중일 이름 어절도 뺀 글로 정합니다. 데이터 값(행사 「夏休み準備セール」 ·
///    고객 이예준 · 쇼핑몰 こもれび商店)의 문자 체계가 문장 언어를 덮어 영어 질의가 일본어 사전 · 변환본으로 풀리던 것을 막습니다
///    (v2 관계형 합성 데이터 커머스 질의 120개 중 32개).
///  · 문자 체계: 한글 → ko, 가나 → ja, 가나 없는 한자 → zh-hans (앱 언어가 일본어면 ja), 아랍 문자 → ar
///    한글은 글자 수가 한자 · 가나보다 적어도 어절 끝(조사 · 어미 자리)이 더 많이 한글이면 ko (株式会社ひなた工房에서 보낸 선하증권),
///    가나가 있어도 간체에만 쓰는 한자가 둘 이상이면 zh-hans (在こもれび商店商品详情页更改选项的记录)
///  · 라틴 문자: 그 언어에만 쓰는 글자(ě š č ř ž ů ť ď ň → cs, ã õ → pt, ñ ¿ ¡ → es, ß → de)를 먼저 보고, 다음으로 도메인 사전
///    언어 구역 낱말 투표(배포 사전에서만, [`latin_vocab_vote`]), 아니면 whatlang 을 8개 라틴 언어 안에서 돌립니다. 후보를 넓게 두면
///    짧은 질의가 라틴어 · 슬로바키아어 · 아프리칸스어로 판정되어 영어로 떨어졌습니다 (12개 언어 질의 144개 중 26개).
///    판정을 믿을 수 없고 앱 언어가 라틴 문자 언어면 앱 언어.
pub fn detect_query_lang(query: &str, ui_language: &str) -> Option<&'static str> {
    let text = drop_cjk_names(&strip_quoted(query));
    let query = text.as_str();
    let (mut hangul, mut kana, mut han, mut arabic, mut latin, mut other) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
    for word in query.split_whitespace() {
        // 숫자가 든 낱말(서류 번호 BL-DE-260911 · 주문 번호 · 코드)의 로마자는 문장 언어가 아님 (blueprint-sandbox: 10LABI0AA102 عالق 이 라틴 문자로 판정되던 것)
        let id_like = word.chars().any(|c| c.is_ascii_digit());
        for c in word.chars() {
            match Script::of(c) {
                Some(Script::Hangul) => hangul += 1,
                Some(Script::Kana) => kana += 1,
                Some(Script::Han) => han += 1,
                Some(Script::Arabic) => arabic += 1,
                Some(Script::Latin) if !id_like => latin += 1,
                Some(Script::Latin) => {}
                Some(Script::Other) => other += 1,
                None => {}
            }
        }
    }
    // 어절 끝 한중일 글자의 문자 체계: 한국어는 조사 · 어미가 어절 끝에 붙습니다 (…工房에서 · Goods에서)
    let (mut hangul_tail, mut kanji_tail) = (0usize, 0usize);
    for t in query.split_whitespace() {
        match t.chars().rev().filter_map(Script::of).find(|s| s.cjk()) {
            Some(Script::Hangul) => hangul_tail += 1,
            Some(_) => kanji_tail += 1,
            None => {}
        }
    }
    if hangul > 0 && (hangul >= kana + han || hangul_tail > kanji_tail) {
        return Some("ko");
    }
    if kana > 0 {
        let simplified = query.chars().filter(|c| SIMPLIFIED_ONLY.contains(*c)).count();
        return Some(if simplified >= 2 { "zh-hans" } else { "ja" });
    }
    if han > 0 {
        // 가나 없는 한자만: 앱 언어가 일본어면 일본어. 아니면 일본어 신자체에만 쓰는 한자(気 · 圧 · 弁 · 発 …)가 간체 전용 한자보다 많으면
        // 일본어, 아니면 중국어 (blueprint-sandbox 다국어 검색: 主蒸気圧力 이 앱 언어가 일본어가 아닐 때 중국어로 판정되던 것).
        // 번체와 같은 글자만 쓴 질의(誘引通風機)는 가릴 근거가 없어 예전처럼 중국어입니다.
        if lang_code(ui_language) == Some("ja") {
            return Some("ja");
        }
        let ja_sig = query.chars().filter(|c| KANJI_NOT_SIMPLIFIED.contains(*c)).count();
        let zh_sig = query.chars().filter(|c| SIMPLIFIED_ONLY.contains(*c)).count();
        return Some(if ja_sig > zh_sig { "ja" } else { "zh-hans" });
    }
    if arabic > 0 && arabic >= latin {
        return Some("ar");
    }
    if other > latin {
        return None;
    }
    let lower = query.to_lowercase();
    for (letters, code) in [("ěščřžůťďň", "cs"), ("ãõ", "pt"), ("ñ¿¡", "es"), ("ß", "de")] {
        if lower.chars().any(|c| letters.contains(c)) {
            return Some(code);
        }
    }
    if let Some(code) = latin_vocab_vote(query) {
        return Some(code);
    }
    let ui = lang_code(ui_language).filter(|c| LATIN_LANGS.iter().any(|(_, l)| l == c));
    match LATIN_DETECTOR.detect(query) {
        Some(info) => {
            let code = LATIN_LANGS.iter().find(|(l, _)| *l == info.lang()).map(|(_, c)| *c).unwrap_or("en");
            // 짧은 질의는 whatlang 이 자주 '믿을 수 없음' 입니다. 그때 앱 언어가 라틴 문자 언어면 앱 언어를 따릅니다.
            if info.is_reliable() || ui.is_none() {
                Some(code)
            } else {
                ui
            }
        }
        None => Some(ui.unwrap_or("en")),
    }
}

/// 공백으로 낱말을 나누지 않는 언어 (질의 단위를 Stanza 토큰으로 씀)
pub fn lang_without_spaces(lang: &str) -> bool {
    matches!(lang, "ja" | "zh-hans")
}

/// 앱 모델 목록 이름(`stanza_<이름>`) ↔ 변환본 언어 코드
pub const UI_LANGS: [(&str, &str); 12] = [
    ("korean", "ko"),
    ("english", "en"),
    ("japanese", "ja"),
    ("chinese", "zh-hans"),
    ("french", "fr"),
    ("german", "de"),
    ("spanish", "es"),
    ("italian", "it"),
    ("portuguese", "pt"),
    ("dutch", "nl"),
    ("arabic", "ar"),
    ("czech", "cs"),
];

use chrono::{Datelike, Duration, FixedOffset, NaiveDate, TimeZone, Utc};
use serde_json::json;

/// 평가 하네스가 현재 시각을 고정하는 환경 변수 (epoch ms). 없으면 실제 시각입니다.
///
/// 검색 순위의 최근성 감쇠, 분석 답변 프롬프트의 "Current UTC time" · `days_ago`, 질의 해석의 '오늘' 이 모두 현재 시각에서
/// 나옵니다. 두 실행을 같은 입력으로 대조하려면 (예: 어휘 자르기 off 와 full) 실행 시각이 달라도 이 값들이 같아야 합니다.
pub const EVAL_NOW_ENV: &str = "LOGIS_EVAL_NOW_MS";

/// 현재 시각 (UTC epoch ms). [`EVAL_NOW_ENV`] 가 있으면 그 값
pub fn now_ms() -> i64 {
    static PIN: std::sync::OnceLock<Option<i64>> = std::sync::OnceLock::new();
    PIN.get_or_init(|| std::env::var(EVAL_NOW_ENV).ok().and_then(|v| v.trim().parse::<i64>().ok()).filter(|v| *v > 0))
        .unwrap_or_else(|| Utc::now().timestamp_millis())
}

/// [`now_ms`] 의 UTC 시각
pub fn now_utc() -> chrono::DateTime<Utc> {
    chrono::DateTime::from_timestamp_millis(now_ms()).unwrap_or_else(Utc::now)
}

/// [`now_ms`] 의 지역 시각
pub fn now_local() -> chrono::DateTime<chrono::Local> {
    now_utc().with_timezone(&chrono::Local)
}

// 🌟 벡터 매칭 결과와 언어(국가) 코드를 입력받아 동적으로 완벽한 날짜 필터를 생성합니다.
pub fn get_deterministic_time_guide(vector_guide: &str, lang_code: &str) -> (String, Option<serde_json::Value>) {
    let time_key = marker_key(vector_guide, "Time Intent [");
    let season_key = marker_key(vector_guide, "Season Intent [");
    let p = match resolve_intent(&time_key, &season_key, lang_code, SeasonAnchor::Current) {
        Some(p) => p,
        None => return (String::new(), None),
    };
    let guide = format!(
        "- [DETERMINISTIC OVERRIDE] {} detected ({} ~ {}). DO NOT extract date properties (like started_at, expired_at, date). The system will auto-inject them.",
        p.label, p.start, p.end
    );
    (guide, Some(serde_json::Value::Object(validity_condition(p.start, p.end, "between"))))
}

pub struct IntentPeriod {
    pub start: NaiveDate,
    pub end: NaiveDate,
    pub label: String,
}

pub fn resolve_intent(time_key: &str, season_key: &str, lang_code: &str, anchor: SeasonAnchor) -> Option<IntentPeriod> {
    let (offset, southern) = lang_clock(lang_code);
    let today = today_in(&offset);
    let (start, end) = intent_period(time_key, season_key, today, southern, anchor)?;
    let label = if season_year(season_key, time_key, today, southern, anchor).is_none() {
        format!("Time intent '{}'", time_key)
    } else if time_key.is_empty() {
        format!("Season '{}'", season_key)
    } else {
        format!("Season '{}' of time intent '{}'", season_key, time_key)
    };
    Some(IntentPeriod { start, end, label })
}

pub fn iso_bounds(start: NaiveDate, end: NaiveDate) -> (String, String) {
    (
        format!("{}T00:00:00", start.format("%Y-%m-%d")),
        format!("{}T23:59:59", end.format("%Y-%m-%d")),
    )
}

pub fn stored_wall_clock_ms(start: NaiveDate, end: NaiveDate) -> (i64, i64) {
    let enc = |d: NaiveDate| -> i64 {
        logis_schema::canonical::iso_to_epoch_ms(&format!("{}T00:00:00", d.format("%Y-%m-%d"))).unwrap_or(0)
    };
    let s = enc(start);
    let e = match end.succ_opt() {
        Some(next) => enc(next) - 1,
        None => enc(end) + 86_400_000 - 1,
    };
    (s, e)
}

pub fn validity_condition(start: NaiveDate, end: NaiveDate, operator: &str) -> serde_json::Map<String, serde_json::Value> {
    let (s, e) = stored_wall_clock_ms(start, end);
    let mut m = serde_json::Map::new();
    if operator != "gte" {
        m.insert("started_at".to_string(), json!({ "operator": "lte", "value": e }));
    }
    if operator != "lte" {
        m.insert("expired_at".to_string(), json!({ "operator": "gte", "value": s }));
    }
    m
}

pub const RECENT_DAYS: i64 = 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeasonAnchor {
    Current,
    Latest,
}

fn marker_key(text: &str, marker: &str) -> String {
    text.split(marker)
        .nth(1)
        .and_then(|rest| rest.split(']').next())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

pub fn lang_clock(lang_code: &str) -> (FixedOffset, bool) {
    let norm = if lang_code.trim().is_empty() {
        String::new()
    } else {
        crate::bias_schema::lang_code_of(lang_code)
    };
    let code = norm.split('-').next().unwrap_or("");
    let (offset_minutes, is_southern) = match code {
        "sq" | "ca" | "hr" | "cs" | "da" | "nl" | "fr" | "de" | "hu" | "it" | "no" | "pl" | "sr" | "sk" | "sl" | "es" | "sv" => (60, false),
        "bg" | "et" | "fi" | "el" | "he" | "lv" | "lt" | "ro" | "uk" => (120, false),
        "ar" | "tr" | "ru" => (180, false),
        "sw" => (180, true),
        "fa" => (210, false),
        "az" | "ka" => (240, false),
        "kk" | "ur" | "uz" => (300, false),
        "hi" | "mr" | "te" => (330, false),
        "bn" => (360, false),
        "id" | "km" | "th" | "vi" => (420, false),
        "zh" | "ms" | "tl" => (480, false),
        "ja" | "ko" => (540, false),
        "hy" => (240, false),
        "ml" => (330, false),
        "pt" => (-180, true),
        "is" | "en" => (0, false),
        _ => (540, false),
    };
    let offset = FixedOffset::east_opt(offset_minutes * 60)
        .unwrap_or_else(|| FixedOffset::east_opt(0).unwrap());
    (offset, is_southern)
}

pub fn today_in(offset: &FixedOffset) -> NaiveDate {
    now_utc().with_timezone(offset).date_naive()
}

pub fn date_of_ms(offset: &FixedOffset, ms: i64) -> Option<NaiveDate> {
    chrono::DateTime::from_timestamp_millis(ms).map(|dt| dt.with_timezone(offset).date_naive())
}

pub fn day_start_ms(offset: &FixedOffset, d: NaiveDate) -> i64 {
    offset
        .with_ymd_and_hms(d.year(), d.month(), d.day(), 0, 0, 0)
        .single()
        .map(|t| t.timestamp_millis())
        .unwrap_or(0)
}

pub fn period_ms(offset: &FixedOffset, start: NaiveDate, end: NaiveDate) -> (i64, i64) {
    let s = day_start_ms(offset, start);
    let e = match end.succ_opt() {
        Some(next) => day_start_ms(offset, next) - 1,
        None => day_start_ms(offset, end) + 86_400_000 - 1,
    };
    (s, e)
}

pub fn operator_bounds_ms(offset: &FixedOffset, start: NaiveDate, end: NaiveDate, operator: &str) -> (i64, i64) {
    let (s, e) = period_ms(offset, start, end);
    match operator {
        "gte" => (s, 0),
        "lte" => (0, e),
        _ => (s, e),
    }
}

fn month_last(y: i32, m: u32) -> Option<NaiveDate> {
    let (ny, nm) = if m == 12 { (y + 1, 1) } else { (y, m + 1) };
    NaiveDate::from_ymd_opt(ny, nm, 1)?.pred_opt()
}

fn month_period(y: i32, m: u32) -> Option<(NaiveDate, NaiveDate)> {
    Some((NaiveDate::from_ymd_opt(y, m, 1)?, month_last(y, m)?))
}

fn year_period(y: i32) -> Option<(NaiveDate, NaiveDate)> {
    Some((NaiveDate::from_ymd_opt(y, 1, 1)?, NaiveDate::from_ymd_opt(y, 12, 31)?))
}

pub fn relative_period(key: &str, today: NaiveDate) -> Option<(NaiveDate, NaiveDate)> {
    match key {
        "today" => Some((today, today)),
        "yesterday" => {
            let d = today.pred_opt()?;
            Some((d, d))
        }
        "this_month" => month_period(today.year(), today.month()),
        "last_month" => {
            let prev = NaiveDate::from_ymd_opt(today.year(), today.month(), 1)?.pred_opt()?;
            month_period(prev.year(), prev.month())
        }
        "this_year" => year_period(today.year()),
        "last_year" => year_period(today.year() - 1),
        "recently" => Some((today.checked_sub_signed(Duration::days(RECENT_DAYS))?, today)),
        _ => None,
    }
}

fn season_start_month(season: &str, southern: bool) -> Option<u32> {
    let north: u32 = match season {
        "spring" => 3,
        "summer" => 6,
        "autumn" => 9,
        "winter" => 12,
        _ => return None,
    };
    Some(if southern { (north + 5) % 12 + 1 } else { north })
}

pub fn season_period(season: &str, year: i32, southern: bool) -> Option<(NaiveDate, NaiveDate)> {
    let sm = season_start_month(season, southern)?;
    let start = NaiveDate::from_ymd_opt(year, sm, 1)?;
    let (ey, em) = if sm + 2 > 12 { (year + 1, sm + 2 - 12) } else { (year, sm + 2) };
    Some((start, month_last(ey, em)?))
}

pub fn season_year(season: &str, time_key: &str, today: NaiveDate, southern: bool, anchor: SeasonAnchor) -> Option<i32> {
    season_start_month(season, southern)?;
    let y = today.year();
    if time_key == "last_year" {
        return Some(y - 1);
    }
    for cand in [y - 1, y] {
        if let Some((s, e)) = season_period(season, cand, southern) {
            if s <= today && today <= e {
                return Some(cand);
            }
        }
    }
    let anchor = if time_key == "this_year" { SeasonAnchor::Current } else { anchor };
    match anchor {
        SeasonAnchor::Current => Some(y),
        SeasonAnchor::Latest => {
            let (s, _) = season_period(season, y, southern)?;
            Some(if s <= today { y } else { y - 1 })
        }
    }
}

pub fn intent_period(
    time_key: &str,
    season_key: &str,
    today: NaiveDate,
    southern: bool,
    anchor: SeasonAnchor,
) -> Option<(NaiveDate, NaiveDate)> {
    if let Some(y) = season_year(season_key, time_key, today, southern, anchor) {
        if let Some(p) = season_period(season_key, y, southern) {
            return Some(p);
        }
    }
    relative_period(time_key, today)
}

pub fn anchor_exact_period(
    start: NaiveDate,
    end: NaiveDate,
    year_explicit: bool,
    time_key: &str,
    today: NaiveDate,
    past_only: bool,
) -> (NaiveDate, NaiveDate) {
    if year_explicit {
        return (start, end);
    }
    let back = match time_key {
        "last_year" => true,
        "this_year" => false,
        _ => past_only && start > today,
    };
    if !back {
        return (start, end);
    }
    let shift = |d: NaiveDate| d.checked_sub_months(chrono::Months::new(12)).unwrap_or(d);
    let shifted_end = shift(end);
    let whole_months = start.day() == 1 && end.succ_opt().map_or(false, |n| n.month() != end.month());
    let end_back = if whole_months {
        month_last(shifted_end.year(), shifted_end.month()).unwrap_or(shifted_end)
    } else {
        shifted_end
    };
    (shift(start), end_back)
}

pub fn exact_with_season(
    start: NaiveDate,
    end: NaiveDate,
    granularity: &str,
    season_key: &str,
    southern: bool,
) -> (NaiveDate, NaiveDate, bool) {
    if granularity == "year" && !season_key.is_empty() {
        if let Some((s, e)) = season_period(season_key, start.year(), southern) {
            return (s, e, true);
        }
    }
    (start, end, false)
}

// 🌟 [PERIOD LEXICAL GATE] 기간 뱅크 판정의 글자 근거 (커머스 질의 해석이 씁니다; 예전 analytic.rs 에서 옮김).
/// 기간(시간 · 계절) 뱅크 판정의 글자 근거: 청크 낱말 하나가 그 키의 구 낱말을 품는가 (변화형 포함).
///  · 한중일 · 아랍 문자 구 낱말: 청크 낱말이 그 낱말을 포함 (여름에 ⊃ 여름 · 夏休みに ⊃ 夏 · الصيف ⊃ صيف)
///  · 그 밖의 문자: 앞 글자가 max(3, 낱말 길이 − 2)자 이상 같음 (summers · zomerse · létě ↔ léto). 두 글자 이하 구 낱말과
///    숫자가 든 청크 낱말은 근거로 쓰지 않습니다 (2026 → this_year 는 절대 기간 해석기 몫).
/// 벡터만으로는 근거 없는 기간이 붙었습니다 (v2 행동 로그 질의 중 시간 낱말이 없는 60개 중 55개: "actions on Copperleaf" →
/// 가을 cos 0.67, "UV-blocking parasol" → 여름, "bij" → 올해). 행동 로그는 기간이 틀리면 정답이 통째로 걸러집니다.
/// 기간의 글자 근거가 될 수 없는 기능어 품사 (관계사 · 한정사 · 접속사 · 전치사 · 불변화사 · 대명사 · 문장 부호)
pub const PERIOD_FUNCTION_TAGS: [&str; 7] = ["ADP", "PUNCT", "PART", "SCONJ", "CCONJ", "PRON", "DET"];

/// 🌟 [PERIOD LEXICAL GATE / FUNCTION WORD] 기간 어휘 근거를 볼 글에서 뺄 라틴 문자 기능어 (소문자, 앞뒤 기호 뗌).
/// 라틴 문자 낱말은 앞 글자 일치로 근거를 보는데(굴절 létě ↔ léto · jaře ↔ jaro), 영어 'that'(PRON)이 봄 구 낱말 'thaw' 와 앞 세 글자로
/// 맞아 'visits on 소래마켓 that reached checkout but never paid' 에 가짜 봄 기간이 붙었습니다(v2 질의 A4 영어). 기능어는 기간을 말하지
/// 않으므로 품사로 뺍니다 — 짧은 낱말을 완전 일치로만 받으면 체코어 굴절이 깨집니다. 한중일 · 아랍 문자는 글자 포함으로 보므로 그대로 둡니다.
pub fn period_function_words<'a>(tagged: impl IntoIterator<Item = (&'a str, &'a str)>) -> std::collections::HashSet<String> {
    tagged
        .into_iter()
        .filter(|(_, tag)| PERIOD_FUNCTION_TAGS.contains(tag))
        .map(|(w, _)| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| !w.is_empty() && w.chars().all(|c| c.is_alphabetic() && (c as u32) < 0x0250))
        .collect()
}

/// 글에서 [`period_function_words`] 낱말을 뺍니다.
pub fn strip_period_function_words(text: &str, function_words: &std::collections::HashSet<String>) -> String {
    text.split_whitespace()
        .filter(|w| !function_words.contains(&w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase()))
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn period_lexical_support(chunk: &str, phrases: &[&str]) -> bool {
    let words = |s: &str| -> Vec<String> {
        s.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()).map(str::to_string).collect()
    };
    let cjk = |c: char| matches!(c as u32, 0x3040..=0x30FF | 0x3400..=0x4DBF | 0x4E00..=0x9FFF | 0xAC00..=0xD7A3);
    let arabic = |c: char| matches!(c as u32, 0x0600..=0x06FF);
    let chunk_words: Vec<String> = words(chunk).into_iter().filter(|w| !w.chars().any(|c| c.is_ascii_digit())).collect();
    phrases.iter().flat_map(|p| words(p)).any(|v| {
        let n = v.chars().count();
        match v.chars().next() {
            Some(c) if cjk(c) || (arabic(c) && n >= 3) => chunk_words.iter().any(|w| w.contains(v.as_str())),
            Some(c) if !arabic(c) && n >= 3 => {
                let need = 3.max(n - 2);
                chunk_words.iter().any(|w| w.chars().zip(v.chars()).take_while(|(a, b)| a == b).count() >= need)
            }
            _ => false,
        }
    })
}

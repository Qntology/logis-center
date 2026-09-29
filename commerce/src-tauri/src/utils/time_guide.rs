use chrono::{Datelike, Duration, FixedOffset, NaiveDate, TimeZone, Utc};
use serde_json::json;

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
        crate::utils::canonical::iso_to_epoch_ms(&format!("{}T00:00:00", d.format("%Y-%m-%d"))).unwrap_or(0)
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
        crate::utils::bias_schema::lang_code_of(lang_code)
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
    Utc::now().with_timezone(offset).date_naive()
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
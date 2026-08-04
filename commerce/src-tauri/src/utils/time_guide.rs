use chrono::{Datelike, Utc, Duration, FixedOffset, TimeZone};
use serde_json::json;

// 🌟 벡터 매칭 결과와 언어(국가) 코드를 입력받아 동적으로 완벽한 날짜 필터를 생성합니다.
pub fn get_deterministic_time_guide(vector_guide: &str, lang_code: &str) -> (String, Option<serde_json::Value>) {
    // 🌟 [CRITICAL FIX] 50개 언어 코드를 기반으로 해당 언어권의 대표 UTC Offset(분) 및 남반구(계절 반전) 여부를 매핑합니다.
    let (offset_minutes, is_southern) = match lang_code.to_lowercase().as_str() {
        "sq" | "ca" | "hr" | "cs" | "da" | "nl" | "fr" | "de" | "hu" | "it" | "no" | "pl" | "sr" | "sk" | "sl" | "es" | "sv" => (60, false), // UTC+1
        "bg" | "et" | "fi" | "el" | "he" | "lv" | "lt" | "ro" | "uk" => (120, false), // UTC+2
        "ar" | "tr" | "ru" | "sw" => (180, lang_code.to_lowercase() == "sw"), // UTC+3 (스와힐리어는 남반구/적도 기준)
        "fa" => (210, false), // UTC+3:30
        "az" | "ka" => (240, false), // UTC+4
        "kk" | "ur" | "uz" => (300, false), // UTC+5
        "hi" | "mr" | "te" => (330, false), // UTC+5:30
        "bn" => (360, false), // UTC+6
        "id" | "km" | "th" | "vi" => (420, false), // UTC+7
        "zh" | "ms" | "tl" => (480, false), // UTC+8
        "ja" | "ko" => (540, false), // UTC+9
        "pt" => (-180, true), // UTC-3 (브라질 등 남반구 포르투갈어권 대표)
        "is" | "en" => (0, false), // UTC+0 (영어는 기본 북반구/UTC0로 설정)
        _ => (540, false), // 매칭 안 될 경우 기본값 (한국 KST)
    };
    let offset = FixedOffset::east_opt(offset_minutes * 60).unwrap_or(FixedOffset::east_opt(0).unwrap());
    let now = Utc::now().with_timezone(&offset);
    let mut guide = String::new();
    let mut condition_json = None;
    // 🌟 [CRITICAL FIX] 다국어 계절(Season) 기간 매핑 (남반구/북반구 반전 완벽 적용 및 created_at 쌍방향 주입)
    let mut start_m = 0;
    let mut end_m = 0;
    let mut start_y = now.year();
    let mut end_y = now.year();
    if is_southern {
        if vector_guide.contains("Season Intent [spring]") { start_m = 9; end_m = 11; }
        else if vector_guide.contains("Season Intent [summer]") { start_m = 12; end_m = 2; if now.month() <= 2 { start_y -= 1; } else { end_y += 1; } }
        else if vector_guide.contains("Season Intent [autumn]") { start_m = 3; end_m = 5; }
        else if vector_guide.contains("Season Intent [winter]") { start_m = 6; end_m = 8; }
    } else {
        if vector_guide.contains("Season Intent [spring]") { start_m = 3; end_m = 5; }
        else if vector_guide.contains("Season Intent [summer]") { start_m = 6; end_m = 8; }
        else if vector_guide.contains("Season Intent [autumn]") { start_m = 9; end_m = 11; }
        else if vector_guide.contains("Season Intent [winter]") { start_m = 12; end_m = 2; if now.month() <= 2 { start_y -= 1; } else { end_y += 1; } }
    }
    if start_m != 0 {
        let end_day = match end_m {
            2 => if end_y % 4 == 0 && (end_y % 100 != 0 || end_y % 400 == 0) { 29 } else { 28 },
            4 | 6 | 9 | 11 => 30,
            _ => 31,
        };
        // 🌟 [CRITICAL FIX] LLM에게 추출을 맡기지 않고 확정 JSON 객체를 생성합니다. (밀리세컨드 Timestamp 적용)
        let start_ts = offset.with_ymd_and_hms(start_y, start_m, 1, 0, 0, 0).unwrap().timestamp_millis();
        let end_ts = offset.with_ymd_and_hms(end_y, end_m, end_day, 23, 59, 59).unwrap().timestamp_millis();
        guide = format!("- [DETERMINISTIC OVERRIDE] Season detected. DO NOT extract date properties (like started_at, expired_at, date). The system will auto-inject them.");
        condition_json = Some(json!({
            "started_at": { "operator": "gte", "value": start_ts },
            "expired_at": { "operator": "lte", "value": end_ts }
        }));
        return (guide, condition_json);
    }
    // 🌟 [CRITICAL FIX] 상대적 시간(Time) 로직도 LLM 추론을 우회하여 확정 JSON으로 반환합니다. (밀리세컨드 Timestamp 적용)
    if vector_guide.contains("Time Intent [today]") {
        let start_ts = offset.with_ymd_and_hms(now.year(), now.month(), now.day(), 0, 0, 0).unwrap().timestamp_millis();
        let end_ts = offset.with_ymd_and_hms(now.year(), now.month(), now.day(), 23, 59, 59).unwrap().timestamp_millis();
        guide = format!("- [DETERMINISTIC OVERRIDE] Time intent 'Today' detected. DO NOT extract date properties. The system will auto-inject them.");
        condition_json = Some(json!({
            "started_at": { "operator": "gte", "value": start_ts },
            "expired_at": { "operator": "lte", "value": end_ts }
        }));
    } else if vector_guide.contains("Time Intent [yesterday]") {
        let yesterday = now - Duration::days(1);
        let start_ts = offset.with_ymd_and_hms(yesterday.year(), yesterday.month(), yesterday.day(), 0, 0, 0).unwrap().timestamp_millis();
        let end_ts = offset.with_ymd_and_hms(yesterday.year(), yesterday.month(), yesterday.day(), 23, 59, 59).unwrap().timestamp_millis();
        guide = format!("- [DETERMINISTIC OVERRIDE] Time intent 'Yesterday' detected. DO NOT extract date properties. The system will auto-inject them.");
        condition_json = Some(json!({
            "started_at": { "operator": "gte", "value": start_ts },
            "expired_at": { "operator": "lte", "value": end_ts }
        }));
    } else if vector_guide.contains("Time Intent [this_month]") {
        let end_date = offset.with_ymd_and_hms(
            if now.month() == 12 { now.year() + 1 } else { now.year() },
            if now.month() == 12 { 1 } else { now.month() + 1 },
            1, 0, 0, 0
        ).unwrap() - Duration::seconds(1);
        let start_ts = offset.with_ymd_and_hms(now.year(), now.month(), 1, 0, 0, 0).unwrap().timestamp_millis();
        let end_ts = end_date.timestamp_millis();
        guide = format!("- [DETERMINISTIC OVERRIDE] Time intent 'This Month' detected. DO NOT extract date properties. The system will auto-inject them.");
        condition_json = Some(json!({
            "started_at": { "operator": "gte", "value": start_ts },
            "expired_at": { "operator": "lte", "value": end_ts }
        }));
    } else if vector_guide.contains("Time Intent [last_month]") {
        let (y, m) = if now.month() == 1 { (now.year() - 1, 12) } else { (now.year(), now.month() - 1) };
        let next_m = if m == 12 { 1 } else { m + 1 };
        let next_y = if m == 12 { y + 1 } else { y };
        let end_date = offset.with_ymd_and_hms(next_y, next_m, 1, 0, 0, 0).unwrap() - Duration::seconds(1);
        let start_ts = offset.with_ymd_and_hms(y, m, 1, 0, 0, 0).unwrap().timestamp_millis();
        let end_ts = end_date.timestamp_millis();
        guide = format!("- [DETERMINISTIC OVERRIDE] Time intent 'Last Month' detected. DO NOT extract date properties. The system will auto-inject them.");
        condition_json = Some(json!({
            "started_at": { "operator": "gte", "value": start_ts },
            "expired_at": { "operator": "lte", "value": end_ts }
        }));
    } else if vector_guide.contains("Time Intent [this_year]") {
        let y = now.year();
        let start_ts = offset.with_ymd_and_hms(y, 1, 1, 0, 0, 0).unwrap().timestamp_millis();
        let end_ts = offset.with_ymd_and_hms(y, 12, 31, 23, 59, 59).unwrap().timestamp_millis();
        guide = format!("- [DETERMINISTIC OVERRIDE] Time intent 'This Year' detected. DO NOT extract date properties. The system will auto-inject them.");
        condition_json = Some(json!({
            "started_at": { "operator": "gte", "value": start_ts },
            "expired_at": { "operator": "lte", "value": end_ts }
        }));
    } else if vector_guide.contains("Time Intent [last_year]") {
        let y = now.year() - 1;
        let start_ts = offset.with_ymd_and_hms(y, 1, 1, 0, 0, 0).unwrap().timestamp_millis();
        let end_ts = offset.with_ymd_and_hms(y, 12, 31, 23, 59, 59).unwrap().timestamp_millis();
        guide = format!("- [DETERMINISTIC OVERRIDE] Time intent 'Last Year' detected. DO NOT extract date properties. The system will auto-inject them.");
        condition_json = Some(json!({
            "started_at": { "operator": "gte", "value": start_ts },
            "expired_at": { "operator": "lte", "value": end_ts }
        }));
    } else if vector_guide.contains("Time Intent [recently]") {
        let past = now - Duration::days(30);
        let start_ts = offset.with_ymd_and_hms(past.year(), past.month(), past.day(), 0, 0, 0).unwrap().timestamp_millis();
        let end_ts = offset.with_ymd_and_hms(now.year(), now.month(), now.day(), 23, 59, 59).unwrap().timestamp_millis();
        guide = format!("- [DETERMINISTIC OVERRIDE] Time intent 'Recently' detected. DO NOT extract date properties. The system will auto-inject them.");
        condition_json = Some(json!({
            "started_at": { "operator": "gte", "value": start_ts },
            "expired_at": { "operator": "lte", "value": end_ts }
        }));
    }
    (guide, condition_json)
}
//! `Date` 의 쓰는 부분만: `new Date(x).getTime()` · `toISOString()` · `Date.parse`.
//!
//! Workers 런타임의 시간대는 UTC 라서 "지역 시각"도 UTC 로 본다. 문자열 해석은 ISO 8601 꼴(V8 과 같은 규칙)과
//! `YYYY/MM/DD[ hh:mm[:ss]]` 만 한다. V8 의 옛 해석기가 받는 그 밖의 꼴("Sep 25 2025" 등)은 NaN 이 된다
//! (원본 프롬프트가 `yyyy-MM-ddThh:mm:ss` 를 요구하므로 실제 입력은 ISO 꼴이다).

use crate::jsv::{num_to_string, V};

const MS_PER_DAY: f64 = 86_400_000.0;

fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// JS `MakeDate(MakeDay(y, m0, d), MakeTime(h, mi, s, ms))` (달 · 날 넘침을 굴림)
pub fn make_time(y: f64, month0: f64, day: f64, h: f64, mi: f64, s: f64, ms: f64) -> f64 {
    if ![y, month0, day, h, mi, s, ms].iter().all(|x| x.is_finite()) {
        return f64::NAN;
    }
    let (y, month0) = (y.trunc() + (month0.trunc() / 12.0).floor(), month0.trunc().rem_euclid(12.0));
    let days = days_from_civil(y as i64, month0 as i64 + 1, 1) as f64 + day.trunc() - 1.0;
    time_clip(days * MS_PER_DAY + h.trunc() * 3_600_000.0 + mi.trunc() * 60_000.0 + s.trunc() * 1000.0 + ms.trunc())
}

/// JS `TimeClip`
pub fn time_clip(t: f64) -> f64 {
    if !t.is_finite() || t.abs() > 8.64e15 {
        f64::NAN
    } else {
        t.trunc() + 0.0
    }
}

/// (년, 월 1~12, 일, 시, 분, 초, 밀리초) — UTC
pub fn parts(t: f64) -> (i64, i64, i64, i64, i64, i64, i64) {
    let days = (t / MS_PER_DAY).floor() as i64;
    let ms_in_day = (t - days as f64 * MS_PER_DAY) as i64;
    let (y, m, d) = civil_from_days(days);
    (y, m, d, ms_in_day / 3_600_000, ms_in_day / 60_000 % 60, ms_in_day / 1000 % 60, ms_in_day % 1000)
}

/// `new Date(t).toISOString()` — 잘못된 시각이면 RangeError (Err)
pub fn to_iso_string(t: f64) -> Result<String, crate::jsv::Throw> {
    if t.is_nan() {
        return Err(crate::jsv::Throw::range_error("Invalid time value"));
    }
    let (y, m, d, h, mi, s, ms) = parts(t);
    let year = if (0..=9999).contains(&y) {
        format!("{y:04}")
    } else if y < 0 {
        format!("-{:06}", -y)
    } else {
        format!("+{y:06}")
    };
    Ok(format!("{year}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}.{ms:03}Z"))
}

fn digits(s: &str, n: usize) -> Option<i64> {
    (s.len() == n && s.bytes().all(|b| b.is_ascii_digit())).then(|| s.parse().ok()).flatten()
}

/// ISO 8601 (ECMAScript 날짜 시각 문자열 형식) — 형식 밖이면 None
fn parse_iso(s: &str) -> Option<f64> {
    let (date, rest) = match s.find('T') {
        Some(i) => (&s[..i], Some(&s[i + 1..])),
        None => (s, None),
    };
    let (sign, body) = match date.as_bytes().first() {
        Some(b'+') => (1, &date[1..]),
        Some(b'-') => (-1, &date[1..]),
        _ => (0, date),
    };
    let mut it = body.split('-');
    let y = if sign == 0 { digits(it.next()?, 4)? } else { sign * digits(it.next()?, 6)? };
    if sign == -1 && y == 0 {
        return None;
    }
    let m = match it.next() {
        Some(x) => digits(x, 2)?,
        None => 1,
    };
    let d = match it.next() {
        Some(x) => digits(x, 2)?,
        None => 1,
    };
    if it.next().is_some() || !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let date_only = rest.is_none();
    let (mut h, mut mi, mut sec, mut ms, mut offset_min, mut has_offset) = (0, 0, 0, 0, 0i64, date_only);
    if let Some(time) = rest {
        let (clock, zone) = if let Some(z) = time.strip_suffix('Z') {
            (z, Some(0))
        } else if let Some(i) = time.rfind(['+', '-']) {
            let z = &time[i..];
            let sign = if z.starts_with('-') { -1 } else { 1 };
            let hm = &z[1..];
            let (zh, zm) = hm.split_once(':')?;
            (&time[..i], Some(sign * (digits(zh, 2)? * 60 + digits(zm, 2)?)))
        } else {
            (time, None)
        };
        if let Some(z) = zone {
            offset_min = z;
            has_offset = true;
        }
        let (hms, frac) = match clock.split_once('.') {
            Some((a, f)) => (a, Some(f)),
            None => (clock, None),
        };
        let mut p = hms.split(':');
        h = digits(p.next()?, 2)?;
        mi = digits(p.next()?, 2)?;
        if let Some(x) = p.next() {
            sec = digits(x, 2)?;
        } else if frac.is_some() {
            return None;
        }
        if p.next().is_some() {
            return None;
        }
        if let Some(f) = frac {
            if f.is_empty() || !f.bytes().all(|b| b.is_ascii_digit()) {
                return None;
            }
            let f3: String = f.chars().chain("000".chars()).take(3).collect();
            ms = f3.parse().ok()?;
        }
        if h > 24 || mi > 59 || sec > 59 || (h == 24 && (mi > 0 || sec > 0 || ms > 0)) {
            return None;
        }
    }
    // 시각만 있고 오프셋이 없으면 지역 시각 — Workers 는 UTC
    let _ = has_offset;
    let t = make_time(y as f64, (m - 1) as f64, d as f64, h as f64, mi as f64, sec as f64, ms as f64);
    Some(t - offset_min as f64 * 60_000.0)
}

/// `YYYY/MM/DD[ hh:mm[:ss]]` (V8 옛 해석기가 받는 흔한 꼴, 지역 = UTC)
fn parse_slash(s: &str) -> Option<f64> {
    let (date, time) = match s.split_once(' ') {
        Some((a, b)) => (a, Some(b.trim())),
        None => (s, None),
    };
    let mut it = date.split('/');
    let y: i64 = it.next()?.parse().ok()?;
    let m: i64 = it.next()?.parse().ok()?;
    let d: i64 = it.next()?.parse().ok()?;
    if it.next().is_some() || !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let (mut h, mut mi, mut sec) = (0, 0, 0);
    if let Some(t) = time.filter(|t| !t.is_empty()) {
        let mut p = t.split(':');
        h = p.next()?.parse().ok()?;
        mi = p.next()?.parse().ok()?;
        if let Some(x) = p.next() {
            sec = x.parse().ok()?;
        }
    }
    Some(make_time(y as f64, (m - 1) as f64, d as f64, h as f64, mi as f64, sec as f64, 0.0))
}

/// `Date.parse(s)`
pub fn parse(s: &str) -> f64 {
    let s = s.trim();
    parse_iso(s).or_else(|| parse_slash(s)).map(time_clip).unwrap_or(f64::NAN)
}

/// `new Date(v).getTime()` — 수는 그대로(TimeClip), 문자열은 해석, null 은 0, undefined 는 NaN
pub fn get_time(v: &V) -> f64 {
    match v {
        V::Num(n) => time_clip(*n),
        V::Str(s) => parse(s),
        V::Null => 0.0,
        V::Bool(b) => *b as u8 as f64,
        V::Undef => f64::NAN,
        other => parse(&other.to_str()),
    }
}

/// 템플릿에 넣은 시각 (`${new Date(v).getTime()}`)
pub fn get_time_str(v: &V) -> String {
    num_to_string(get_time(v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_round_trip_and_local_is_utc() {
        assert_eq!(to_iso_string(0.0).unwrap(), "1970-01-01T00:00:00.000Z");
        let t = parse("2025-09-25T18:23:46.364Z");
        assert_eq!(to_iso_string(t).unwrap(), "2025-09-25T18:23:46.364Z");
        assert_eq!(parse("2025-09-25T18:23:46"), parse("2025-09-25T18:23:46Z"));
        assert_eq!(parse("2025-09-25"), parse("2025-09-25T00:00:00Z"));
        assert_eq!(parse("2025-09-25T18:23:46+09:00"), parse("2025-09-25T09:23:46Z"));
        assert!(parse("yyyy-MM-ddThh:mm:ss").is_nan());
        assert!(parse("2025-13-01").is_nan());
        assert_eq!(get_time(&V::Null), 0.0);
        assert!(get_time(&V::Undef).is_nan());
        assert!(to_iso_string(f64::NAN).is_err());
    }
}

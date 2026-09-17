use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

pub const SDS_RECIPE: &str = "sds-v2:welford+ring/tail-sd-entropy/granite-384/bias-json";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Track {
    Vision,
    Trading,
    Commerce,
    Analytic,
}

impl Track {
    pub fn as_str(&self) -> &'static str {
        match self {
            Track::Vision => "vision",
            Track::Trading => "trading",
            Track::Commerce => "commerce",
            Track::Analytic => "analytic",
        }
    }
    /// (2차 스코프 발동, 1차 스코프 발동, 전역 발동)
    pub fn min_obs(&self) -> (u64, u64, u64) {
        match self {
            Track::Vision => (20, 8, 30),
            Track::Trading => (30, 12, 40),
            Track::Commerce => (50, 20, 60),
            Track::Analytic => (40, 15, 0),
        }
    }
    /// 링 버퍼 크기 = 1차 스코프 발동 수. 새 상수를 만들지 않기 위한 재사용입니다.
    pub fn ring_len(&self) -> usize {
        self.min_obs().1 as usize
    }
}

#[derive(Debug, Clone, Default)]
pub struct Scope {
    pub team: String,
    pub track_name: String,
    pub primary: String,
    pub secondary: String,
}

impl Scope {
    pub fn key_secondary(&self) -> String {
        format!("{}|{}|{}", self.track_name, self.primary, self.secondary)
    }
    pub fn key_primary(&self) -> String {
        format!("{}|{}|", self.track_name, self.primary)
    }
    pub fn key_global(&self) -> String {
        format!("{}||", self.track_name)
    }
    pub fn is_empty(&self) -> bool {
        self.track_name.is_empty()
    }
}

static ACTIVE_SCOPE: Lazy<RwLock<Scope>> = Lazy::new(|| RwLock::new(Scope::default()));

pub fn enter_scope(team: &str, track: Track, primary: &str, secondary: &str) {
    if let Ok(mut w) = ACTIVE_SCOPE.write() {
        *w = Scope {
            team: team.to_string(),
            track_name: track.as_str().to_string(),
            primary: primary.trim().to_lowercase(),
            secondary: secondary.trim().to_lowercase(),
        };
    }
}

pub fn refine_primary(primary: &str) {
    let new_primary = primary.trim().to_lowercase();
    let (old_key, new_key, ring) = {
        let s = match ACTIVE_SCOPE.read() { Ok(v) => v.clone(), Err(_) => return };
        if s.is_empty() { return; }
        if s.primary == new_primary { return; }
        let mut ns = s.clone();
        ns.primary = new_primary.clone();
        let ring = match current_track() { Some(t) => t.ring_len(), None => 12 };
        (s.key_secondary(), ns.key_secondary(), ring)
    };

    // 스코프 키를 먼저 교체합니다. (이후 관측은 새 키로)
    if let Ok(mut w) = ACTIVE_SCOPE.write() {
        w.primary = new_primary.clone();
    }

    // 이전 키의 관측을 새 키로 흡수합니다.
    if old_key == new_key { return; }
    let moved = {
        match SDS.write() {
            Ok(mut store) => match store.scopes.remove(&old_key) {
                Some(prev) => {
                    let has = !prev.baseline.is_empty()
                        || !prev.decay.is_empty()
                        || !prev.axis_variance.is_empty()
                        || !prev.field.is_empty()
                        || !prev.confusion.is_empty()
                        || !prev.category.is_empty()
                        || !prev.spatial.is_empty()
                        || !prev.transition.is_empty();
                    if has {
                        let cnt = prev.baseline.len()
                            + prev.decay.len()
                            + prev.field.len()
                            + prev.confusion.len();
                        store
                            .scopes
                            .entry(new_key.clone())
                            .or_insert_with(ScopeStat::default)
                            .absorb(prev, ring);
                        cnt
                    } else {
                        0
                    }
                }
                None => 0,
            },
            Err(_) => 0,
        }
    };
    if moved > 0 {
        if let Ok(mut d) = DIRTY.write() { *d = true; }
        println!(
            "[SDS] 🔀 스코프 정밀화: '{}' → '{}' | 이전 관측 {}축을 새 스코프로 이관했습니다. (같은 문서의 관측이므로 귀속이 정확합니다)",
            old_key, new_key, moved
        );
    }
}

pub fn leave_scope() {
    if let Ok(mut w) = ACTIVE_SCOPE.write() {
        *w = Scope::default();
    }
}

fn current_scope() -> Option<Scope> {
    let s = ACTIVE_SCOPE.read().ok()?.clone();
    if s.is_empty() { None } else { Some(s) }
}

static UNSCOPED_WARNED: Lazy<RwLock<bool>> = Lazy::new(|| RwLock::new(false));

fn effective_scope_key() -> (String, usize) {
    match (current_scope(), current_track()) {
        (Some(s), Some(t)) => (s.key_secondary(), t.ring_len()),
        _ => {
            let already = UNSCOPED_WARNED.read().map(|w| *w).unwrap_or(true);
            if !already {
                if let Ok(mut w) = UNSCOPED_WARNED.write() { *w = true; }
                println!(
                    "[SDS] ⚠️ 활성 스코프 없이 관측이 들어왔습니다. 'unscoped' 로 기록합니다. \
                     enter_scope 호출부가 누락된 경로가 있습니다 — 통계는 남지만 서식별 격리가 되지 않습니다. \
                     (이 경고는 프로세스당 1회만 출력됩니다)"
                );
            }
            ("unscoped||".to_string(), 12usize)
        }
    }
}

fn current_track() -> Option<Track> {
    let s = current_scope()?;
    Some(match s.track_name.as_str() {
        "vision" => Track::Vision,
        "trading" => Track::Trading,
        "commerce" => Track::Commerce,
        "analytic" => Track::Analytic,
        _ => return None,
    })
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Welford {
    pub n: u64,
    pub mean: f64,
    pub m2: f64,
    #[serde(default)]
    pub ring: Vec<f64>,
}

impl Welford {
    pub fn push(&mut self, x: f64, ring_len: usize) {
        if !x.is_finite() { return; }
        self.n += 1;
        let d = x - self.mean;
        self.mean += d / (self.n as f64);
        self.m2 += d * (x - self.mean);
        self.ring.push(x);
        if ring_len > 0 && self.ring.len() > ring_len {
            let excess = self.ring.len() - ring_len;
            self.ring.drain(0..excess);
        }
    }
    pub fn variance(&self) -> f64 {
        if self.n < 2 { 0.0 } else { self.m2 / ((self.n - 1) as f64) }
    }
    pub fn sd(&self) -> f64 {
        self.variance().max(0.0).sqrt()
    }
    pub fn recent_mean(&self) -> f64 {
        if self.ring.is_empty() { return self.mean; }
        self.ring.iter().sum::<f64>() / (self.ring.len() as f64)
    }
    pub fn drift_z(&self) -> f64 {
        let sd = self.sd();
        if sd <= 0.0 || self.ring.is_empty() { return 0.0; }
        (self.recent_mean() - self.mean) / sd
    }
    pub fn merge(&mut self, other: &Welford, ring_len: usize) {
        if other.n == 0 { return; }
        if self.n == 0 {
            self.n = other.n;
            self.mean = other.mean;
            self.m2 = other.m2;
            self.ring = other.ring.clone();
        } else {
            let na = self.n as f64;
            let nb = other.n as f64;
            let n = na + nb;
            let delta = other.mean - self.mean;
            self.mean += delta * (nb / n);
            self.m2 += other.m2 + delta * delta * (na * nb / n);
            self.n += other.n;
            self.ring.extend(other.ring.iter().cloned());
        }
        if ring_len > 0 && self.ring.len() > ring_len {
            let excess = self.ring.len() - ring_len;
            self.ring.drain(0..excess);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DecayShape {
    pub n: usize,
    pub top1: f64,
    pub top2: f64,
    pub margin: f64,
    pub top_gap_ratio: f64,
    pub tail_flatness: f64,
    pub entropy_norm: f64,
    pub positive_ratio: f64,
}

pub fn decay_shape(scores: &[f32]) -> Option<DecayShape> {
    let mut v: Vec<f64> = scores
        .iter()
        .filter(|s| s.is_finite())
        .map(|s| *s as f64)
        .collect();
    if v.len() < 2 { return None; }
    v.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let top1 = v[0];
    let top2 = v[1];
    let margin = top1 - top2;
    let tail = &v[1..];
    let tail_median = tail[tail.len() / 2];
    let span = (top1 - tail_median).abs();
    let top_gap_ratio = if span > 1e-9 { (margin / span).clamp(0.0, 1.0) } else { 0.0 };
    let tail_mean = tail.iter().sum::<f64>() / (tail.len() as f64);
    let tail_var = tail.iter().map(|x| (x - tail_mean).powi(2)).sum::<f64>() / (tail.len() as f64);
    let full_range = (v[0] - v[n - 1]).abs();
    let tail_flatness = if full_range > 1e-9 {
        (tail_var.sqrt() / full_range).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let entropy_norm = {
        let tail_sd_for_scale = tail_var.sqrt();
        if tail_sd_for_scale <= 1e-9 {
            1.0f64
        } else {
            let mx = v[0];
            let exps: Vec<f64> = v.iter().map(|x| ((x - mx) / tail_sd_for_scale).exp()).collect();
            let sum: f64 = exps.iter().sum::<f64>().max(1e-12);
            let mut h = 0.0f64;
            for e in exps.iter() {
                let p = e / sum;
                if p > 1e-12 { h -= p * p.ln(); }
            }
            let hmax = (n as f64).ln();
            if hmax > 1e-9 { (h / hmax).clamp(0.0, 1.0) } else { 0.0 }
        }
    };

    let positive_ratio = v.iter().filter(|x| **x > 0.0).count() as f64 / (n as f64);

    Some(DecayShape {
        n,
        top1,
        top2,
        margin,
        top_gap_ratio,
        tail_flatness,
        entropy_norm,
        positive_ratio,
    })
}

// =====================================================================
// 🌟 [저장 레코드] 기획 6-1 의 통계 레코드 정의를 그대로 옮긴 구조입니다.
// =====================================================================
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DecayStat {
    pub margin: Welford,
    pub top_gap_ratio: Welford,
    pub tail_flatness: Welford,
    pub entropy_norm: Welford,
    pub positive_ratio: Welford,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FieldRejectStat {
    /// 후보로 검토된 횟수
    pub seen: u64,
    pub reject_format: u64,
    pub reject_prejudice: u64,
    pub reject_enum: u64,
    pub reject_self_id: u64,
    /// 니어미스(CONFIRM FLAG) 발생 횟수
    pub near_miss: u64,
    /// 최종 확정 횟수
    pub assigned: u64,
    /// 확정 시 마진 분포
    pub assign_margin: Welford,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConfusionStat {
    pub ties: u64,
    pub a_wins: u64,
    pub b_wins: u64,
    pub margin: Welford,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CategoryStat {
    /// 스키마상 필드 수 (구조가 가정한 드로잉 수 N)
    pub n_fields: u64,
    /// 실현 최댓값 분포. 여기서 N_eff 를 역산합니다(Phase 2).
    pub realized_max: Welford,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpatialStat {
    /// 활성 패치 / 전체 패치
    pub active_ratio: Welford,
    /// 크롭 밖으로 밀려난 활성 패치 비율
    pub coverage_loss: Welford,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ScopeStat {
    /// 축별 베이스라인 (TITLE FLOOR, 잡음대, pooled σ 등)
    #[serde(default)]
    pub baseline: HashMap<String, Welford>,
    /// 축별 감쇠 형상
    #[serde(default)]
    pub decay: HashMap<String, DecayStat>,
    /// 축별 후보 분산 (역분산 융합의 입력, Phase 1)
    #[serde(default)]
    pub axis_variance: HashMap<String, Welford>,
    /// 필드별 거절/확정 트레이스
    #[serde(default)]
    pub field: HashMap<String, FieldRejectStat>,
    /// 필드 쌍 혼동 사전
    #[serde(default)]
    pub confusion: HashMap<String, ConfusionStat>,
    /// 카테고리별 실현 최댓값
    #[serde(default)]
    pub category: HashMap<String, CategoryStat>,
    /// 비전 공간 통계
    #[serde(default)]
    pub spatial: HashMap<String, SpatialStat>,
    /// analytic 도메인 전이 카운트 ("from>to" → 횟수)
    #[serde(default)]
    pub transition: HashMap<String, u64>,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SdsFile {
    pub recipe: String,
    pub team: String,
    pub updated_at: i64,
    #[serde(default)]
    pub scopes: HashMap<String, ScopeStat>,
}

impl ScopeStat {
    /// 🌟 [MERGE] 다른 스코프의 통계를 흡수합니다. (refine_primary 이관용)
    pub fn absorb(&mut self, other: ScopeStat, ring: usize) {
        for (k, v) in other.baseline {
            self.baseline.entry(k).or_insert_with(Welford::default).merge(&v, ring);
        }
        for (k, v) in other.decay {
            let d = self.decay.entry(k).or_insert_with(DecayStat::default);
            d.margin.merge(&v.margin, ring);
            d.top_gap_ratio.merge(&v.top_gap_ratio, ring);
            d.tail_flatness.merge(&v.tail_flatness, ring);
            d.entropy_norm.merge(&v.entropy_norm, ring);
            d.positive_ratio.merge(&v.positive_ratio, ring);
        }
        for (k, v) in other.axis_variance {
            self.axis_variance.entry(k).or_insert_with(Welford::default).merge(&v, ring);
        }
        for (k, v) in other.field {
            let f = self.field.entry(k).or_insert_with(FieldRejectStat::default);
            f.seen += v.seen;
            f.reject_format += v.reject_format;
            f.reject_prejudice += v.reject_prejudice;
            f.reject_enum += v.reject_enum;
            f.reject_self_id += v.reject_self_id;
            f.near_miss += v.near_miss;
            f.assigned += v.assigned;
            f.assign_margin.merge(&v.assign_margin, ring);
        }
        for (k, v) in other.confusion {
            let c = self.confusion.entry(k).or_insert_with(ConfusionStat::default);
            c.ties += v.ties;
            c.a_wins += v.a_wins;
            c.b_wins += v.b_wins;
            c.margin.merge(&v.margin, ring);
        }
        for (k, v) in other.category {
            let c = self.category.entry(k).or_insert_with(CategoryStat::default);
            if c.n_fields == 0 { c.n_fields = v.n_fields; }
            c.realized_max.merge(&v.realized_max, ring);
        }
        for (k, v) in other.spatial {
            let s = self.spatial.entry(k).or_insert_with(SpatialStat::default);
            s.active_ratio.merge(&v.active_ratio, ring);
            s.coverage_loss.merge(&v.coverage_loss, ring);
        }
        for (k, v) in other.transition {
            *self.transition.entry(k).or_insert(0) += v;
        }
        self.updated_at = chrono::Utc::now().timestamp_millis();
    }
}

static SDS: Lazy<RwLock<SdsFile>> = Lazy::new(|| RwLock::new(SdsFile::default()));
static DIRTY: Lazy<RwLock<bool>> = Lazy::new(|| RwLock::new(false));
fn sds_path() -> std::path::PathBuf {
    crate::utils::get_app_dir().join("score_dynamics.json")
}

pub fn load(team: &str) {
    let path = sds_path();
    let mut fresh = SdsFile {
        recipe: SDS_RECIPE.to_string(),
        team: team.to_string(),
        updated_at: chrono::Utc::now().timestamp_millis(),
        scopes: HashMap::new(),
    };
    if path.exists() {
        match std::fs::read_to_string(&path) {
            Ok(txt) => match serde_json::from_str::<SdsFile>(&txt) {
                Ok(f) => {
                    if f.recipe != SDS_RECIPE {
                        println!(
                            "[SDS] 🧹 레시피 세대 불일치 (저장 '{}' vs 현재 '{}'). 통계를 전량 폐기하고 새로 시작합니다.",
                            f.recipe, SDS_RECIPE
                        );
                    } else if !team.is_empty()
                        && !f.team.is_empty()
                        && f.team != team
                        && f.team == local_default_team()
                    {
                        let n = f.scopes.len();
                        let prev_team = f.team.clone();
                        fresh = f;
                        fresh.team = team.to_string();
                        println!(
                            "[SDS] 🚚 [TEAM MIGRATE] 로컬 기본 팀 '{}' 에 귀속된 스코프 {}개를 실제 팀 '{}' 로 이관했습니다.",
                            prev_team, n, team
                        );
                    } else if !team.is_empty() && !f.team.is_empty() && f.team != team {
                        println!(
                            "[SDS] 🧹 팀 불일치 (저장 '{}' vs 현재 '{}'). 통계를 전량 폐기합니다. (스코프 격리 원칙)",
                            f.team, team
                        );
                    } else {
                        let n = f.scopes.len();
                        fresh = f;
                        fresh.team = team.to_string();
                        println!("[SDS] ✅ 점수 동역학 통계를 불러왔습니다. 스코프 {}개.", n);
                    }
                }
                Err(e) => println!("[SDS] ⚠️ 통계 파싱 실패({}). 새로 시작합니다.", e),
            },
            Err(e) => println!("[SDS] ⚠️ 통계 읽기 실패({}). 새로 시작합니다.", e),
        }
    } else {
        println!("[SDS] 🆕 저장된 통계가 없습니다. 냉간 시작합니다. (현행 판정 그대로)");
    }
    if let Ok(mut w) = SDS.write() { *w = fresh; }
    if let Ok(mut d) = DIRTY.write() { *d = true; }
    flush();
    println!("[SDS] 📍 통계 파일 경로: {}", sds_path().display());
}

fn local_default_team() -> String {
    crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000")
}

pub fn rebind_team(team: &str) {
    let prev = SDS.read().ok().map(|s| s.team.clone()).unwrap_or_default();
    if prev.is_empty() || prev == team {
        if let Ok(mut w) = SDS.write() {
            w.team = team.to_string();
        }
        return;
    }
    if prev == local_default_team() {
        let scopes = SDS.read().ok().map(|s| s.scopes.len()).unwrap_or(0);
        if let Ok(mut w) = SDS.write() {
            w.team = team.to_string();
        }
        if let Ok(mut d) = DIRTY.write() { *d = true; }
        flush();
        let after = SDS.read().ok().map(|s| s.scopes.len()).unwrap_or(0);
        println!(
            "[SDS] 🚚 [TEAM MIGRATE] 로컬 기본 팀의 스코프 {}개를 실제 팀 '{}' 로 이관했습니다. (이관 후 잔존 {}개)",
            scopes, team, after
        );
        if scopes == 0 {
            println!(
                "[SDS] ⚠️ [TEAM MIGRATE] 이관 시점에 메모리 스코프가 0개였습니다. load() 가 이미 폐기했을 가능성이 있으니 파일의 team 값을 확인하십시오."
            );
        }
        return;
    }
    println!("[SDS] 🔄 팀 전환 감지. 이전 팀 통계를 폐기하고 새 팀으로 재바인딩합니다.");
    purge();
    load(team);
}

pub fn flush() {
    let dirty = DIRTY.read().map(|d| *d).unwrap_or(false);
    if !dirty {
        // 🌟 [진단] '쓸 것이 없어서 안 썼다' 를 명시합니다.
        //    이 줄이 없으면 '호출은 됐는데 아무 일도 안 일어난' 상황과
        //    '호출 자체가 안 된' 상황을 구분할 수 없습니다.
        println!("[SDS] ⏭️ 새 관측이 없어 저장을 건너뜁니다. (dirty=false)");
        return;
    }
    let snapshot = match SDS.read() { Ok(s) => s.clone(), Err(_) => return };
    let mut snapshot = snapshot;
    snapshot.updated_at = chrono::Utc::now().timestamp_millis();
    snapshot.recipe = SDS_RECIPE.to_string();

    // 🌟 [상한] 기획 6-1 의 2MB 상한. 초과 시 관측 수가 적고 오래된 스코프부터 절삭합니다.
    let mut txt = match serde_json::to_string_pretty(&snapshot) {
        Ok(t) => t,
        Err(_) => return,
    };
    const CAP_BYTES: usize = 2 * 1024 * 1024;
    if txt.len() > CAP_BYTES {
        let mut keys: Vec<(String, u64, i64)> = snapshot
            .scopes
            .iter()
            .map(|(k, v)| {
                let obs: u64 = v.baseline.values().map(|w| w.n).sum::<u64>()
                    + v.field.values().map(|f| f.seen).sum::<u64>();
                (k.clone(), obs, v.updated_at)
            })
            .collect();
        // 관측이 적고 오래된 순으로 정렬
        keys.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)));
        let mut trimmed = snapshot.clone();
        for (k, _, _) in keys {
            if txt.len() <= CAP_BYTES { break; }
            trimmed.scopes.remove(&k);
            txt = serde_json::to_string_pretty(&trimmed).unwrap_or(txt);
        }
        println!("[SDS] ✂️ 상한 초과로 저관측 스코프를 절삭했습니다. (잔존 {}개)", trimmed.scopes.len());
    }

    let path = sds_path();
    if let Some(dir) = path.parent() { let _ = std::fs::create_dir_all(dir); }
    match std::fs::write(&path, txt.as_bytes()) {
        Ok(_) => {
            if let Ok(mut d) = DIRTY.write() { *d = false; }
            // 🌟 [DEBOUNCE v2] 저장에 성공했으므로 누적 관측 카운터를 0 으로 되돌립니다.
            //    이 리셋이 없으면 카운터가 임계 이상으로 굳어 매 관측마다 파일을 씁니다.
            if let Ok(mut t) = WRITE_TICK.write() { *t = 0; }
            println!("[SDS] 💾 점수 동역학 통계를 저장했습니다. ({}바이트)", txt.len());
        }
        Err(e) => println!("[SDS] ⚠️ 통계 저장 실패: {}", e),
    }
}

pub fn purge() {
    if let Ok(mut w) = SDS.write() {
        *w = SdsFile {
            recipe: SDS_RECIPE.to_string(),
            team: w.team.clone(),
            updated_at: chrono::Utc::now().timestamp_millis(),
            scopes: HashMap::new(),
        };
    }
    let _ = std::fs::remove_file(sds_path());
    if let Ok(mut d) = DIRTY.write() { *d = false; }
    println!("[SDS] 🗑️ 점수 동역학 통계를 전량 삭제했습니다. 판정은 즉시 현행 상수로 복귀합니다.");
}

fn with_scope_mut<F: FnOnce(&mut ScopeStat, usize)>(f: F) {
    // 🌟 [UNSCOPED FALLBACK] 스코프 유무와 무관하게 반드시 기록합니다.
    let (key, ring) = effective_scope_key();
    if let Ok(mut w) = SDS.write() {
        let e = w.scopes.entry(key).or_insert_with(ScopeStat::default);
        f(e, ring);
        e.updated_at = chrono::Utc::now().timestamp_millis();
    }
    if let Ok(mut d) = DIRTY.write() { *d = true; }
    // 🌟 [AUTO FLUSH] 관측이 일정량 쌓이면 태스크 종료를 기다리지 않고 기록합니다.
    bump_and_maybe_flush();
}

static WRITE_TICK: Lazy<RwLock<u32>> = Lazy::new(|| RwLock::new(0));
static LAST_FLUSH: Lazy<RwLock<Option<std::time::Instant>>> = Lazy::new(|| RwLock::new(None));
const AUTO_FLUSH_EVERY: u32 = 8;
const AUTO_FLUSH_MIN_GAP_MS: u128 = 5_000;

fn bump_and_maybe_flush() {
    let tick_ok = match WRITE_TICK.write() {
        Ok(mut t) => {
            *t = t.saturating_add(1);
            *t >= AUTO_FLUSH_EVERY
        }
        Err(_) => false,
    };
    if !tick_ok { return; }

    let time_ok = match LAST_FLUSH.read() {
        Ok(g) => match *g {
            Some(inst) => inst.elapsed().as_millis() >= AUTO_FLUSH_MIN_GAP_MS,
            None => true,
        },
        Err(_) => false,
    };
    if !time_ok { return; }

    if let Ok(mut g) = LAST_FLUSH.write() { *g = Some(std::time::Instant::now()); }
    flush();
}

// =====================================================================
// 🌟 [SSR] 관측 기록 API
// ---------------------------------------------------------------------
//  전부 반환값이 없고 실패해도 조용히 무시합니다.
//  계측이 판정을 방해하면 안 되기 때문입니다.
// =====================================================================

/// 축 베이스라인. TITLE FLOOR, 잡음대, pooled σ, net 표준편차 등.
pub fn record_baseline(axis: &str, value: f32) {
    if !value.is_finite() { return; }
    with_scope_mut(|s, ring| {
        s.baseline
            .entry(axis.to_string())
            .or_insert_with(Welford::default)
            .push(value as f64, ring);
    });
}

/// 순위 감쇠 곡선. 후보 점수 배열 전체를 넘기면 형상만 압축해 저장합니다.
pub fn record_decay(axis: &str, scores: &[f32]) {
    let shape = match decay_shape(scores) { Some(s) => s, None => return };
    with_scope_mut(|s, ring| {
        let d = s.decay.entry(axis.to_string()).or_insert_with(DecayStat::default);
        d.margin.push(shape.margin, ring);
        d.top_gap_ratio.push(shape.top_gap_ratio, ring);
        d.tail_flatness.push(shape.tail_flatness, ring);
        d.entropy_norm.push(shape.entropy_norm, ring);
        d.positive_ratio.push(shape.positive_ratio, ring);
    });
    // 축 분산은 역분산 융합(Phase 1)의 직접 입력이므로 별도 축으로도 남깁니다.
    let var = {
        let v: Vec<f64> = scores.iter().filter(|x| x.is_finite()).map(|x| *x as f64).collect();
        if v.len() < 2 { 0.0 } else {
            let m = v.iter().sum::<f64>() / (v.len() as f64);
            v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / ((v.len() - 1) as f64)
        }
    };
    if var > 0.0 {
        with_scope_mut(|s, ring| {
            s.axis_variance
                .entry(axis.to_string())
                .or_insert_with(Welford::default)
                .push(var, ring);
        });
    }
}

#[derive(Debug, Clone, Copy)]
pub enum GateKind {
    Format,
    Prejudice,
    Enum,
    SelfId,
}

/// 필드가 후보로 검토되었음을 기록합니다.
pub fn record_field_seen(field: &str) {
    with_scope_mut(|s, _| {
        s.field.entry(field.to_string()).or_insert_with(FieldRejectStat::default).seen += 1;
    });
}

/// 게이트 거절. 학습형 prejudice(Phase 2)의 유일한 입력입니다.
///
/// ⚠️ 이 신호가 중요한 이유: 게이트 거절률은 '판정 결과' 가 아니라
///    '판정과 독립된 형식·구조 사실' 입니다. 판정 결과를 학습하면
///    자기강화 피드백(R1)에 걸리지만, 거절률은 그 위험이 없습니다.
pub fn record_field_reject(field: &str, kind: GateKind) {
    with_scope_mut(|s, _| {
        let e = s.field.entry(field.to_string()).or_insert_with(FieldRejectStat::default);
        match kind {
            GateKind::Format => e.reject_format += 1,
            GateKind::Prejudice => e.reject_prejudice += 1,
            GateKind::Enum => e.reject_enum += 1,
            GateKind::SelfId => e.reject_self_id += 1,
        }
    });
}

/// 필드 확정. 마진 분포를 함께 남겨 적응형 마진(Phase 1)의 기준선으로 씁니다.
pub fn record_field_assigned(field: &str, margin: f32) {
    with_scope_mut(|s, ring| {
        let e = s.field.entry(field.to_string()).or_insert_with(FieldRejectStat::default);
        e.assigned += 1;
        if margin.is_finite() {
            e.assign_margin.push(margin as f64, ring);
        }
    });
}

pub fn record_near_miss(field: &str) {
    with_scope_mut(|s, _| {
        s.field.entry(field.to_string()).or_insert_with(FieldRejectStat::default).near_miss += 1;
    });
}

/// 혼동 쌍. 값은 저장하지 않고 필드명만 남깁니다(프라이버시 정책).
pub fn record_confusion(winner: &str, loser: &str, margin: f32) {
    if winner.is_empty() || loser.is_empty() || winner == loser { return; }
    // 키 순서를 사전순으로 고정해 (A,B) 와 (B,A) 가 같은 레코드를 쓰게 합니다.
    let (a, b, winner_is_a) = if winner <= loser {
        (winner.to_string(), loser.to_string(), true)
    } else {
        (loser.to_string(), winner.to_string(), false)
    };
    let key = format!("{}|{}", a, b);
    with_scope_mut(|s, ring| {
        let e = s.confusion.entry(key).or_insert_with(ConfusionStat::default);
        e.ties += 1;
        if winner_is_a { e.a_wins += 1; } else { e.b_wins += 1; }
        if margin.is_finite() { e.margin.push(margin as f64, ring); }
    });
}

/// 카테고리 실현 최댓값. CATEGORY-NEUTRAL 의 N_eff 캘리브레이션(Phase 2) 입력.
pub fn record_category_max(category: &str, n_fields: usize, realized_max: f32) {
    if !realized_max.is_finite() { return; }
    with_scope_mut(|s, ring| {
        let e = s.category.entry(category.to_string()).or_insert_with(CategoryStat::default);
        e.n_fields = n_fields as u64;
        e.realized_max.push(realized_max as f64, ring);
    });
}

/// 비전 히트맵 확산도. V-1 공간 잔차화(Phase 1)의 판정 기준선.
pub fn record_spatial(category: &str, active: usize, total: usize) {
    if total == 0 { return; }
    let ratio = active as f64 / total as f64;
    with_scope_mut(|s, ring| {
        s.spatial
            .entry(category.to_string())
            .or_insert_with(SpatialStat::default)
            .active_ratio
            .push(ratio, ring);
    });
}

/// 크롭 커버리지 손실률. V-2(Phase 2) 입력.
pub fn record_coverage_loss(category: &str, lost_ratio: f32) {
    if !lost_ratio.is_finite() { return; }
    with_scope_mut(|s, ring| {
        s.spatial
            .entry(category.to_string())
            .or_insert_with(SpatialStat::default)
            .coverage_loss
            .push(lost_ratio as f64, ring);
    });
}

/// analytic 도메인 전이. U-2(Phase 3) 입력.
pub fn record_transition(from: &str, to: &str) {
    if from.is_empty() || to.is_empty() { return; }
    let key = format!("{}>{}", from, to);
    with_scope_mut(|s, _| {
        *s.transition.entry(key).or_insert(0) += 1;
    });
}

fn resolve<T, F>(pick: F) -> Option<T>
where
    F: Fn(&ScopeStat) -> Option<(T, u64)>,
{
    let scope = current_scope()?;
    let track = current_track()?;
    let (m2, m1, mg) = track.min_obs();
    let store = SDS.read().ok()?;
    for (key, need) in [
        (scope.key_secondary(), m2),
        (scope.key_primary(), m1),
        (scope.key_global(), mg),
        ("unscoped||".to_string(), mg.max(m1)),
    ] {
        if need == 0 { continue; }
        if let Some(st) = store.scopes.get(&key) {
            if let Some((val, n)) = pick(st) {
                if n >= need { return Some(val); }
            }
        }
    }
    None
}

/// 적응형 베이스라인. (평균, 표준편차) 를 돌려줍니다.
/// TITLE FLOOR / 잡음대 / dedup_floor 를 대체할 때 씁니다(Phase 1).
pub fn adaptive_baseline(axis: &str) -> Option<(f32, f32)> {
    resolve(|st| {
        st.baseline
            .get(axis)
            .map(|w| ((w.mean as f32, w.sd() as f32), w.n))
    })
}

/// 축 신뢰도. 역분산 융합의 가중치로 씁니다(Phase 1).
/// 분산이 작을수록(변별력 없음) 낮은 값을 돌려줍니다.
pub fn axis_confidence(axis: &str) -> Option<f32> {
    resolve(|st| {
        st.axis_variance.get(axis).map(|w| {
            let v = w.mean.max(1e-9);
            ((1.0 / v) as f32, w.n)
        })
    })
}

/// 이 축의 통상 감쇠 형상. 적응형 마진 판정의 기준선입니다(Phase 1).
/// (평탄도 평균, 평탄도 표준편차, 마진 평균, 마진 표준편차)
pub fn decay_baseline(axis: &str) -> Option<(f32, f32, f32, f32)> {
    resolve(|st| {
        st.decay.get(axis).map(|d| {
            (
                (
                    d.tail_flatness.mean as f32,
                    d.tail_flatness.sd() as f32,
                    d.margin.mean as f32,
                    d.margin.sd() as f32,
                ),
                d.margin.n,
            )
        })
    })
}

/// 학습형 특이도. prejudice 뱅크가 빈 필드의 대체 페널티입니다(Phase 2).
/// 거절률이 높은 필드일수록 큰 값을 돌려줍니다.
pub fn learned_specificity(field: &str) -> Option<f32> {
    resolve(|st| {
        st.field.get(field).map(|f| {
            let rejected = f.reject_format + f.reject_prejudice + f.reject_enum + f.reject_self_id;
            let denom = f.seen.max(1) as f64;
            ((rejected as f64 / denom) as f32, f.seen)
        })
    })
}

pub fn confusion_winner(a: &str, b: &str) -> Option<(String, f32)> {
    if a.is_empty() || b.is_empty() || a == b { return None; }
    let (x, y) = if a <= b { (a, b) } else { (b, a) };
    let key = format!("{}|{}", x, y);
    let x_owned = x.to_string();
    let y_owned = y.to_string();
    resolve(move |st| {
        st.confusion.get(&key).map(|c| {
            let total = c.ties.max(1) as f32;
            if c.a_wins >= c.b_wins {
                ((x_owned.clone(), c.a_wins as f32 / total), c.ties)
            } else {
                ((y_owned.clone(), c.b_wins as f32 / total), c.ties)
            }
        })
    })
}

pub fn effective_draws(category: &str) -> Option<f32> {
    resolve(|st| {
        st.category.get(category).map(|c| {
            let m = c.realized_max.mean.max(0.0);
            let n_eff = (m * m / 2.0).exp().max(1.0);
            let cap = c.n_fields.max(1) as f64;
            (n_eff.min(cap) as f32, c.realized_max.n)
        })
    })
}

/// 비전 확산도 드리프트. V-1 폴백 판정에 씁니다(Phase 1).
pub fn spatial_drift(category: &str) -> Option<f32> {
    resolve(|st| {
        st.spatial
            .get(category)
            .map(|s| (s.active_ratio.drift_z() as f32, s.active_ratio.n))
    })
}

/// 도메인 전이 확률. analytic 동적 사전(Phase 3).
pub fn transition_prior(from: &str, to: &str) -> Option<f32> {
    let scope = current_scope()?;
    let track = current_track()?;
    let (m2, m1, _) = track.min_obs();
    let store = SDS.read().ok()?;
    for (key, need) in [(scope.key_secondary(), m2), (scope.key_primary(), m1)] {
        let st = match store.scopes.get(&key) { Some(v) => v, None => continue };
        let total: u64 = st
            .transition
            .iter()
            .filter(|(k, _)| k.starts_with(&format!("{}>", from)))
            .map(|(_, v)| *v)
            .sum();
        if total < need { continue; }
        let hit = st.transition.get(&format!("{}>{}", from, to)).copied().unwrap_or(0);
        // 디리클레 평활: 관측되지 않은 전이도 0 이 되지 않게 합니다.
        let k = st.transition.len().max(1) as f64;
        return Some((((hit as f64) + 1.0) / ((total as f64) + k)) as f32);
    }
    None
}

// =====================================================================
// 🌟 [진단] 현재 축적 상태 한 줄 요약. 태스크 종료 시 로그로 남깁니다.
// =====================================================================
pub fn report() -> String {
    let store = match SDS.read() { Ok(s) => s, Err(_) => return "[SDS] (잠금 실패)".to_string() };
    let scopes = store.scopes.len();
    let baseline_obs: u64 = store.scopes.values().flat_map(|s| s.baseline.values()).map(|w| w.n).sum();
    let decay_obs: u64 = store.scopes.values().flat_map(|s| s.decay.values()).map(|d| d.margin.n).sum();
    let field_obs: u64 = store.scopes.values().flat_map(|s| s.field.values()).map(|f| f.seen).sum();
    let confusion_obs: u64 = store.scopes.values().flat_map(|s| s.confusion.values()).map(|c| c.ties).sum();
    // 🌟 [진단 보강] 기존 리포트는 baseline / decay / field / confusion 네 축만 셌습니다.
    //    그래서 '비전은 도는데 category·spatial 이 안 쌓인다', 'analytic 은 아예 0건이다'
    //    같은 배선 누락이 리포트만 봐서는 드러나지 않았습니다.
    //    (실측: 비전 태스크가 spatial 20건을 쌓았는데 리포트는 "베이스라인 5 감쇠 1 필드 2")
    //    기획 6-1 의 레코드군을 전부 세어, 리포트 한 줄로 배선 구멍을 특정할 수 있게 합니다.
    let category_obs: u64 = store.scopes.values().flat_map(|s| s.category.values()).map(|c| c.realized_max.n).sum();
    let spatial_obs: u64 = store.scopes.values().flat_map(|s| s.spatial.values()).map(|s| s.active_ratio.n).sum();
    let transition_obs: u64 = store.scopes.values().flat_map(|s| s.transition.values()).sum();
    let axis_obs: u64 = store.scopes.values().flat_map(|s| s.axis_variance.values()).map(|w| w.n).sum();
    format!(
        "[SDS REPORT] 스코프 {}개 | 베이스라인 {} | 감쇠 {} | 축분산 {} | 필드 {} | 혼동 {} | 카테고리 {} | 공간 {} | 전이 {} | (Phase 0: 판정 미개입)",
        scopes, baseline_obs, decay_obs, axis_obs, field_obs, confusion_obs,
        category_obs, spatial_obs, transition_obs
    )
}
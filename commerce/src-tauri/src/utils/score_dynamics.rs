use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::RwLock;

pub const SDS_RECIPE: &str = "sds-v3:welford+ring/run-files/ledger-split/presence/granite-384/bias-json";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Track {
    Vision,
    Trading,
    Commerce,
    Analytic,
    Search,
    Index,
}

impl Track {
    pub fn as_str(&self) -> &'static str {
        match self {
            Track::Vision => "vision",
            Track::Trading => "trading",
            Track::Commerce => "commerce",
            Track::Analytic => "analytic",
            Track::Search => "search",
            Track::Index => "index",
        }
    }
    pub fn from_name(name: &str) -> Option<Track> {
        match name {
            "vision" => Some(Track::Vision),
            "trading" => Some(Track::Trading),
            "commerce" => Some(Track::Commerce),
            "analytic" => Some(Track::Analytic),
            "search" => Some(Track::Search),
            "index" => Some(Track::Index),
            _ => None,
        }
    }
    pub fn min_obs(&self) -> (u64, u64, u64) {
        match self {
            Track::Vision => (20, 8, 30),
            Track::Trading => (30, 12, 40),
            Track::Commerce => (50, 20, 60),
            Track::Analytic => (40, 15, 0),
            Track::Search => (20, 8, 30),
            Track::Index => (20, 8, 30),
        }
    }
    pub fn ring_len(&self) -> usize {
        self.min_obs().1 as usize
    }
    pub fn ledger(&self) -> Ledger {
        match self {
            Track::Vision | Track::Trading | Track::Commerce => Ledger::Extract,
            Track::Analytic => Ledger::Analytic,
            Track::Search => Ledger::Search,
            Track::Index => Ledger::Index,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ledger {
    Extract,
    Search,
    Analytic,
    Index,
}

impl Ledger {
    pub fn as_str(&self) -> &'static str {
        match self {
            Ledger::Extract => "extract",
            Ledger::Search => "search",
            Ledger::Analytic => "analytic",
            Ledger::Index => "index",
        }
    }
    pub fn file_name(&self) -> &'static str {
        match self {
            Ledger::Extract => "extract.json",
            Ledger::Search => "search.json",
            Ledger::Analytic => "analytic.json",
            Ledger::Index => "index.json",
        }
    }
    pub fn all() -> [Ledger; 4] {
        [Ledger::Extract, Ledger::Search, Ledger::Analytic, Ledger::Index]
    }
    pub fn from_name(name: &str) -> Option<Ledger> {
        match name {
            "extract" => Some(Ledger::Extract),
            "search" => Some(Ledger::Search),
            "analytic" => Some(Ledger::Analytic),
            "index" => Some(Ledger::Index),
            _ => None,
        }
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

    let ledger = current_ledger();

    if let Ok(mut w) = ACTIVE_SCOPE.write() {
        w.primary = new_primary.clone();
    }

    if old_key == new_key { return; }
    let moved = {
        match RUNS.write() {
            Ok(mut runs) => match runs.get_mut(&ledger) {
                Some(run) => match run.scopes.remove(&old_key) {
                    Some(prev) => {
                        let has = !prev.baseline.is_empty()
                            || !prev.decay.is_empty()
                            || !prev.axis_variance.is_empty()
                            || !prev.field.is_empty()
                            || !prev.confusion.is_empty()
                            || !prev.category.is_empty()
                            || !prev.spatial.is_empty()
                            || !prev.transition.is_empty()
                            || !prev.search_field.is_empty();
                        if has {
                            let cnt = prev.baseline.len()
                                + prev.decay.len()
                                + prev.field.len()
                                + prev.confusion.len();
                            run.scopes
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
                None => 0,
            },
            Err(_) => 0,
        }
    };
    if moved > 0 {
        mark_dirty(ledger);
        println!(
            "[SDS] 🔀 스코프 정밀화: '{}' → '{}' | 이전 관측 {}축을 이번 실행 안에서 새 스코프로 이관했습니다. (같은 문서의 관측이므로 귀속이 정확합니다)",
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
        "search" => Track::Search,
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
    #[serde(default)]
    pub search_field: HashMap<String, SearchFieldStat>,
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
        for (k, v) in other.search_field {
            let f = self.search_field.entry(k).or_insert_with(SearchFieldStat::default);
            f.proposed += v.proposed;
            f.hard += v.hard;
            f.hint += v.hint;
            f.evaluated += v.evaluated;
            f.satisfied_any += v.satisfied_any;
            f.killed_all += v.killed_all;
            f.sole_blocker += v.sole_blocker;
            f.demoted += v.demoted;
        }
        self.updated_at = chrono::Utc::now().timestamp_millis();
    }
}

static TEAM: Lazy<RwLock<String>> = Lazy::new(|| RwLock::new(String::new()));
static BASES: Lazy<RwLock<HashMap<Ledger, SdsFile>>> = Lazy::new(|| RwLock::new(HashMap::new()));
static RUNS: Lazy<RwLock<HashMap<Ledger, RunBuf>>> = Lazy::new(|| RwLock::new(HashMap::new()));
static DIRTY: Lazy<RwLock<HashMap<Ledger, bool>>> = Lazy::new(|| RwLock::new(HashMap::new()));
static INDEXING_DEPTH: Lazy<RwLock<u32>> = Lazy::new(|| RwLock::new(0));

#[derive(Debug, Clone, Default)]
struct RunBuf {
    stamp: String,
    label: String,
    file: String,
    started_ms: i64,
    closing: bool,
    observations: u64,
    scopes: HashMap<String, ScopeStat>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct RunFile {
    recipe: String,
    team: String,
    ledger: String,
    label: String,
    started_at: i64,
    updated_at: i64,
    observations: u64,
    #[serde(default)]
    scopes: HashMap<String, ScopeStat>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PresenceDoc {
    pub doc_type: String,
    pub fields: Vec<String>,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PresenceIndex {
    pub recipe: String,
    pub team: String,
    pub updated_at: i64,
    #[serde(default)]
    pub docs: HashMap<String, PresenceDoc>,
}

static PRESENCE: Lazy<RwLock<PresenceIndex>> = Lazy::new(|| RwLock::new(PresenceIndex::default()));
static PRESENCE_DIRTY: Lazy<RwLock<bool>> = Lazy::new(|| RwLock::new(false));

fn sds_dir() -> std::path::PathBuf {
    crate::utils::get_app_dir().join("score_dynamics")
}

fn legacy_sds_path() -> std::path::PathBuf {
    crate::utils::get_app_dir().join("score_dynamics.json")
}

fn ledger_path(l: Ledger) -> std::path::PathBuf {
    sds_dir().join(l.file_name())
}

fn runs_dir(l: Ledger) -> std::path::PathBuf {
    sds_dir().join("runs").join(l.as_str())
}

fn presence_path() -> std::path::PathBuf {
    sds_dir().join("presence.json")
}

fn mark_dirty(l: Ledger) {
    if let Ok(mut d) = DIRTY.write() { d.insert(l, true); }
}

fn is_dirty(l: Ledger) -> bool {
    DIRTY.read().map(|d| *d.get(&l).unwrap_or(&false)).unwrap_or(false)
}

pub struct IndexingGuard;

impl Drop for IndexingGuard {
    fn drop(&mut self) {
        if let Ok(mut d) = INDEXING_DEPTH.write() {
            *d = d.saturating_sub(1);
        }
    }
}

pub fn indexing_guard() -> IndexingGuard {
    if let Ok(mut d) = INDEXING_DEPTH.write() { *d = d.saturating_add(1); }
    IndexingGuard
}

fn indexing_active() -> bool {
    INDEXING_DEPTH.read().map(|d| *d > 0).unwrap_or(false)
}

fn current_ledger() -> Ledger {
    if indexing_active() { return Ledger::Index; }
    match current_track() {
        Some(t) => t.ledger(),
        None => Ledger::Extract,
    }
}

fn ledger_for_axis(axis: &str) -> Ledger {
    let head = axis.split('.').next().unwrap_or("");
    match Track::from_name(head) {
        Some(Track::Index) => Ledger::Index,
        Some(t) => {
            let by_track = t.ledger();
            if indexing_active() && by_track != Ledger::Index {
                return Ledger::Index;
            }
            by_track
        }
        None => match head {
            "indexing" => Ledger::Index,
            "extract" => Ledger::Extract,
            _ => current_ledger(),
        },
    }
}

fn ring_for_key(key: &str) -> usize {
    let head = key.split('|').next().unwrap_or("");
    match Track::from_name(head) {
        Some(t) => t.ring_len(),
        None => 12,
    }
}

fn sanitize_label(raw: &str) -> String {
    let mut out = String::new();
    for ch in raw.trim().chars() {
        if out.chars().count() >= 48 { break; }
        if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
            out.push(ch);
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    let t = out.trim_matches('_').to_string();
    if t.is_empty() { "run".to_string() } else { t }
}

fn unique_run_file(l: Ledger, stamp: &str, label: &str) -> String {
    let dir = runs_dir(l);
    let stem = format!("{}_{}", stamp, label);
    let first = format!("{}.json", stem);
    if !dir.join(&first).exists() { return first; }
    for i in 2..1000u32 {
        let c = format!("{}-{}.json", stem, i);
        if !dir.join(&c).exists() { return c; }
    }
    format!("{}-{}.json", stem, chrono::Utc::now().timestamp_millis())
}

fn fresh_run(l: Ledger, label: &str) -> RunBuf {
    let now = chrono::Local::now();
    let stamp = now.format("%Y%m%d-%H%M%S").to_string();
    let file = unique_run_file(l, &stamp, label);
    RunBuf {
        stamp,
        label: label.to_string(),
        file,
        started_ms: now.timestamp_millis(),
        closing: false,
        observations: 0,
        scopes: HashMap::new(),
    }
}

fn open_run<'a>(runs: &'a mut HashMap<Ledger, RunBuf>, l: Ledger, label: &str) -> &'a mut RunBuf {
    let run = runs.entry(l).or_insert_with(|| fresh_run(l, label));
    if run.closing { *run = fresh_run(l, label); }
    run
}

pub fn set_run_label(label: &str) {
    let l = sanitize_label(label);
    let ledger = current_ledger();
    if let Ok(mut runs) = RUNS.write() {
        let run = open_run(&mut runs, ledger, &l);
        if run.label != l {
            run.label = l.clone();
            run.file = unique_run_file(ledger, &run.stamp, &l);
        }
    }
}

pub fn run_stamp() -> String {
    let ledger = current_ledger();
    RUNS.read()
        .ok()
        .and_then(|r| r.get(&ledger).map(|x| x.stamp.clone()))
        .unwrap_or_default()
}

fn merged_scope(l: Ledger, key: &str) -> Option<ScopeStat> {
    let base = BASES.read().ok()?.get(&l).and_then(|f| f.scopes.get(key).cloned());
    let run = RUNS.read().ok()?.get(&l).and_then(|r| r.scopes.get(key).cloned());
    match (base, run) {
        (Some(mut b), Some(r)) => { b.absorb(r, ring_for_key(key)); Some(b) }
        (Some(b), None) => Some(b),
        (None, Some(r)) => Some(r),
        (None, None) => None,
    }
}

fn merged_file(l: Ledger) -> SdsFile {
    let team = TEAM.read().map(|t| t.clone()).unwrap_or_default();
    let mut out = BASES
        .read()
        .ok()
        .and_then(|b| b.get(&l).cloned())
        .unwrap_or_else(|| SdsFile {
            recipe: SDS_RECIPE.to_string(),
            team: team.clone(),
            updated_at: 0,
            scopes: HashMap::new(),
        });
    out.recipe = SDS_RECIPE.to_string();
    out.team = team;
    if let Ok(runs) = RUNS.read() {
        if let Some(run) = runs.get(&l) {
            for (k, v) in run.scopes.iter() {
                out.scopes
                    .entry(k.clone())
                    .or_insert_with(ScopeStat::default)
                    .absorb(v.clone(), ring_for_key(k));
            }
        }
    }
    out.updated_at = chrono::Utc::now().timestamp_millis();
    out
}

pub fn load(team: &str) {
    if let Ok(mut w) = TEAM.write() { *w = team.to_string(); }
    let dir = sds_dir();
    let _ = std::fs::create_dir_all(&dir);
    archive_legacy_ledger(team);

    let mut loaded = 0usize;
    let mut scopes_total = 0usize;
    let mut bases: HashMap<Ledger, SdsFile> = HashMap::new();
    for l in Ledger::all() {
        let path = ledger_path(l);
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
                                "[SDS] 🧹 [{}] 레시피 세대 불일치 (저장 '{}' vs 현재 '{}'). 이 원장을 폐기하고 새로 시작합니다.",
                                l.as_str(), f.recipe, SDS_RECIPE
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
                                "[SDS] 🚚 [TEAM MIGRATE / {}] 로컬 기본 팀 '{}' 의 스코프 {}개를 실제 팀 '{}' 로 이관했습니다.",
                                l.as_str(), prev_team, n, team
                            );
                        } else if !team.is_empty() && !f.team.is_empty() && f.team != team {
                            println!(
                                "[SDS] 🧹 [{}] 팀 불일치 (저장 '{}' vs 현재 '{}'). 이 원장을 폐기합니다. (스코프 격리 원칙)",
                                l.as_str(), f.team, team
                            );
                        } else {
                            scopes_total += f.scopes.len();
                            loaded += 1;
                            fresh = f;
                            fresh.team = team.to_string();
                        }
                    }
                    Err(e) => println!("[SDS] ⚠️ [{}] 원장 파싱 실패({}). 새로 시작합니다.", l.as_str(), e),
                },
                Err(e) => println!("[SDS] ⚠️ [{}] 원장 읽기 실패({}). 새로 시작합니다.", l.as_str(), e),
            }
        }
        bases.insert(l, fresh);
    }
    if let Ok(mut w) = BASES.write() { *w = bases; }
    if let Ok(mut w) = RUNS.write() { w.clear(); }
    if let Ok(mut d) = DIRTY.write() { d.clear(); }
    presence_load(team);

    if loaded == 0 {
        println!("[SDS] 🆕 저장된 원장이 없습니다. 냉간 시작합니다. (현행 판정 그대로)");
    } else {
        println!(
            "[SDS] ✅ 원장 {}종을 불러왔습니다. 스코프 {}개 | 저장 보유 문서 {}건.",
            loaded,
            scopes_total,
            PRESENCE.read().map(|p| p.docs.len()).unwrap_or(0)
        );
    }
    for l in Ledger::all() { mark_dirty(l); }
    flush();
    println!("[SDS] 📍 원장 디렉터리: {}", sds_dir().display());
    println!("[SDS] 📍 실행별 파일: {}", sds_dir().join("runs").join("<종류>").join("<시작시각>_<라벨>.json").display());
}

fn archive_legacy_ledger(team: &str) {
    let legacy = legacy_sds_path();
    if !legacy.exists() { return; }
    let archive_dir = sds_dir().join("legacy");
    let _ = std::fs::create_dir_all(&archive_dir);
    let stamp = chrono::Local::now().format("%Y%m%d-%H%M%S").to_string();
    let target = archive_dir.join(format!("score_dynamics.{}.json", stamp));
    match std::fs::rename(&legacy, &target) {
        Ok(_) => println!(
            "[SDS] 📦 [LEGACY ARCHIVE] 단일 누적 파일을 '{}' 로 보관했습니다. 전처리·인덱싱·검색 관측이 한 스코프에 섞여 있어 새 원장에 합산하지 않습니다. (팀 '{}')",
            target.display(), team
        ),
        Err(e) => println!("[SDS] ⚠️ [LEGACY ARCHIVE] 단일 누적 파일 보관 실패({}). 그대로 두고 진행합니다.", e),
    }
}

fn presence_load(team: &str) {
    let path = presence_path();
    let mut fresh = PresenceIndex {
        recipe: SDS_RECIPE.to_string(),
        team: team.to_string(),
        updated_at: chrono::Utc::now().timestamp_millis(),
        docs: HashMap::new(),
    };
    if path.exists() {
        if let Ok(txt) = std::fs::read_to_string(&path) {
            if let Ok(p) = serde_json::from_str::<PresenceIndex>(&txt) {
                let team_ok = team.is_empty()
                    || p.team.is_empty()
                    || p.team == team
                    || p.team == local_default_team();
                if p.recipe == SDS_RECIPE && team_ok {
                    fresh = p;
                    fresh.team = team.to_string();
                } else {
                    println!("[SDS] 🧹 [PRESENCE] 세대 또는 팀 불일치로 저장 보유 현황을 새로 시작합니다.");
                }
            }
        }
    }
    if let Ok(mut w) = PRESENCE.write() { *w = fresh; }
    if let Ok(mut d) = PRESENCE_DIRTY.write() { *d = true; }
}

pub fn presence_record(doc_type: &str, id: &str, fields: &[String]) {
    let t = doc_type.trim().to_lowercase();
    let key = id.trim().to_string();
    if t.is_empty() || key.is_empty() { return; }
    let mut sorted: Vec<String> = Vec::with_capacity(fields.len());
    for f in fields.iter() {
        let n = f.trim().to_string();
        if n.is_empty() { continue; }
        if !sorted.iter().any(|x| *x == n) { sorted.push(n); }
    }
    sorted.sort();
    let now = chrono::Utc::now().timestamp_millis();
    let changed = match PRESENCE.write() {
        Ok(mut p) => {
            let prev = p.docs.insert(
                key,
                PresenceDoc { doc_type: t, fields: sorted.clone(), updated_at: now },
            );
            match prev {
                Some(old) => old.fields != sorted,
                None => true,
            }
        }
        Err(_) => false,
    };
    if changed {
        if let Ok(mut d) = PRESENCE_DIRTY.write() { *d = true; }
    }
}

pub fn presence_forget(id: &str) {
    let key = id.trim();
    if key.is_empty() { return; }
    let removed = PRESENCE.write().map(|mut p| p.docs.remove(key).is_some()).unwrap_or(false);
    if removed {
        if let Ok(mut d) = PRESENCE_DIRTY.write() { *d = true; }
    }
}

fn presence_flush() {
    let dirty = PRESENCE_DIRTY.read().map(|d| *d).unwrap_or(false);
    if !dirty { return; }
    let snapshot = match PRESENCE.read() { Ok(p) => p.clone(), Err(_) => return };
    let mut snapshot = snapshot;
    snapshot.recipe = SDS_RECIPE.to_string();
    snapshot.updated_at = chrono::Utc::now().timestamp_millis();
    let txt = match serde_json::to_string_pretty(&snapshot) { Ok(t) => t, Err(_) => return };
    let path = presence_path();
    if let Some(dir) = path.parent() { let _ = std::fs::create_dir_all(dir); }
    if std::fs::write(&path, txt.as_bytes()).is_ok() {
        if let Ok(mut d) = PRESENCE_DIRTY.write() { *d = false; }
    }
}

fn local_default_team() -> String {
    crate::utils::hash::hash_id("0x0000000000000000000000000000000000000000")
}

pub fn rebind_team(team: &str) {
    let prev = TEAM.read().map(|t| t.clone()).unwrap_or_default();
    if prev.is_empty() || prev == team {
        if let Ok(mut w) = TEAM.write() { *w = team.to_string(); }
        if let Ok(mut b) = BASES.write() {
            for f in b.values_mut() { f.team = team.to_string(); }
        }
        if let Ok(mut p) = PRESENCE.write() { p.team = team.to_string(); }
        return;
    }
    if prev == local_default_team() {
        let scopes: usize = BASES.read().map(|b| b.values().map(|f| f.scopes.len()).sum()).unwrap_or(0);
        let docs = PRESENCE.read().map(|p| p.docs.len()).unwrap_or(0);
        if let Ok(mut w) = TEAM.write() { *w = team.to_string(); }
        if let Ok(mut b) = BASES.write() {
            for f in b.values_mut() { f.team = team.to_string(); }
        }
        if let Ok(mut p) = PRESENCE.write() { p.team = team.to_string(); }
        for l in Ledger::all() { mark_dirty(l); }
        if let Ok(mut d) = PRESENCE_DIRTY.write() { *d = true; }
        flush();
        println!(
            "[SDS] 🚚 [TEAM MIGRATE] 로컬 기본 팀의 스코프 {}개 · 저장 보유 문서 {}건을 실제 팀 '{}' 로 이관했습니다.",
            scopes, docs, team
        );
        if scopes == 0 {
            println!(
                "[SDS] ⚠️ [TEAM MIGRATE] 이관 시점에 메모리 스코프가 0개였습니다. load() 가 이미 폐기했을 가능성이 있으니 원장의 team 값을 확인하십시오."
            );
        }
        return;
    }
    println!("[SDS] 🔄 팀 전환 감지. 이전 팀 통계를 폐기하고 새 팀으로 재바인딩합니다.");
    purge();
    load(team);
}

pub fn flush() {
    presence_flush();
    let open: Vec<Ledger> = RUNS.read().map(|r| r.keys().cloned().collect()).unwrap_or_default();
    let mut targets: Vec<Ledger> = Vec::new();
    for l in Ledger::all() {
        if is_dirty(l) || open.contains(&l) { targets.push(l); }
    }
    if targets.is_empty() {
        println!("[SDS] ⏭️ 새 관측이 없어 저장을 건너뜁니다. (열린 실행 0건)");
        return;
    }

    let team = TEAM.read().map(|t| t.clone()).unwrap_or_default();
    let mut finalize: Vec<Ledger> = Vec::new();
    let mut lines: Vec<String> = Vec::new();

    for l in targets {
        let run_meta = RUNS.read().ok().and_then(|r| r.get(&l).cloned());
        if let Some(run) = run_meta.as_ref() {
            let rf = RunFile {
                recipe: SDS_RECIPE.to_string(),
                team: team.clone(),
                ledger: l.as_str().to_string(),
                label: run.label.clone(),
                started_at: run.started_ms,
                updated_at: chrono::Utc::now().timestamp_millis(),
                observations: run.observations,
                scopes: run.scopes.clone(),
            };
            if let Ok(txt) = serde_json::to_string_pretty(&rf) {
                let dir = runs_dir(l);
                let _ = std::fs::create_dir_all(&dir);
                let file = dir.join(&run.file);
                match std::fs::write(&file, txt.as_bytes()) {
                    Ok(_) => lines.push(format!(
                        "{}={}건→{}",
                        l.as_str(),
                        run.observations,
                        file.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default()
                    )),
                    Err(e) => println!("[SDS] ⚠️ [{}] 실행 파일 저장 실패: {}", l.as_str(), e),
                }
            }
            if run.closing { finalize.push(l); }
        }

        let snapshot = merged_file(l);
        let mut txt = match serde_json::to_string_pretty(&snapshot) { Ok(t) => t, Err(_) => continue };
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
            keys.sort_by(|a, b| a.1.cmp(&b.1).then(a.2.cmp(&b.2)));
            let mut trimmed = snapshot.clone();
            for (k, _, _) in keys {
                if txt.len() <= CAP_BYTES { break; }
                trimmed.scopes.remove(&k);
                txt = serde_json::to_string_pretty(&trimmed).unwrap_or(txt);
            }
            println!("[SDS] ✂️ [{}] 상한 초과로 저관측 스코프를 절삭했습니다. (잔존 {}개)", l.as_str(), trimmed.scopes.len());
        }

        let path = ledger_path(l);
        if let Some(dir) = path.parent() { let _ = std::fs::create_dir_all(dir); }
        match std::fs::write(&path, txt.as_bytes()) {
            Ok(_) => {
                if let Ok(mut d) = DIRTY.write() { d.insert(l, false); }
                lines.push(format!("{}.json={}바이트", l.as_str(), txt.len()));
            }
            Err(e) => println!("[SDS] ⚠️ [{}] 원장 저장 실패: {}", l.as_str(), e),
        }
    }

    for l in finalize {
        let run = RUNS.write().ok().and_then(|mut r| r.remove(&l));
        if let Some(run) = run {
            if let Ok(mut b) = BASES.write() {
                let base = b.entry(l).or_insert_with(|| SdsFile {
                    recipe: SDS_RECIPE.to_string(),
                    team: team.clone(),
                    updated_at: 0,
                    scopes: HashMap::new(),
                });
                for (k, v) in run.scopes.into_iter() {
                    let ring = ring_for_key(&k);
                    base.scopes.entry(k).or_insert_with(ScopeStat::default).absorb(v, ring);
                }
                base.updated_at = chrono::Utc::now().timestamp_millis();
            }
            println!(
                "[SDS] 🧾 [RUN CLOSED] {} | '{}' 관측 {}건을 원장에 합산했습니다. 다음 관측부터 새 실행 파일이 만들어집니다.",
                l.as_str(), run.file, run.observations
            );
        }
    }

    if let Ok(mut t) = WRITE_TICK.write() { *t = 0; }
    if !lines.is_empty() {
        println!("[SDS] 💾 저장 완료: {}", lines.join(" | "));
    }
}

pub fn purge() {
    let team = TEAM.read().map(|t| t.clone()).unwrap_or_default();
    if let Ok(mut b) = BASES.write() { b.clear(); }
    if let Ok(mut r) = RUNS.write() { r.clear(); }
    if let Ok(mut d) = DIRTY.write() { d.clear(); }
    if let Ok(mut p) = PRESENCE.write() {
        *p = PresenceIndex {
            recipe: SDS_RECIPE.to_string(),
            team: team.clone(),
            updated_at: chrono::Utc::now().timestamp_millis(),
            docs: HashMap::new(),
        };
    }
    if let Ok(mut d) = PRESENCE_DIRTY.write() { *d = false; }
    let _ = std::fs::remove_dir_all(sds_dir());
    let _ = std::fs::remove_file(legacy_sds_path());
    println!("[SDS] 🗑️ 점수 동역학 원장·실행 파일·저장 보유 현황을 전량 삭제했습니다. 판정은 즉시 현행 상수로 복귀합니다.");
}

fn with_scope_mut<F: FnOnce(&mut ScopeStat, usize)>(f: F) {
    with_scope_mut_in(current_ledger(), f);
}

fn with_scope_mut_in<F: FnOnce(&mut ScopeStat, usize)>(ledger: Ledger, f: F) {
    let (key, ring) = effective_scope_key();
    if let Ok(mut runs) = RUNS.write() {
        let run = open_run(&mut runs, ledger, ledger.as_str());
        run.observations = run.observations.saturating_add(1);
        let e = run.scopes.entry(key).or_insert_with(ScopeStat::default);
        f(e, ring);
        e.updated_at = chrono::Utc::now().timestamp_millis();
    }
    mark_dirty(ledger);
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
    with_scope_mut_in(ledger_for_axis(axis), |s, ring| {
        s.baseline
            .entry(axis.to_string())
            .or_insert_with(Welford::default)
            .push(value as f64, ring);
    });
}

/// 순위 감쇠 곡선. 후보 점수 배열 전체를 넘기면 형상만 압축해 저장합니다.
pub fn record_decay(axis: &str, scores: &[f32]) {
    let shape = match decay_shape(scores) { Some(s) => s, None => return };
    let ledger = ledger_for_axis(axis);
    with_scope_mut_in(ledger, |s, ring| {
        let d = s.decay.entry(axis.to_string()).or_insert_with(DecayStat::default);
        d.margin.push(shape.margin, ring);
        d.top_gap_ratio.push(shape.top_gap_ratio, ring);
        d.tail_flatness.push(shape.tail_flatness, ring);
        d.entropy_norm.push(shape.entropy_norm, ring);
        d.positive_ratio.push(shape.positive_ratio, ring);
    });
    let var = {
        let v: Vec<f64> = scores.iter().filter(|x| x.is_finite()).map(|x| *x as f64).collect();
        if v.len() < 2 { 0.0 } else {
            let m = v.iter().sum::<f64>() / (v.len() as f64);
            v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / ((v.len() - 1) as f64)
        }
    };
    if var > 0.0 {
        with_scope_mut_in(ledger, |s, ring| {
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
    let ledger = current_ledger();
    let (m2, m1, mg) = track.min_obs();
    for (key, need) in [
        (scope.key_secondary(), m2),
        (scope.key_primary(), m1),
        (scope.key_global(), mg),
        ("unscoped||".to_string(), mg.max(m1)),
    ] {
        if need == 0 { continue; }
        if let Some(st) = merged_scope(ledger, &key) {
            if let Some((val, n)) = pick(&st) {
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
    let ledger = current_ledger();
    let (m2, m1, _) = track.min_obs();
    for (key, need) in [(scope.key_secondary(), m2), (scope.key_primary(), m1)] {
        let st = match merged_scope(ledger, &key) { Some(v) => v, None => continue };
        let total: u64 = st
            .transition
            .iter()
            .filter(|(k, _)| k.starts_with(&format!("{}>", from)))
            .map(|(_, v)| *v)
            .sum();
        if total < need { continue; }
        let hit = st.transition.get(&format!("{}>{}", from, to)).copied().unwrap_or(0);
        let k = st.transition.len().max(1) as f64;
        return Some((((hit as f64) + 1.0) / ((total as f64) + k)) as f32);
    }
    None
}

// =====================================================================
// 🌟 [진단] 현재 축적 상태 한 줄 요약. 태스크 종료 시 로그로 남깁니다.
// =====================================================================
pub fn report() -> String {
    let mut closed: Vec<String> = Vec::new();
    if let Ok(mut runs) = RUNS.write() {
        for (l, run) in runs.iter_mut() {
            if run.observations == 0 { continue; }
            run.closing = true;
            closed.push(format!("{}:{}건", l.as_str(), run.observations));
        }
    }
    let mut parts: Vec<String> = Vec::new();
    let mut docs_total = 0usize;
    for l in Ledger::all() {
        let f = merged_file(l);
        if f.scopes.is_empty() { continue; }
        let baseline_obs: u64 = f.scopes.values().flat_map(|s| s.baseline.values()).map(|w| w.n).sum();
        let decay_obs: u64 = f.scopes.values().flat_map(|s| s.decay.values()).map(|d| d.margin.n).sum();
        let field_obs: u64 = f.scopes.values().flat_map(|s| s.field.values()).map(|x| x.seen).sum();
        let confusion_obs: u64 = f.scopes.values().flat_map(|s| s.confusion.values()).map(|c| c.ties).sum();
        let category_obs: u64 = f.scopes.values().flat_map(|s| s.category.values()).map(|c| c.realized_max.n).sum();
        let spatial_obs: u64 = f.scopes.values().flat_map(|s| s.spatial.values()).map(|s| s.active_ratio.n).sum();
        let transition_obs: u64 = f.scopes.values().flat_map(|s| s.transition.values()).sum();
        let axis_obs: u64 = f.scopes.values().flat_map(|s| s.axis_variance.values()).map(|w| w.n).sum();
        let search_obs: u64 = f.scopes.values().flat_map(|s| s.search_field.values()).map(|x| x.proposed).sum();
        let search_eval: u64 = f.scopes.values().flat_map(|s| s.search_field.values()).map(|x| x.evaluated).sum();
        docs_total = PRESENCE.read().map(|p| p.docs.len()).unwrap_or(0);
        parts.push(format!(
            "{}(스코프 {} | 베이스라인 {} | 감쇠 {} | 축분산 {} | 필드 {} | 혼동 {} | 카테고리 {} | 공간 {} | 전이 {} | 검색조건 {} | 검색결과평가 {})",
            l.as_str(), f.scopes.len(), baseline_obs, decay_obs, axis_obs, field_obs,
            confusion_obs, category_obs, spatial_obs, transition_obs, search_obs, search_eval
        ));
    }
    if parts.is_empty() { parts.push("관측 없음".to_string()); }
    format!(
        "[SDS REPORT] {} | 저장 보유 문서 {}건 | 닫는 실행 [{}] | (Phase 0: 판정 미개입)",
        parts.join(" | "),
        docs_total,
        if closed.is_empty() { "-".to_string() } else { closed.join(", ") }
    )
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFieldStat {
    pub proposed: u64,
    pub hard: u64,
    pub hint: u64,
    pub evaluated: u64,
    pub satisfied_any: u64,
    pub killed_all: u64,
    pub sole_blocker: u64,
    pub demoted: u64,
}

pub fn search_scope_key(types: &[String]) -> String {
    let mut codes: Vec<String> = Vec::new();
    for t in types.iter() {
        let c = t.trim().to_lowercase();
        if c.is_empty() { continue; }
        if !codes.contains(&c) { codes.push(c); }
    }
    codes.sort();
    if codes.is_empty() || codes.len() > Track::Search.ring_len() {
        return "all".to_string();
    }
    codes.join("+")
}

pub fn record_search_proposal(field: &str, hard: bool) {
    if field.is_empty() { return; }
    with_scope_mut(|s, _| {
        let e = s.search_field.entry(field.to_string()).or_insert_with(SearchFieldStat::default);
        e.proposed += 1;
        if hard { e.hard += 1; } else { e.hint += 1; }
    });
}

pub fn record_search_outcome(field: &str, satisfied_any: bool, sole_blocker: bool) {
    if field.is_empty() { return; }
    with_scope_mut(|s, _| {
        let e = s.search_field.entry(field.to_string()).or_insert_with(SearchFieldStat::default);
        e.evaluated += 1;
        if satisfied_any { e.satisfied_any += 1; } else { e.killed_all += 1; }
        if sole_blocker { e.sole_blocker += 1; }
    });
}

pub fn record_search_demotion(field: &str) {
    if field.is_empty() { return; }
    with_scope_mut(|s, _| {
        s.search_field.entry(field.to_string()).or_insert_with(SearchFieldStat::default).demoted += 1;
    });
}

pub fn search_kill_rate(field: &str) -> Option<f32> {
    resolve(|st| {
        st.search_field.get(field).map(|f| {
            ((f.killed_all as f64 / f.evaluated.max(1) as f64) as f32, f.evaluated)
        })
    })
}

pub fn storage_fill_prior(doc_types: &[String], field: &str) -> Option<(f32, u64, u64)> {
    let wanted: Vec<String> = doc_types
        .iter()
        .map(|t| t.trim().to_lowercase())
        .filter(|t| !t.is_empty())
        .collect();
    if wanted.is_empty() || field.is_empty() { return None; }
    let p = PRESENCE.read().ok()?;
    let mut docs = 0u64;
    let mut filled = 0u64;
    for d in p.docs.values() {
        if !wanted.iter().any(|t| *t == d.doc_type) { continue; }
        docs += 1;
        if d.fields.iter().any(|f| f == field) { filled += 1; }
    }
    if docs < Track::Vision.min_obs().1 { return None; }
    Some((((filled as f64 + 0.5) / (docs as f64 + 1.0)) as f32, filled, docs))
}

pub fn storage_doc_count(doc_types: &[String]) -> u64 {
    let wanted: Vec<String> = doc_types
        .iter()
        .map(|t| t.trim().to_lowercase())
        .filter(|t| !t.is_empty())
        .collect();
    if wanted.is_empty() { return 0; }
    PRESENCE
        .read()
        .map(|p| p.docs.values().filter(|d| wanted.iter().any(|t| *t == d.doc_type)).count() as u64)
        .unwrap_or(0)
}
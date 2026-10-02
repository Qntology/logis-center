use scraper::{ElementRef, Html, Selector};
use std::collections::{HashMap, HashSet};

const SKIP_CLASSES: [&str; 8] = ["active", "selected", "on", "current", "focus", "hover", "enabled", "disabled"];
const LANDMARK_TAGS: [&str; 4] = ["nav", "header", "footer", "aside"];
const LANDMARK_ROLES: [&str; 6] = ["navigation", "menubar", "menu", "banner", "contentinfo", "tablist"];
const NON_ITEM_TAGS: [&str; 12] = ["script", "style", "option", "br", "hr", "input", "meta", "link", "col", "colgroup", "source", "param"];
const CHROME_TEXT: &str = "global navigation, menus, header, footer, sidebar, breadcrumb, admin main menu, main menu, admin page, administrator page, dashboard, control panel, site name, shopping mall, welcome, home, index, basic search, search form, search filter, login, logout, notice, banner, copyright, pagination, top menu, quick menu, sub menu, side navigation, left menu, category menu, management menu, settings menu, configuration menu, quick links";
const DATA_MIN_CELLS: f32 = 3.0;
const DATA_MIN_DISTINCT: f32 = 0.5;
const MIRROR_MIN: f32 = 0.8;
const MIRROR_MAX_COPIES: usize = 3;
const MIRROR_SAMPLE: usize = 12;
const FALLBACK_MIN_ROWS: usize = 3;
const FALLBACK_MIN_CELLS: f32 = 4.0;
const FALLBACK_DOMINANCE: f32 = 2.0;
const MAX_KEPT: usize = 12;
const EMPTY_MAX_BODY_ROWS: usize = 2;
const SAMPLE_CHARS: usize = 160;
const MATCH_CHARS: usize = 600;
const EXCERPT_ROWS: usize = 6;
const EXCERPT_ROW_LINES: usize = 40;
const EXCERPT_LINE_CHARS: usize = 120;
const EXCERPT_HEADER_CELLS: usize = 40;

#[derive(Debug, Clone)]
pub struct CensusGroup {
    pub parent_sig: String,
    pub item_selector: String,
    pub tag: String,
    pub members: usize,
    pub avg_cells: f32,
    pub distinct_ratio: f32,
    pub form_like: bool,
    pub exact: bool,
    pub score: f32,
    pub mirror: f32,
    pub content_margin: Option<f32>,
    pub samples: Vec<String>,
    member_order: Vec<usize>,
    member_texts: Vec<String>,
    ranges: Vec<(usize, usize)>,
}

impl CensusGroup {
    pub fn structural_grade(&self) -> bool {
        self.members >= 2
            && self.avg_cells >= DATA_MIN_CELLS
            && self.distinct_ratio >= DATA_MIN_DISTINCT
            && !self.form_like
            && !self.mirror_copy()
    }

    pub fn mirror_copy(&self) -> bool {
        self.members <= MIRROR_MAX_COPIES && self.mirror >= MIRROR_MIN
    }

    pub fn data_grade(&self) -> bool {
        self.structural_grade() && self.content_margin.map_or(true, |m| m > 0.0)
    }

    pub fn selector_json(&self) -> serde_json::Value {
        serde_json::json!({
            "parent": self.parent_sig,
            "itemSelector": self.item_selector,
            "matchCount": self.members,
        })
    }

    fn envelope(&self) -> (usize, usize) {
        let lo = self.ranges.iter().map(|r| r.0).min().unwrap_or(0);
        let hi = self.ranges.iter().map(|r| r.1).max().unwrap_or(0);
        (lo, hi)
    }
}

#[derive(Debug, Clone)]
pub struct EmptyTable {
    pub selector: String,
    pub header_cells: usize,
    pub body_rows: usize,
    pub notice: String,
}

#[derive(Debug, Clone, Default)]
pub struct ListCensus {
    pub groups: Vec<CensusGroup>,
    pub suppressed: usize,
    pub landmark_dropped: usize,
    pub empty_tables: Vec<EmptyTable>,
}

struct Probe {
    img: Selector,
    editable: Selector,
    all: Selector,
    table: Selector,
    tr: Selector,
}

impl Probe {
    fn new() -> Option<Self> {
        Some(Self {
            img: Selector::parse("img").ok()?,
            editable: Selector::parse("select, textarea, input[type=text], input[type=search], input[type=date], input[type=number], input[type=email], input[type=tel], input:not([type])").ok()?,
            all: Selector::parse("*").ok()?,
            table: Selector::parse("table").ok()?,
            tr: Selector::parse("tr").ok()?,
        })
    }
}

fn css_ident_ok(s: &str) -> bool {
    let mut it = s.chars();
    match it.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' || c == '-' => {}
        _ => return false,
    }
    s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}

fn is_hash_like(c: &str) -> bool {
    c.len() >= 8 && c.chars().all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit())
}

fn clean_classes(el: &ElementRef, strip_numbers: bool) -> Vec<String> {
    let mut v: Vec<String> = el
        .value()
        .classes()
        .filter(|c| {
            let l = c.to_lowercase();
            !SKIP_CLASSES.contains(&l.as_str()) && !c.contains("__") && !is_hash_like(c) && css_ident_ok(c)
        })
        .map(|c| {
            if strip_numbers {
                c.trim_end_matches(|ch: char| ch.is_ascii_digit()).to_string()
            } else {
                c.to_string()
            }
        })
        .filter(|c| !c.is_empty())
        .collect();
    v.sort();
    v.dedup();
    v
}

fn signature(el: &ElementRef, include_id: bool) -> String {
    let mut s = el.value().name().to_lowercase();
    if include_id {
        if let Some(id) = el.value().id() {
            if css_ident_ok(id) {
                s.push('#');
                s.push_str(id);
            }
        }
    }
    let cls = clean_classes(el, false);
    if !cls.is_empty() {
        s.push('.');
        s.push_str(&cls.join("."));
    }
    s
}

fn element_children<'a>(el: &ElementRef<'a>) -> Vec<ElementRef<'a>> {
    el.children().filter_map(ElementRef::wrap).collect()
}

fn attr_span(el: &ElementRef, name: &str) -> usize {
    el.value()
        .attr(name)
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(1)
        .max(1)
}

fn row_cells<'a>(tr: &ElementRef<'a>) -> Vec<ElementRef<'a>> {
    element_children(tr)
        .into_iter()
        .filter(|c| matches!(c.value().name(), "td" | "th"))
        .collect()
}

fn colspan_sum(tr: &ElementRef) -> usize {
    row_cells(tr).iter().map(|c| attr_span(c, "colspan")).sum()
}

fn similarity(a: &ElementRef, b: &ElementRef) -> f32 {
    let (na, nb) = (a.value().name(), b.value().name());
    if na != nb {
        return 0.0;
    }
    if na == "tr" {
        let (ca, cb) = (colspan_sum(a), colspan_sum(b));
        if ca > 0 && cb > 0 && ca.abs_diff(cb) > 1 {
            return 0.0;
        }
    }
    if na == "td" || na == "th" {
        if attr_span(a, "colspan") != attr_span(b, "colspan") || attr_span(a, "rowspan") != attr_span(b, "rowspan") {
            return 0.0;
        }
    }
    let ca = clean_classes(a, true);
    let cb = clean_classes(b, true);
    if ca.is_empty() && cb.is_empty() {
        return 100.0;
    }
    if ca.is_empty() {
        return 0.0;
    }
    let m = ca.iter().filter(|c| cb.contains(c)).count();
    m as f32 / ca.len() as f32 * 100.0
}

fn norm_text(el: &ElementRef, cap: usize) -> String {
    let joined = el.text().collect::<Vec<_>>().join(" ");
    let t = joined.split_whitespace().collect::<Vec<_>>().join(" ");
    t.chars().take(cap).collect()
}

fn has_own_text(el: &ElementRef) -> bool {
    el.children()
        .filter_map(|c| c.value().as_text().map(|t| t.to_string()))
        .any(|t| t.chars().any(|c| c.is_alphanumeric()))
}

fn has_content(el: &ElementRef, probe: &Probe) -> bool {
    el.value().name() == "img"
        || el.text().any(|t| t.chars().any(|c| c.is_alphanumeric()))
        || el.select(&probe.img).next().is_some()
}

fn effective_cells(el: &ElementRef, probe: &Probe) -> usize {
    if el.value().name() == "tr" {
        return row_cells(el).len();
    }
    let mut cur = *el;
    for _ in 0..3 {
        let kids = element_children(&cur);
        if kids.len() == 1 && !has_own_text(&cur) {
            cur = kids[0];
            continue;
        }
        break;
    }
    let mut n = element_children(&cur).iter().filter(|k| has_content(k, probe)).count();
    if has_own_text(&cur) {
        n += 1;
    }
    n
}

fn in_landmark(el: &ElementRef) -> bool {
    let mut cur = Some(*el);
    while let Some(e) = cur {
        let name = e.value().name();
        if LANDMARK_TAGS.contains(&name) {
            return true;
        }
        if let Some(role) = e.value().attr("role") {
            let r = role.trim().to_lowercase();
            if LANDMARK_ROLES.contains(&r.as_str()) {
                return true;
            }
        }
        cur = e.parent().and_then(ElementRef::wrap);
    }
    false
}

fn is_data_row(tr: &ElementRef) -> bool {
    let cells = row_cells(tr);
    if cells.len() < 2 {
        return false;
    }
    if cells.iter().all(|c| c.value().name() == "th") {
        return false;
    }
    tr.text().any(|t| t.chars().any(|c| c.is_alphanumeric()))
}

fn form_like(members: &[ElementRef], probe: &Probe) -> bool {
    let n = members.len().max(1) as f32;
    if members.first().map_or(false, |m| m.value().name() == "tr") {
        let with_th = members
            .iter()
            .filter(|m| row_cells(m).iter().any(|c| c.value().name() == "th"))
            .count();
        return with_th as f32 / n >= 0.5;
    }
    let with_input = members.iter().filter(|m| m.select(&probe.editable).next().is_some()).count();
    with_input as f32 / n >= 0.5
}

fn anchor_parent<'a>(p: &ElementRef<'a>) -> ElementRef<'a> {
    if p.value().id().map_or(false, css_ident_ok) {
        return *p;
    }
    let mut final_p = *p;
    let mut walk = *p;
    for _ in 0..5 {
        let g = match walk.parent().and_then(ElementRef::wrap) {
            Some(g) => g,
            None => break,
        };
        let name = g.value().name();
        if name == "html" || name == "body" {
            break;
        }
        let has_id = g.value().id().map_or(false, css_ident_ok);
        if has_id || matches!(name, "table" | "ul" | "ol" | "nav") {
            final_p = g;
            if has_id || name == "table" {
                break;
            }
        }
        walk = g;
    }
    final_p
}

fn text_key(s: &str) -> String {
    s.to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Debug, Clone)]
pub struct TitleExcerpt {
    pub text: String,
    pub selector: String,
    pub rows: usize,
    pub members: usize,
    pub header_cells: usize,
    pub dominance: f32,
}

fn own_text(el: &ElementRef) -> String {
    let joined = el
        .children()
        .filter_map(|c| c.value().as_text().map(|t| t.to_string()))
        .collect::<Vec<_>>()
        .join(" ");
    joined.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn run(html: &str) -> ListCensus {
    let probe = match Probe::new() {
        Some(p) => p,
        None => return ListCensus::default(),
    };
    let doc = Html::parse_document(html);
    let mut order = HashMap::new();
    let mut subtree_end: Vec<usize> = Vec::new();
    for (i, n) in doc.tree.root().descendants().enumerate() {
        order.insert(n.id(), i);
    }
    subtree_end.resize(order.len(), 0);
    for n in doc.tree.root().descendants() {
        let start = order.get(&n.id()).copied().unwrap_or(0);
        let count = n.descendants().count();
        if start < subtree_end.len() {
            subtree_end[start] = start + count.saturating_sub(1);
        }
    }

    let order_of = |e: &ElementRef| order.get(&e.id()).copied();
    let mut raw: Vec<CensusGroup> = Vec::new();
    let mut landmark_dropped = 0usize;
    for p in doc.select(&probe.all) {
        let kids = element_children(&p);
        if kids.len() < 2 {
            continue;
        }
        let mut assigned = vec![false; kids.len()];
        for i in 0..kids.len() {
            if assigned[i] {
                continue;
            }
            let rep = kids[i];
            if NON_ITEM_TAGS.contains(&rep.value().name()) {
                continue;
            }
            let mut idx = vec![i];
            for j in (i + 1)..kids.len() {
                if !assigned[j] && similarity(&rep, &kids[j]) >= 60.0 {
                    idx.push(j);
                }
            }
            if idx.len() < 2 {
                continue;
            }
            for &k in &idx {
                assigned[k] = true;
            }
            let mut members: Vec<ElementRef> = idx.iter().map(|&k| kids[k]).collect();
            if rep.value().name() == "tr" {
                members.retain(|m| is_data_row(m));
            }
            if members.len() < 2 {
                continue;
            }
            if in_landmark(&p) {
                landmark_dropped += 1;
                continue;
            }
            raw.push(build_group(&p, &members, &order_of, &subtree_end, &probe, &doc));
        }
    }

    raw.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut kept: Vec<CensusGroup> = Vec::new();
    let mut suppressed = 0usize;
    for c in raw.into_iter() {
        if kept.iter().any(|k| nested(k, &c)) {
            suppressed += 1;
            continue;
        }
        kept.push(c);
        if kept.len() >= MAX_KEPT {
            break;
        }
    }

    let empty_tables = find_empty_tables(&doc, &probe);
    ListCensus { groups: kept, suppressed, landmark_dropped, empty_tables }
}

fn build_group(
    p: &ElementRef,
    members: &[ElementRef],
    order_of: &dyn Fn(&ElementRef) -> Option<usize>,
    subtree_end: &[usize],
    probe: &Probe,
    doc: &Html,
) -> CensusGroup {
    let anchor = anchor_parent(p);
    let parent_sig = signature(&anchor, true);
    let mut sigs: Vec<String> = Vec::new();
    for m in members {
        let s = signature(m, false);
        if !sigs.contains(&s) {
            sigs.push(s);
        }
    }
    let item_selector = sigs
        .iter()
        .map(|s| format!("{} {}", parent_sig, s))
        .collect::<Vec<_>>()
        .join(", ");
    let member_order: Vec<usize> = members
        .iter()
        .filter_map(|m| order_of(m))
        .collect();
    let ranges: Vec<(usize, usize)> = member_order
        .iter()
        .map(|&s| (s, subtree_end.get(s).copied().unwrap_or(s)))
        .collect();
    let exact = match Selector::parse(&item_selector) {
        Ok(sel) => {
            let picked: HashSet<usize> = doc
                .select(&sel)
                .filter_map(|e| order_of(&e))
                .collect();
            let mine: HashSet<usize> = member_order.iter().copied().collect();
            let covered = mine.iter().all(|i| picked.contains(i));
            let extra = picked.len().saturating_sub(mine.len());
            covered && extra <= (mine.len() / 5).max(1)
        }
        Err(_) => false,
    };
    let cells: Vec<usize> = members.iter().map(|m| effective_cells(m, probe)).collect();
    let avg_cells = cells.iter().sum::<usize>() as f32 / cells.len().max(1) as f32;
    let member_texts: Vec<String> = members.iter().map(|m| norm_text(m, MATCH_CHARS)).collect();
    let distinct: HashSet<String> = member_texts.iter().map(|t| text_key(t)).filter(|t| !t.is_empty()).collect();
    let distinct_ratio = distinct.len() as f32 / members.len().max(1) as f32;
    let token_sets: Vec<HashSet<String>> = member_texts
        .iter()
        .take(MIRROR_SAMPLE)
        .map(|t| t.split_whitespace().map(|w| w.to_lowercase()).collect())
        .collect();
    let mirror = mirror_ratio(&token_sets);
    let fl = form_like(members, probe);
    let score = members.len() as f32 * avg_cells.min(30.0) * distinct_ratio * if exact { 1.0 } else { 0.5 };
    let samples: Vec<String> = member_texts
        .iter()
        .filter(|t| !t.is_empty())
        .take(3)
        .map(|t| t.chars().take(SAMPLE_CHARS).collect())
        .collect();
    CensusGroup {
        parent_sig,
        item_selector,
        tag: members.first().map(|m| m.value().name().to_lowercase()).unwrap_or_default(),
        members: members.len(),
        avg_cells,
        distinct_ratio,
        form_like: fl,
        exact,
        score,
        mirror,
        content_margin: None,
        samples,
        member_order,
        member_texts,
        ranges,
    }
}

fn mirror_ratio(sets: &[HashSet<String>]) -> f32 {
    let mut sum = 0.0f32;
    let mut pairs = 0usize;
    for i in 0..sets.len() {
        for j in (i + 1)..sets.len() {
            let uni = sets[i].union(&sets[j]).count();
            if uni == 0 {
                continue;
            }
            sum += sets[i].intersection(&sets[j]).count() as f32 / uni as f32;
            pairs += 1;
        }
    }
    if pairs == 0 { 0.0 } else { sum / pairs as f32 }
}

fn nested(a: &CensusGroup, b: &CensusGroup) -> bool {
    let (alo, ahi) = a.envelope();
    let (blo, bhi) = b.envelope();
    if bhi < alo || ahi < blo {
        return false;
    }
    b.ranges.iter().any(|&(bs, be)| {
        a.ranges
            .iter()
            .any(|&(as_, ae)| (as_ <= bs && bs <= ae) || (bs <= as_ && as_ <= be))
    })
}

fn owning_table<'a>(tr: &ElementRef<'a>) -> Option<ElementRef<'a>> {
    let mut cur = tr.parent().and_then(ElementRef::wrap);
    while let Some(e) = cur {
        if e.value().name() == "table" {
            return Some(e);
        }
        cur = e.parent().and_then(ElementRef::wrap);
    }
    None
}

fn find_empty_tables(doc: &Html, probe: &Probe) -> Vec<EmptyTable> {
    let mut out: Vec<EmptyTable> = Vec::new();
    for table in doc.select(&probe.table) {
        if in_landmark(&table) {
            continue;
        }
        let rows: Vec<ElementRef> = table
            .select(&probe.tr)
            .filter(|r| owning_table(r).map_or(false, |t| t.id() == table.id()))
            .collect();
        let mut header_cells = 0usize;
        let mut body: Vec<ElementRef> = Vec::new();
        for r in rows.iter() {
            let cells = row_cells(r);
            let in_thead = r
                .parent()
                .and_then(ElementRef::wrap)
                .map_or(false, |p| p.value().name() == "thead");
            let all_th = !cells.is_empty() && cells.iter().all(|c| c.value().name() == "th");
            if in_thead || (all_th && body.is_empty()) {
                header_cells = header_cells.max(cells.iter().map(|c| attr_span(c, "colspan")).sum());
                continue;
            }
            if norm_text(r, 8).is_empty() && r.select(&probe.img).next().is_none() {
                continue;
            }
            body.push(*r);
        }
        if header_cells < 3 {
            continue;
        }
        if body.iter().any(|r| is_data_row(r)) {
            continue;
        }
        let notice = body
            .iter()
            .map(|r| norm_text(r, 80))
            .find(|t| !t.is_empty())
            .unwrap_or_default();
        out.push(EmptyTable {
            selector: signature(&anchor_parent(&table), true),
            header_cells,
            body_rows: body.len(),
            notice,
        });
    }
    out.sort_by(|a, b| b.header_cells.cmp(&a.header_cells));
    out
}

impl ListCensus {
    pub fn data_grade_count(&self) -> usize {
        self.groups.iter().filter(|g| g.data_grade()).count()
    }

    pub fn empty_verdict(&self) -> Option<&EmptyTable> {
        if self.groups.iter().any(|g| g.structural_grade()) {
            return None;
        }
        self.empty_tables.iter().find(|t| t.body_rows <= EMPTY_MAX_BODY_ROWS)
    }

    pub fn decisive_fallback(&self) -> Option<&CensusGroup> {
        let data: Vec<&CensusGroup> = self.groups.iter().filter(|g| g.data_grade() && g.exact).collect();
        let top = *data.first()?;
        if top.tag != "tr" || top.members < FALLBACK_MIN_ROWS || top.avg_cells < FALLBACK_MIN_CELLS {
            return None;
        }
        if top.content_margin.map_or(true, |m| m <= 0.0) {
            return None;
        }
        let second = data.get(1).map_or(0.0, |g| g.score);
        if top.score < FALLBACK_DOMINANCE * second {
            return None;
        }
        Some(top)
    }

    pub fn anchor_titles(&self, titles: &[String]) -> Option<&CensusGroup> {
        let ts: Vec<String> = titles
            .iter()
            .map(|t| text_key(t))
            .filter(|t| t.chars().count() >= 2)
            .collect();
        let mut best: Option<(&CensusGroup, usize)> = None;
        for g in self.groups.iter() {
            if g.form_like || !g.exact {
                continue;
            }
            let keys: Vec<String> = g.member_texts.iter().map(|m| text_key(m)).collect();
            let hits = ts.iter().filter(|t| keys.iter().any(|k| k.contains(t.as_str()))).count();
            if hits == 0 {
                continue;
            }
            let better = match best {
                None => true,
                Some((b, h)) => hits > h || (hits == h && g.score > b.score),
            };
            if better {
                best = Some((g, hits));
            }
        }
        best.map(|(g, _)| g)
    }

    pub fn shadow_agreement(&self, html: &str, selector: &str) -> Option<(f32, &CensusGroup, usize, usize)> {
        let top = self.groups.iter().find(|g| g.structural_grade())?;
        let sel = Selector::parse(selector).ok()?;
        let cell_q = Selector::parse("td, th").ok()?;
        let doc = Html::parse_document(html);
        let mut rows = 0usize;
        let mut pending = 0usize;
        let mut boa: HashSet<usize> = HashSet::new();
        for (i, n) in doc.tree.root().descendants().enumerate() {
            let e = match ElementRef::wrap(n) {
                Some(e) => e,
                None => continue,
            };
            if !sel.matches(&e) {
                continue;
            }
            rows += 1;
            if pending > 0 {
                pending -= 1;
                continue;
            }
            let span = e
                .select(&cell_q)
                .filter_map(|c| c.value().attr("rowspan"))
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .max()
                .unwrap_or(1);
            pending = span.saturating_sub(1);
            boa.insert(i);
        }
        let mine: HashSet<usize> = top.member_order.iter().copied().collect();
        let inter = boa.intersection(&mine).count();
        let uni = boa.union(&mine).count().max(1);
        Some((inter as f32 / uni as f32, top, rows, boa.len()))
    }

    pub fn title_excerpt(&self, html: &str) -> Option<TitleExcerpt> {
        let top_idx = self.groups.iter().position(|g| g.structural_grade())?;
        let top = &self.groups[top_idx];
        if top.members < FALLBACK_MIN_ROWS || top.avg_cells < FALLBACK_MIN_CELLS {
            return None;
        }
        let rival = self
            .groups
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != top_idx)
            .map(|(_, g)| g.score)
            .fold(0.0f32, f32::max);
        if top.score < FALLBACK_DOMINANCE * rival {
            return None;
        }
        let doc = Html::parse_document(html);
        let wanted: HashSet<usize> = top.member_order.iter().take(EXCERPT_ROWS).copied().collect();
        let mut order = HashMap::new();
        let mut rows: Vec<ElementRef> = Vec::new();
        for (i, n) in doc.tree.root().descendants().enumerate() {
            order.insert(n.id(), i);
            if wanted.contains(&i) {
                if let Some(e) = ElementRef::wrap(n) {
                    rows.push(e);
                }
            }
        }
        if rows.is_empty() {
            return None;
        }
        let mut header: Vec<String> = Vec::new();
        if top.tag == "tr" {
            if let (Some(table), Ok(th_q)) = (owning_table(&rows[0]), Selector::parse("th")) {
                let first = order.get(&rows[0].id()).copied().unwrap_or(usize::MAX);
                for th in table.select(&th_q) {
                    if header.len() >= EXCERPT_HEADER_CELLS {
                        break;
                    }
                    if owning_table(&th).map_or(true, |t| t.id() != table.id()) {
                        continue;
                    }
                    if order.get(&th.id()).copied().unwrap_or(usize::MAX) > first {
                        continue;
                    }
                    let t = norm_text(&th, EXCERPT_LINE_CHARS);
                    if !t.is_empty() && !header.contains(&t) {
                        header.push(t);
                    }
                }
            }
        }
        let mut text = format!(
            "[LIST ROWS] {} | rows {} (first {} shown) | each line under [ROW n] is the text of one cell, copied verbatim\n",
            top.item_selector,
            top.members,
            rows.len()
        );
        if !header.is_empty() {
            text.push_str(&format!("[HEADER] {}\n", header.join(" | ")));
        }
        for (k, row) in rows.iter().enumerate() {
            text.push_str(&format!("[ROW {}]\n", k + 1));
            let mut lines = 0usize;
            let mut last = String::new();
            for n in row.descendants() {
                let e = match ElementRef::wrap(n) {
                    Some(e) => e,
                    None => continue,
                };
                if NON_ITEM_TAGS.contains(&e.value().name()) {
                    continue;
                }
                let t: String = own_text(&e).chars().take(EXCERPT_LINE_CHARS).collect();
                if t.is_empty() || !t.chars().any(|c| c.is_alphanumeric()) || t == last {
                    continue;
                }
                text.push_str(&t);
                text.push('\n');
                last = t;
                lines += 1;
                if lines >= EXCERPT_ROW_LINES {
                    break;
                }
            }
        }
        Some(TitleExcerpt {
            text,
            selector: top.item_selector.clone(),
            rows: rows.len(),
            members: top.members,
            header_cells: header.len(),
            dominance: if rival > 0.0 { top.score / rival } else { f32::INFINITY },
        })
    }

    pub fn report_lines(&self) -> Vec<String> {
        let mut out = vec![format!(
            "  🧮 [LIST CENSUS] 반복 구조 후보 {}개 (겹침 억제 {} · 랜드마크 제외 {}) | 구조급 {}개 · 데이터급 {}개 | 머리행만 있는 빈 표 {}개 — DOM 이 후보를 내고, 행 내용 임베딩이 본문/크롬을 가르고, 조상·자손 관계로 겹치는 후보는 점수가 높은 쪽만 남깁니다. 구조급은 여러 칸 · 폼 아님 · 위아래 복제본 아님, 데이터급은 거기에 내용마진이 양수인 것입니다.",
            self.groups.len(),
            self.suppressed,
            self.landmark_dropped,
            self.groups.iter().filter(|g| g.structural_grade()).count(),
            self.data_grade_count(),
            self.empty_tables.len()
        )];
        for (i, g) in self.groups.iter().take(5).enumerate() {
            out.push(format!(
                "    · #{} '{}' | 행 {} · 칸 평균 {:.1} · 고유 {:.2} · 복제도 {:.2} · 폼 {} · 셀렉터 정확 {} · 구조점수 {:.1} · 내용마진 {} · 데이터급 {} | 예: \"{}\"",
                i + 1,
                g.item_selector,
                g.members,
                g.avg_cells,
                g.distinct_ratio,
                g.mirror,
                g.form_like,
                g.exact,
                g.score,
                g.content_margin.map_or("-".to_string(), |m| format!("{:+.4}", m)),
                g.data_grade(),
                g.samples.first().cloned().unwrap_or_default()
            ));
        }
        for t in self.empty_tables.iter().take(3) {
            out.push(format!(
                "    · 📭 빈 표 '{}' | 머리행 {}칸 · 본문 {}행 · \"{}\"",
                t.selector, t.header_cells, t.body_rows, t.notice
            ));
        }
        out
    }
}

pub async fn score_content(census: &mut ListCensus, model: &crate::model::LogisModel, page_type: &str, doc_lang: &str) {
    let targets: Vec<usize> = census
        .groups
        .iter()
        .enumerate()
        .filter(|(_, g)| g.structural_grade())
        .map(|(i, _)| i)
        .take(6)
        .collect();
    if targets.is_empty() {
        return;
    }
    let domain_raw = crate::parsing::get_page_type_classification_bias(page_type, doc_lang);
    let mut domain: Vec<String> = crate::utils::ai_utils::split_bias_phrases(&domain_raw);
    domain.truncate(48);
    let mut chrome: Vec<String> = crate::utils::ai_utils::split_bias_phrases(CHROME_TEXT);
    if let Some(local) = crate::logic::site_chrome_sentence(doc_lang) {
        for p in crate::utils::ai_utils::split_bias_phrases(&local) {
            if !chrome.contains(&p) {
                chrome.push(p);
            }
        }
    }
    chrome.truncate(48);
    if domain.is_empty() || chrome.is_empty() {
        return;
    }
    let mut samples: Vec<(usize, String)> = Vec::new();
    for &gi in targets.iter() {
        for s in census.groups[gi].samples.iter().take(3) {
            samples.push((gi, s.clone()));
        }
    }
    if samples.is_empty() {
        return;
    }
    let mut batch: Vec<String> = Vec::with_capacity(domain.len() + chrome.len() + samples.len());
    batch.extend(domain.iter().cloned());
    batch.extend(chrome.iter().cloned());
    batch.extend(samples.iter().map(|(_, s)| s.clone()));
    let embs = match model.get_embedding_batch(batch).await {
        Ok(e) if e.len() == domain.len() + chrome.len() + samples.len() => e,
        _ => return,
    };
    let d_embs: Vec<Vec<f32>> = embs[..domain.len()].to_vec();
    let c_embs: Vec<Vec<f32>> = embs[domain.len()..domain.len() + chrome.len()].to_vec();
    let s_embs = &embs[domain.len() + chrome.len()..];
    let mut acc: HashMap<usize, (f32, usize)> = HashMap::new();
    for ((gi, _), e) in samples.iter().zip(s_embs.iter()) {
        if e.iter().all(|v| *v == 0.0) {
            continue;
        }
        let m = crate::utils::ai_utils::max_pool_sim(e, &d_embs) - crate::utils::ai_utils::max_pool_sim(e, &c_embs);
        let entry = acc.entry(*gi).or_insert((0.0, 0));
        entry.0 += m;
        entry.1 += 1;
    }
    for (gi, (sum, n)) in acc.into_iter() {
        if n == 0 {
            continue;
        }
        let m = sum / n as f32;
        census.groups[gi].content_margin = Some(m);
        crate::utils::score_dynamics::record_baseline("commerce.census_content_margin", m);
    }
}

pub fn harvest_titles(info: &serde_json::Value, raw: &str) -> Vec<String> {
    fn push_title(s: &str, out: &mut Vec<String>) {
        let t = s.trim();
        if t.is_empty() || t.chars().count() > 200 {
            return;
        }
        let compact: String = t.replace(',', "").replace('.', "");
        let compact = compact.trim();
        if !compact.is_empty() && compact.chars().all(|c| c.is_ascii_digit()) {
            return;
        }
        if !out.iter().any(|o| o == t) {
            out.push(t.to_string());
        }
    }
    fn from_array(v: &serde_json::Value, out: &mut Vec<String>) {
        if let Some(a) = v.as_array() {
            for x in a {
                match x {
                    serde_json::Value::String(s) => push_title(s, out),
                    serde_json::Value::Object(o) => {
                        for k in ["title", "name", "text", "product", "goods"] {
                            if let Some(s) = o.get(k).and_then(|v| v.as_str()) {
                                push_title(s, out);
                                break;
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    let mut out: Vec<String> = Vec::new();
    match info {
        serde_json::Value::Object(o) => {
            let preferred = ["order", "goods", "title", "titles", "product", "products"];
            for k in preferred.iter() {
                if let Some(v) = o.get(*k) {
                    from_array(v, &mut out);
                }
            }
            if out.is_empty() {
                for (k, v) in o.iter() {
                    if preferred.contains(&k.as_str()) {
                        continue;
                    }
                    from_array(v, &mut out);
                }
            }
            if out.is_empty() && o.len() == 1 {
                if let Some(serde_json::Value::String(s)) = o.values().next() {
                    push_title(s, &mut out);
                }
            }
        }
        serde_json::Value::Array(_) => from_array(info, &mut out),
        _ => {}
    }
    let unparsed = match info {
        serde_json::Value::Object(o) => o.is_empty(),
        serde_json::Value::Array(_) => false,
        _ => true,
    };
    if out.is_empty() && unparsed {
        if let Some(start) = raw.find('[') {
            let mut cur = String::new();
            let mut in_str = false;
            let mut escaped = false;
            for ch in raw[start + 1..].chars() {
                if in_str {
                    if escaped {
                        cur.push(ch);
                        escaped = false;
                    } else if ch == '\\' {
                        escaped = true;
                    } else if ch == '"' {
                        push_title(&cur, &mut out);
                        cur.clear();
                        in_str = false;
                    } else {
                        cur.push(ch);
                    }
                } else if ch == '"' {
                    in_str = true;
                } else if ch == ']' {
                    break;
                }
            }
        }
    }
    out
}
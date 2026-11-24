//! Offline vocabulary frequency miner & `VocabPlan` builder CLI (Phase 2 – Part 1).
//!
//! Usage:
//!   vocab_plan_cli --tokenizer <tokenizer.json> --family <qwen3|qwen3_5|granite|siglip2>
//!                  --bundle <ko+en|ja+en|zh+en|ar+en|latin+en|all12|en> --k <K>
//!                  --out <plan.vocab_plan.json>
//!                  [--vocab-size <V>] [--corpus-dir <dir>] [--leak-file <path>]
//!                  [--freq-cache <cache.json>] [--bias-json <full_bias.json>]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use premium_discount_quant::mine::{
    select_hybrid_plan, FreqCache, HybridPlanConfig, LeakBlocklist,
};
use premium_discount_quant::trim::Langs;
use premium_discount_quant::vocab_plan::sha256_hex;

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].clone())
}

fn bundle_langs(bundle: &str) -> Result<(Langs, Vec<String>)> {
    let b = bundle.trim().to_ascii_lowercase();
    match b.as_str() {
        "en" => Ok((Langs::from_codes(&["en"]).unwrap(), vec!["en".into()])),
        "ko+en" => {
            let mut l = Langs::from_codes(&["ko", "en"]).unwrap();
            l.hanja = true;
            Ok((l, vec!["ko".into(), "en".into()]))
        }
        "ja+en" => Ok((
            Langs::from_codes(&["ja", "en"]).unwrap(),
            vec!["ja".into(), "en".into()],
        )),
        "zh+en" => Ok((
            Langs::from_codes(&["zh", "zh-tw", "en"]).unwrap(),
            vec!["zh".into(), "en".into()],
        )),
        "ar+en" => Ok((
            Langs::from_codes(&["ar", "en"]).unwrap(),
            vec!["ar".into(), "en".into()],
        )),
        "latin+en" => Ok((
            Langs::from_codes(&["de", "fr", "es", "it", "pt", "nl", "cs", "en"]).unwrap(),
            vec!["en".into(), "de".into(), "fr".into(), "es".into(), "it".into(), "pt".into(), "nl".into(), "cs".into()],
        )),
        "all12" | "core12" => Ok((
            Langs::all(),
            vec![
                "en".into(), "ko".into(), "ja".into(), "zh".into(),
                "de".into(), "fr".into(), "es".into(), "it".into(),
                "pt".into(), "nl".into(), "cs".into(), "ar".into(),
            ],
        )),
        other => {
            let codes: Vec<String> = other
                .split(|c| c == '+' || c == ',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            let l = Langs::from_codes(&codes).context("empty bundle")?;
            Ok((l, codes))
        }
    }
}

/// Extract bundle-isolated lexicon strings & schema keys from `full_bias.json` (never mixing
/// unrelated non-ASCII languages into e.g. an `en` bundle).
fn load_bundle_lexicon(bias_path: &Path, lang_codes: &[String]) -> Result<(Vec<String>, Vec<String>, String)> {
    let raw = std::fs::read_to_string(bias_path)
        .with_context(|| format!("read bias json {}", bias_path.display()))?;
    let lex_sha = sha256_hex(raw.as_bytes());
    let val: serde_json::Value = serde_json::from_str(&raw)?;
    let mut texts = Vec::new();
    let mut keys = Vec::new();

    fn walk(v: &serde_json::Value, texts: &mut Vec<String>, keys: &mut Vec<String>) {
        match v {
            serde_json::Value::String(s) => texts.push(s.clone()),
            serde_json::Value::Array(a) => a.iter().for_each(|x| walk(x, texts, keys)),
            serde_json::Value::Object(o) => {
                for (k, x) in logis_schema::json_order::sorted_iter(o) {
                    keys.push(k.clone());
                    walk(x, texts, keys);
                }
            }
            serde_json::Value::Number(n) => texts.push(n.to_string()),
            _ => {}
        }
    }

    if let Some(root) = val.as_object() {
        for lang in lang_codes {
            let key = if lang == "zh-hans" || lang == "zh-tw" { "zh" } else { lang.as_str() };
            if let Some(section) = root.get(key) {
                walk(section, &mut texts, &mut keys);
            }
        }
    }
    keys.sort();
    keys.dedup();
    Ok((texts, keys, lex_sha))
}

fn collect_corpus_files(dir: &Path, out: &mut BTreeMap<String, Vec<String>>) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let p = entry.path();
        if p.is_dir() {
            collect_corpus_files(&p, out)?;
            continue;
        }
        let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !matches!(ext, "txt" | "jsonl" | "yaml" | "yml" | "md") {
            continue;
        }
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("en").to_lowercase();
        let lang = ["ko", "en", "ja", "zh", "de", "fr", "es", "it", "pt", "nl", "cs", "ar"]
            .iter()
            .copied()
            .find(|&l| stem == l || stem.starts_with(&format!("{l}_")) || stem.ends_with(&format!("_{l}")))
            .unwrap_or("en")
            .to_string();
        if let Ok(content) = std::fs::read_to_string(&p) {
            let entry_vec = out.entry(lang).or_default();
            for line in content.lines() {
                let t = line.trim();
                if !t.is_empty() {
                    entry_vec.push(t.to_string());
                }
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let tok_path = PathBuf::from(parse_arg(&args, "--tokenizer").context("missing --tokenizer")?);
    let family = parse_arg(&args, "--family").unwrap_or_else(|| "qwen3_5".into());
    let bundle = parse_arg(&args, "--bundle").unwrap_or_else(|| "all12".into());
    let target_k: usize = parse_arg(&args, "--k")
        .unwrap_or_else(|| "65536".into())
        .parse()
        .context("invalid --k")?;
    let out_path = PathBuf::from(parse_arg(&args, "--out").context("missing --out")?);

    let t0 = Instant::now();
    let tok_raw = std::fs::read_to_string(&tok_path)
        .with_context(|| format!("read tokenizer {}", tok_path.display()))?;
    let tok_sha = sha256_hex(tok_raw.as_bytes());
    let tok = tokenizers::Tokenizer::from_file(&tok_path).map_err(anyhow::Error::msg)?;
    let vocab_size = parse_arg(&args, "--vocab-size")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or_else(|| tok.get_vocab_size(true));

    let (langs, lang_codes) = bundle_langs(&bundle)?;

    // Leak blocklist
    let mut blocked_lines = Vec::new();
    if let Some(leak_p) = parse_arg(&args, "--leak-file") {
        if let Ok(content) = std::fs::read_to_string(&leak_p) {
            for line in content.lines() {
                if !line.trim().is_empty() {
                    blocked_lines.push(line.to_string());
                }
            }
        }
    }
    let blocklist = LeakBlocklist::from_texts(&blocked_lines);

    // Load or build FreqCache
    let cache_path = parse_arg(&args, "--freq-cache").map(PathBuf::from);
    let mut cache = if let Some(cp) = &cache_path {
        if cp.is_file() {
            match FreqCache::load(cp) {
                Ok(c) if c.tokenizer_sha256 == tok_sha && c.vocab_size == vocab_size => c,
                _ => FreqCache::new(tok_sha.clone(), vocab_size, blocklist.sha256().to_string()),
            }
        } else {
            FreqCache::new(tok_sha.clone(), vocab_size, blocklist.sha256().to_string())
        }
    } else {
        FreqCache::new(tok_sha.clone(), vocab_size, blocklist.sha256().to_string())
    };

    // Bundle-isolated lexicon from full_bias.json (or embedded BIAS_DICT fallback)
    let (lex_texts, schema_keys, lex_sha) = if let Some(bp) = parse_arg(&args, "--bias-json") {
        load_bundle_lexicon(Path::new(&bp), &lang_codes)?
    } else {
        let mut texts = Vec::new();
        let mut keys = Vec::new();
        if let Some(root) = logis_bias::bias_schema::BIAS_DICT.as_object() {
            for lang in &lang_codes {
                let k = if lang == "zh-hans" || lang == "zh-tw" { "zh" } else { lang.as_str() };
                if let Some(sec) = root.get(k) {
                    texts.push(sec.to_string());
                }
            }
        }
        keys.push("goods".into());
        keys.push("condition".into());
        (texts, keys, sha256_hex(b"embedded_bias"))
    };

    let is_siglip = family.to_ascii_lowercase().contains("siglip");

    // Mine lexicon into domain counts per language
    let mut domain_by_lang: BTreeMap<String, Vec<String>> = BTreeMap::new();
    if let Some(cdir) = parse_arg(&args, "--corpus-dir") {
        collect_corpus_files(Path::new(&cdir), &mut domain_by_lang)?;
    }
    if let Some(wdir) = parse_arg(&args, "--web-dir") {
        let mut web_by_lang: BTreeMap<String, Vec<String>> = BTreeMap::new();
        collect_corpus_files(Path::new(&wdir), &mut web_by_lang)?;
        for (lang, docs) in &web_by_lang {
            cache.mine_texts(&tok, lang, docs, false, &blocklist, is_siglip)?;
        }
    }
    for (lang, docs) in &domain_by_lang {
        cache.mine_texts(&tok, lang, docs, true, &blocklist, is_siglip)?;
    }
    // Always mine lexicon texts into domain counts for target languages so domain phrases rank high
    for lang in &lang_codes {
        cache.mine_texts(&tok, lang, &lex_texts, true, &blocklist, is_siglip)?;
    }

    if let Some(cp) = &cache_path {
        cache.save(cp)?;
    }

    let key_refs: Vec<&str> = schema_keys.iter().map(String::as_str).collect();
    let holdout = BTreeMap::new();
    let cfg = HybridPlanConfig {
        model_family: family.clone(),
        bundle: bundle.clone(),
        target_k,
        langs,
        lang_codes,
        domain_alpha: 10.0,
        hanja_cap: 4096,
        corpus_manifest_sha256: sha256_hex(format!("{family}:{bundle}:{target_k}").as_bytes()),
        lexicon_sha256: lex_sha,
    };

    let plan = select_hybrid_plan(
        &tok_raw,
        &tok,
        vocab_size,
        &cache,
        &lex_texts,
        &key_refs,
        &cfg,
        &holdout,
    )?;
    plan.save(&out_path)?;

    let tc = plan.tier_counts();
    use premium_discount_quant::vocab_plan::PlanTier;
    println!(
        "[VOCAB-MINE] model={} flavor={} bundle={} V={} -> K={} (special={} byte={} struct={} domain={} freq={} closure={} pad={}) orphans={} sha256={} · {:.2} s",
        plan.model_family,
        plan.tokenizer_flavor,
        plan.bundle,
        plan.orig_vocab_size,
        plan.kept_vocab_size,
        tc.get(&PlanTier::MandatorySpecial).copied().unwrap_or(0),
        tc.get(&PlanTier::MandatoryByte).copied().unwrap_or(0),
        tc.get(&PlanTier::MandatoryAsciiStruct).copied().unwrap_or(0),
        tc.get(&PlanTier::DomainLexicon).copied().unwrap_or(0),
        tc.get(&PlanTier::CorpusFreq).copied().unwrap_or(0),
        tc.get(&PlanTier::MergeClosure).copied().unwrap_or(0),
        tc.get(&PlanTier::AlignPad).copied().unwrap_or(0),
        plan.orphan_bpe_tokens,
        &plan.plan_sha256[..8.min(plan.plan_sha256.len())],
        t0.elapsed().as_secs_f64()
    );
    if plan.orphan_bpe_tokens > 0 {
        bail!("VocabPlan contains orphan tokens");
    }
    Ok(())
}

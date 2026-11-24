//! Trimmed artifact generator & E0 byte-exact verifier CLI (Phase 2 – Part 2).
//!
//! Takes a verified `VocabPlan` JSON and produces an isolated `-lc-trim-<bundle>-k<K>-v<ver>`
//! directory containing:
//!   - pruned `tokenizer.json` (0 orphan BPE tokens, remapped special/post_processor/padding IDs)
//!   - remapped `config.json` & `generation_config.json` (all special/vision IDs `< K`)
//!   - `orig_ids.bin` (u32 LE `new_to_old` map)
//!   - zero-requantization repacked `.gguf` OR dtype-preserving repacked `.safetensors`
//!   - `trim_manifest.json` with full SHA-256 checksums + 64-row E0 byte verification.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, ensure, Context, Result};
use premium_discount_quant::artifact_trim::{
    remap_config_json, trim_tokenizer_json, IdRemap, NumberingMode, TrimManifest,
};
use premium_discount_quant::gguf::{repack_gguf_vocab, GgufFile, GgufVocabOverride};
use premium_discount_quant::safetensors::{repack_safetensors_vocab, SafeTensors};
use premium_discount_quant::vocab_plan::VocabPlan;

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.windows(2)
        .find(|w| w[0] == flag)
        .map(|w| w[1].clone())
}

fn find_gguf_in_dir(dir: &Path) -> Option<PathBuf> {
    // Check runtime-q4km/model-q4km.gguf first (Qwen3.5-4B), then top-level *.gguf (excluding mmproj)
    let rt = dir.join("runtime-q4km").join("model-q4km.gguf");
    if rt.is_file() {
        return Some(rt);
    }
    let rd = std::fs::read_dir(dir).ok()?;
    for e in rd.flatten() {
        let p = e.path();
        if p.is_file() {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
            if name.ends_with(".gguf") && !name.contains("mmproj") {
                return Some(p);
            }
        }
    }
    None
}

fn extract_gguf_token_arrays(trimmed_tok_json: &str, k: usize) -> Result<(Vec<String>, Vec<i32>, Vec<String>)> {
    let doc: serde_json::Value = serde_json::from_str(trimmed_tok_json)?;
    let mut tokens = vec![String::new(); k];
    let mut types = vec![1i32; k];
    if let Some(v) = doc["model"]["vocab"].as_object() {
        for (tok, id_v) in v {
            if let Some(id) = id_v.as_u64() {
                if (id as usize) < k {
                    tokens[id as usize] = tok.clone();
                }
            }
        }
    }
    if let Some(arr) = doc["added_tokens"].as_array() {
        for a in arr {
            if let Some(id) = a["id"].as_u64() {
                if (id as usize) < k {
                    if let Some(c) = a["content"].as_str() {
                        tokens[id as usize] = c.to_string();
                    }
                    types[id as usize] = 3; // control/special
                }
            }
        }
    }
    let mut merges = Vec::new();
    if let Some(arr) = doc["model"]["merges"].as_array() {
        for m in arr {
            if let Some(s) = m.as_str() {
                merges.push(s.to_string());
            } else if let Some(pair) = m.as_array() {
                if pair.len() == 2 {
                    if let (Some(a), Some(b)) = (pair[0].as_str(), pair[1].as_str()) {
                        merges.push(format!("{a} {b}"));
                    }
                }
            }
        }
    }
    Ok((tokens, types, merges))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let src_dir = PathBuf::from(parse_arg(&args, "--src-dir").context("missing --src-dir")?);
    let out_dir = PathBuf::from(parse_arg(&args, "--out-dir").context("missing --out-dir")?);
    let exact_tok = parse_arg(&args, "--exact-tokenizer").map(PathBuf::from);
    let exact_map = parse_arg(&args, "--exact-map").map(PathBuf::from);
    let base_repo = parse_arg(&args, "--base-repo").unwrap_or_else(|| "local".into());
    let base_rev = parse_arg(&args, "--base-rev").unwrap_or_else(|| "pinned".into());
    let space_tag = parse_arg(&args, "--space-tag");
    let ancestor_max: usize = std::env::var("LOGIS_TRIM_ANCESTOR_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let t0 = Instant::now();
    std::fs::create_dir_all(&out_dir)?;

    let tok_src = src_dir.join("tokenizer.json");
    let tok_raw = std::fs::read_to_string(&tok_src)
        .with_context(|| format!("read {}", tok_src.display()))?;

    let (trimmed_tok_json, remap, tok_stats, plan_bundle, plan_sha, orig_vocab_size) =
        if let (Some(etok_p), Some(emap_p)) = (&exact_tok, &exact_map) {
            // Exact baseline mode (e.g. reproducing published alphaedge-ai tokenizer + GGUF apples-to-apples)
            let etok_raw = std::fs::read_to_string(etok_p)?;
            let new_to_old: Vec<u32> = serde_json::from_str(&std::fs::read_to_string(emap_p)?)?;
            let src_gguf = find_gguf_in_dir(&src_dir).context("exact mode requires source GGUF")?;
            let g_src = GgufFile::open(&src_gguf)?;
            let orig_v = g_src.tensor("token_embd.weight")?.shape[0];
            let mut old_to_new = vec![-1i32; orig_v];
            for (nid, &oid) in new_to_old.iter().enumerate() {
                ensure!((oid as usize) < orig_v, "exact map oid {oid} >= {orig_v}");
                old_to_new[oid as usize] = nid as i32;
            }
            let remap = IdRemap {
                mode: NumberingMode::Contiguous,
                orig_vocab: orig_v,
                new_to_old,
                old_to_new,
            };
            remap.verify_bijection()?;
            let stats = premium_discount_quant::artifact_trim::TrimTokenizerStats {
                flavor: "exact".into(),
                orig_vocab: orig_v,
                kept_vocab: remap.trimmed_vocab(),
                orig_merges: 0,
                kept_merges: 0,
                ancestors_added: 0,
                orphan_multi: 0,
            };
            (
                etok_raw,
                remap,
                stats,
                "exact".to_string(),
                "exact".to_string(),
                orig_v,
            )
        } else {
            let plan_path = PathBuf::from(parse_arg(&args, "--plan").context("missing --plan")?);
            let plan = VocabPlan::load(&plan_path)?;
            let (tt, rm, st) = trim_tokenizer_json(
                &tok_raw,
                plan.orig_vocab_size,
                &plan.kept_ids,
                ancestor_max,
            )?;
            ensure!(
                st.orphan_multi == 0,
                "trimmed tokenizer still has {} orphan multi-char tokens",
                st.orphan_multi
            );
            (
                tt,
                rm,
                st,
                plan.bundle.clone(),
                plan.plan_sha256.clone(),
                plan.orig_vocab_size,
            )
        };
    std::fs::write(out_dir.join("tokenizer.json"), &trimmed_tok_json)?;
    remap.save_orig_ids_bin(&out_dir.join("orig_ids.bin"))?;

    // 2. Remap config.json & optional generation_config.json, copy auxiliary configs
    let mut manifest_files: Vec<String> = vec!["tokenizer.json".into(), "orig_ids.bin".into()];
    if src_dir.join("config.json").is_file() {
        let cfg_raw = std::fs::read_to_string(src_dir.join("config.json"))?;
        let remapped = remap_config_json(&cfg_raw, &remap)?;
        std::fs::write(out_dir.join("config.json"), remapped)?;
        manifest_files.push("config.json".into());
    }
    if src_dir.join("generation_config.json").is_file() {
        let gc_raw = std::fs::read_to_string(src_dir.join("generation_config.json"))?;
        if let Ok(remapped) = remap_config_json(&gc_raw, &remap) {
            std::fs::write(out_dir.join("generation_config.json"), remapped)?;
            manifest_files.push("generation_config.json".into());
        }
    }
    for aux in [
        "chat_template.jinja",
        "chat_template.json",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "mmproj-BF16.gguf",
    ] {
        let p = src_dir.join(aux);
        if p.is_file() {
            std::fs::copy(&p, out_dir.join(aux))?;
        }
    }

    // Collect special token remap summary
    let mut special_tokens: BTreeMap<String, [u32; 2]> = BTreeMap::new();
    let orig_doc: serde_json::Value = serde_json::from_str(&tok_raw)?;
    if let Some(arr) = orig_doc["added_tokens"].as_array() {
        for a in arr {
            if let (Some(content), Some(old_id)) = (a["content"].as_str(), a["id"].as_u64()) {
                if let Some(new_id) = remap.map_old(old_id as u32) {
                    special_tokens.insert(content.to_string(), [old_id as u32, new_id]);
                }
            }
        }
    }

    // 3. Repack weights (GGUF or Safetensors) with E0 bit-exact row verification
    let orig_bytes: u64;
    let new_bytes: u64;
    if let Some(src_gguf) = find_gguf_in_dir(&src_dir) {
        let is_runtime_4b = src_gguf.parent().and_then(|p| p.file_name()).is_some_and(|n| n == "runtime-q4km");
        let rel_dst = if is_runtime_4b {
            "runtime-q4km/model-q4km.gguf".to_string()
        } else {
            src_gguf.file_name().unwrap().to_string_lossy().to_string()
        };
        let dst_gguf = out_dir.join(&rel_dst);
        let (tokens, token_types, merges) = extract_gguf_token_arrays(&trimmed_tok_json, remap.trimmed_vocab())?;
        let mut extra_kvs = Vec::new();
        for (tok_str, key) in [
            ("<|vision_start|>", "qwen35.vision_start_token_id"),
            ("<|vision_end|>", "qwen35.vision_end_token_id"),
            ("<|image_pad|>", "qwen35.image_token_id"),
            ("<|video_pad|>", "qwen35.video_token_id"),
        ] {
            if let Some(&[_old, new_id]) = special_tokens.get(tok_str) {
                extra_kvs.push((key.to_string(), premium_discount_quant::gguf::Kv::U32(new_id)));
            }
        }
        let ov = GgufVocabOverride {
            tokens: Some(tokens),
            token_types: Some(token_types),
            merges: Some(merges),
            extra_kvs,
        };
        repack_gguf_vocab(&src_gguf, &dst_gguf, &remap, &ov)?;
        orig_bytes = std::fs::metadata(&src_gguf)?.len();
        new_bytes = std::fs::metadata(&dst_gguf)?.len();
        manifest_files.push(rel_dst.clone());

        // Verify 64 sampled rows are E0 byte-identical
        let g_src = GgufFile::open(&src_gguf)?;
        let g_dst = GgufFile::open(&dst_gguf)?;
        let step = (remap.trimmed_vocab() / 64).max(1);
        for new_id in (0..remap.trimmed_vocab()).step_by(step).take(64) {
            let old_id = remap.new_to_old[new_id] as usize;
            ensure!(
                g_src.row_bytes("token_embd.weight", old_id)? == g_dst.row_bytes("token_embd.weight", new_id)?,
                "E0 row verification failed at new_id={new_id} old_id={old_id}"
            );
        }

        // Write READY.json if this is a runtime-q4km artifact
        if is_runtime_4b {
            let t_emb = g_dst.tensor("token_embd.weight")?;
            let (_, cols) = t_emb.rows_cols();
            let embd_row_bytes = t_emb.ty.row_bytes(cols) as u64;
            let backbone_bytes = new_bytes.saturating_sub(t_emb.nbytes() as u64);
            let ready = logis_llm::lang_gguf::ReadyManifest {
                recipe: "qwen35-q4km/2".into(),
                engine_fp: format!("q35_4b-{}-v{}k-q4km", plan_bundle, remap.trimmed_vocab() / 1000),
                source_revision: base_rev.clone(),
                gguf_sha256: premium_discount_manifest::sha256_file(&dst_gguf, 0)?,
                gguf_bytes: new_bytes,
                backbone_bytes,
                embd_row_bytes,
                vocab_size: remap.trimmed_vocab(),
                v_head_order: logis_llm::lang_gguf::v_head_order()?.into(),
                ascii_intact: true,
            };
            logis_llm::lang_gguf::write_ready_manifest(&out_dir, &ready)?;
            manifest_files.push("runtime-q4km/READY.json".into());
        }
    } else if src_dir.join("model.safetensors").is_file() || src_dir.join("model.safetensors.index.json").is_file() {
        let st_open = SafeTensors::open_model_dir(&src_dir, &[])?;
        let vocab_tensor = [
            "embeddings.tok_embeddings.weight",
            "text_model.embeddings.token_embedding.weight",
            "model.embed_tokens.weight",
        ]
        .iter()
        .copied()
        .find(|&n| st_open.tensors.contains_key(n))
        .context("could not find embedding tensor in safetensors")?;
        let dst_st = out_dir.join("model.safetensors");
        repack_safetensors_vocab(&src_dir, &dst_st, &[vocab_tensor], &remap)?;
        orig_bytes = st_open
            .paths
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
            .sum();
        new_bytes = std::fs::metadata(&dst_st)?.len();
        manifest_files.push("model.safetensors".into());

        // Verify 64 sampled rows are E0 byte-identical
        let st_dst = SafeTensors::open(&[dst_st])?;
        let step = (remap.trimmed_vocab() / 64).max(1);
        for new_id in (0..remap.trimmed_vocab()).step_by(step).take(64) {
            let old_id = remap.new_to_old[new_id] as usize;
            ensure!(
                st_open.row_bytes(vocab_tensor, old_id)? == st_dst.row_bytes(vocab_tensor, new_id)?,
                "E0 safetensors row verification failed at new_id={new_id} old_id={old_id}"
            );
        }
    } else {
        bail!("no GGUF or safetensors weights found in {}", src_dir.display());
    }

    // 4. Write and full-verify trim_manifest.json
    let rel_refs: Vec<&str> = manifest_files.iter().map(String::as_str).collect();
    let files = TrimManifest::build_file_entries(&out_dir, &rel_refs)?;
    let manifest = TrimManifest {
        schema_version: 1,
        base_repo,
        base_revision: base_rev,
        vocab_plan_sha256: plan_sha,
        lang_bundle: plan_bundle,
        numbering_mode: NumberingMode::Contiguous,
        orig_vocab_size,
        trimmed_vocab_size: remap.trimmed_vocab(),
        orphan_multi_count: tok_stats.orphan_multi,
        space_tag,
        special_tokens,
        files,
    };
    manifest.save(&out_dir)?;
    manifest.verify_dir(&out_dir, true)?;

    println!(
        "[ARTIFACT-TRIM] built dir={} mode=contiguous V={}->K={} merges={}->{} ancestors_added={} orphans={} bytes={}->{} elapsed_ms={}",
        out_dir.display(),
        tok_stats.orig_vocab,
        tok_stats.kept_vocab,
        tok_stats.orig_merges,
        tok_stats.kept_merges,
        tok_stats.ancestors_added,
        tok_stats.orphan_multi,
        orig_bytes,
        new_bytes,
        t0.elapsed().as_millis()
    );
    Ok(())
}

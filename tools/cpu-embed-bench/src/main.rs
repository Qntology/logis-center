//! docs/mac-verify.md 3-2: where the CPU batched embedding forward spends its time, and whether
//! cache-blocked custom kernels (packed weights, fused RMSNorm→Linear, fused Wi→SiLU·up) beat the
//! candle/gemm path — with the bit-identity contract kept (kern.rs).
//!
//!   cargo run --release -- <granite dir> [--threads N] [--quick]
//!
//! Model: the app's FloatModel (models/embedding.rs) re-implemented op for op ("candle" path), so the
//! profile matches the app. The "custom" path swaps the four Linear layers per decoder layer for
//! kern::gemm and fuses the norms / GLU; attention and RoPE stay candle ops.
mod kern;

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::VarBuilder;
use kern::{Blocking, Epilogue, PackedW};
use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Instant;

// ── profiling ──────────────────────────────────────────────────────────────
static PROF: Mutex<BTreeMap<&'static str, (f64, usize)>> = Mutex::new(BTreeMap::new());
static PROF_ON: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn prof<T>(cat: &'static str, f: impl FnOnce() -> T) -> T {
    if !PROF_ON.load(std::sync::atomic::Ordering::Relaxed) {
        return f();
    }
    let t = Instant::now();
    let r = f();
    let mut g = PROF.lock().unwrap();
    let e = g.entry(cat).or_insert((0.0, 0));
    e.0 += t.elapsed().as_secs_f64();
    e.1 += 1;
    r
}

// ── model (embedding.rs, op for op) ────────────────────────────────────────
#[derive(serde::Deserialize, Clone)]
struct Config {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    norm_eps: f64,
    vocab_size: usize,
}

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}
impl RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        prof("norm", || {
            let x = x.to_dtype(DType::F32)?;
            let variance = x.powf(2.0)?.mean_keepdim(D::Minus1)?;
            let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
            x_normed.broadcast_mul(&self.weight)
        })
    }
}

struct Layer {
    wqkv: candle_nn::Linear,
    attn_wo: candle_nn::Linear,
    wi: candle_nn::Linear,
    mlp_wo: candle_nn::Linear,
    attn_norm: Option<RmsNorm>,
    mlp_norm: RmsNorm,
    // custom path
    p_wqkv: PackedW,
    p_attn_wo: PackedW,
    p_wi: PackedW,
    p_mlp_wo: PackedW,
}

struct Model {
    cfg: Config,
    emb: candle_nn::Embedding,
    emb_norm: RmsNorm,
    layers: Vec<Layer>,
    final_norm: RmsNorm,
    cos: Tensor,
    sin: Tensor,
}

fn tensor_vec(t: &Tensor) -> Vec<f32> {
    t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

impl Model {
    fn load(dir: &std::path::Path) -> Result<Self> {
        let cfg: Config = serde_json::from_str(&std::fs::read_to_string(dir.join("config.json"))?)?;
        let dev = Device::Cpu;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[dir.join("model.safetensors")], DType::F32, &dev)? };
        let h = cfg.hidden_size;
        let norm = |vb: VarBuilder| -> candle_core::Result<RmsNorm> { Ok(RmsNorm { weight: vb.get(h, "weight")?, eps: cfg.norm_eps }) };
        let emb = candle_nn::embedding(cfg.vocab_size, h, vb.pp("embeddings.tok_embeddings"))?;
        let emb_norm = norm(vb.pp("embeddings.norm"))?;
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            let lv = vb.pp(format!("layers.{i}"));
            let wqkv = candle_nn::linear_no_bias(h, 3 * h, lv.pp("attn").pp("Wqkv"))?;
            let attn_wo = candle_nn::linear_no_bias(h, h, lv.pp("attn").pp("Wo"))?;
            let wi = candle_nn::linear_no_bias(h, 2 * cfg.intermediate_size, lv.pp("mlp").pp("Wi"))?;
            let mlp_wo = candle_nn::linear_no_bias(cfg.intermediate_size, h, lv.pp("mlp").pp("Wo"))?;
            let p_wqkv = PackedW::pack(&tensor_vec(wqkv.weight()), 3 * h, h);
            let p_attn_wo = PackedW::pack(&tensor_vec(attn_wo.weight()), h, h);
            let p_wi = PackedW::pack_glu(&tensor_vec(wi.weight()), cfg.intermediate_size, h);
            let p_mlp_wo = PackedW::pack(&tensor_vec(mlp_wo.weight()), h, cfg.intermediate_size);
            layers.push(Layer {
                attn_norm: norm(lv.pp("attn_norm")).ok(),
                mlp_norm: norm(lv.pp("mlp_norm"))?,
                wqkv, attn_wo, wi, mlp_wo, p_wqkv, p_attn_wo, p_wi, p_mlp_wo,
            });
        }
        let final_norm = norm(vb.pp("final_norm"))?;
        // RotaryEmbedding::new (CPU = F32), theta 150000, max 32768 → only what we need (512).
        let head_dim = h / cfg.num_attention_heads;
        let inv: Vec<f32> = (0..head_dim).step_by(2).map(|i| 1f32 / (150000f64.powf(i as f64 / head_dim as f64) as f32)).collect();
        let inv = Tensor::new(inv.as_slice(), &dev)?;
        let t = Tensor::arange(0u32, 512u32, &dev)?.to_dtype(DType::F32)?.unsqueeze(1)?;
        let freqs = t.matmul(&inv.unsqueeze(0)?)?;
        Ok(Self { cos: freqs.cos()?, sin: freqs.sin()?, cfg, emb, emb_norm, layers, final_norm })
    }

    fn cos_sin(&self, l: usize) -> candle_core::Result<(Tensor, Tensor)> {
        prof("rope/copy", || {
            let cos = self.cos.narrow(0, 0, l)?;
            let sin = self.sin.narrow(0, 0, l)?;
            let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?.unsqueeze(0)?.unsqueeze(0)?.contiguous()?;
            let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?.unsqueeze(0)?.unsqueeze(0)?.contiguous()?;
            Ok((cos, sin))
        })
    }

    fn rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        prof("rope/copy", || {
            let d = x.dim(D::Minus1)?;
            let x1 = x.narrow(D::Minus1, 0, d / 2)?.contiguous()?;
            let x2 = x.narrow(D::Minus1, d / 2, d / 2)?.contiguous()?;
            let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
            x.contiguous()?.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?
        })
    }

    /// Attention core after the QKV projection (shared by both paths). qkv [B, L, 3H] → y [B, L, H].
    fn attention(&self, qkv: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let (b, l, _) = qkv.dims3()?;
        let h = self.cfg.hidden_size;
        let nh = self.cfg.num_attention_heads;
        let hd = h / nh;
        let (q, k, v) = prof("rope/copy", || {
            let q = qkv.narrow(2, 0, h)?.contiguous()?.reshape((b, l, nh, hd))?.transpose(1, 2)?.contiguous()?;
            let k = qkv.narrow(2, h, h)?.contiguous()?.reshape((b, l, nh, hd))?.transpose(1, 2)?.contiguous()?;
            let v = qkv.narrow(2, 2 * h, h)?.contiguous()?.reshape((b, l, nh, hd))?.transpose(1, 2)?.contiguous()?;
            Ok::<_, candle_core::Error>((q, k, v))
        })?;
        let q = Self::rope(&q, cos, sin)?;
        let k = Self::rope(&k, cos, sin)?;
        let kt = prof("rope/copy", || k.transpose(2, 3)?.contiguous())?;
        let att = prof("attn matmul", || q.contiguous()?.matmul(&kt)? / (hd as f64).sqrt())?;
        let att = prof("softmax", || candle_nn::ops::softmax(&att, D::Minus1))?;
        let y = prof("attn matmul", || att.contiguous()?.matmul(&v.contiguous()?))?;
        prof("rope/copy", || y.transpose(1, 2)?.contiguous()?.reshape((b, l, h)))
    }

    /// The app's forward_batch, op for op. ids [B*L] → final_norm hidden [B, L, H].
    fn forward_candle(&self, ids: &Tensor, b: usize, l: usize) -> candle_core::Result<Tensor> {
        let x = prof("embed", || self.emb.forward(ids)?.reshape((b, l, ())))?;
        let mut x = self.emb_norm.forward(&x)?;
        let (cos, sin) = self.cos_sin(l)?;
        for layer in &self.layers {
            let xn = match &layer.attn_norm {
                Some(n) => n.forward(&x)?,
                None => x.clone(),
            };
            let qkv = prof("linear", || layer.wqkv.forward(&xn))?;
            let y = self.attention(&qkv, &cos, &sin)?;
            let o = prof("linear", || layer.attn_wo.forward(&y))?;
            x = prof("residual", || &x + o)?;
            let xn = layer.mlp_norm.forward(&x)?;
            let wi = prof("linear", || layer.wi.forward(&xn))?;
            let act = prof("glu act", || {
                let d = wi.dim(D::Minus1)?;
                let gate = wi.narrow(D::Minus1, 0, d / 2)?.contiguous()?;
                let up = wi.narrow(D::Minus1, d / 2, d / 2)?.contiguous()?;
                (candle_nn::ops::silu(&gate)? * up)?.contiguous()
            })?;
            let o = prof("linear", || layer.mlp_wo.forward(&act))?;
            x = prof("residual", || &x + o)?;
        }
        self.final_norm.forward(&x)
    }

    /// Custom path: packed GEMMs, one-pass RMSNorm into a scratch buffer, Wi with the SiLU·up epilogue.
    fn forward_custom(&self, ids: &Tensor, b: usize, l: usize, blk: Blocking, threads: usize, fused_attn: bool) -> candle_core::Result<Tensor> {
        let h = self.cfg.hidden_size;
        let i_sz = self.cfg.intermediate_size;
        let m = b * l;
        let dev = Device::Cpu;
        let x = prof("embed", || self.emb.forward(ids))?;
        let mut x: Vec<f32> = prof("embed", || x.flatten_all()?.to_vec1::<f32>())?;
        let mut xn = vec![0f32; m * h];
        let eps = self.cfg.norm_eps as f32;
        prof("norm", || kern::rms_norm(&x, h, &tensor_vec(&self.emb_norm.weight), eps, &mut xn));
        std::mem::swap(&mut x, &mut xn);
        let (cos, sin) = self.cos_sin(l)?;
        let half = h / self.cfg.num_attention_heads / 2;
        let cos_v: Vec<f32> = self.cos.narrow(0, 0, l)?.flatten_all()?.to_vec1()?;
        let sin_v: Vec<f32> = self.sin.narrow(0, 0, l)?.flatten_all()?.to_vec1()?;
        debug_assert_eq!(cos_v.len(), l * half);
        let mut qkv = vec![0f32; m * 3 * h];
        let mut o = vec![0f32; m * h];
        let mut act = vec![0f32; m * i_sz];
        for layer in &self.layers {
            let src: &[f32] = match &layer.attn_norm {
                Some(n) => {
                    prof("norm", || kern::rms_norm(&x, h, &tensor_vec(&n.weight), eps, &mut xn));
                    &xn
                }
                None => &x,
            };
            prof("linear", || kern::gemm(src, m, &layer.p_wqkv, &mut qkv, blk, Epilogue::Store, threads));
            let y: Vec<f32> = if fused_attn {
                let mut y = vec![0f32; m * h];
                prof("attention (fused)", || kern::fused_attention(&qkv, b, l, self.cfg.num_attention_heads, h / self.cfg.num_attention_heads, &cos_v, &sin_v, &mut y, threads));
                y
            } else {
                let qkv_t = Tensor::from_slice(&qkv, (b, l, 3 * h), &dev)?;
                let y = self.attention(&qkv_t, &cos, &sin)?;
                prof("rope/copy", || y.flatten_all()?.to_vec1::<f32>())?
            };
            prof("linear", || kern::gemm(&y, m, &layer.p_attn_wo, &mut o, blk, Epilogue::Store, threads));
            prof("residual", || x.iter_mut().zip(&o).for_each(|(a, b)| *a += b));
            prof("norm", || kern::rms_norm(&x, h, &tensor_vec(&layer.mlp_norm.weight), eps, &mut xn));
            prof("linear", || kern::gemm(&xn, m, &layer.p_wi, &mut act, blk, Epilogue::SiluMul, threads));
            prof("linear", || kern::gemm(&act, m, &layer.p_mlp_wo, &mut o, blk, Epilogue::Store, threads));
            prof("residual", || x.iter_mut().zip(&o).for_each(|(a, b)| *a += b));
        }
        prof("norm", || kern::rms_norm(&x, h, &tensor_vec(&self.final_norm.weight), eps, &mut xn));
        Tensor::from_vec(xn, (b, l, h), &dev)
    }
}

/// Mean pool + L2 normalize per row ([B, L, H] → B vectors), like embed_many's exact buckets.
fn pool(hidden: &Tensor) -> Vec<Vec<f32>> {
    let (_, l, _) = hidden.dims3().unwrap();
    let mean = (hidden.sum(1).unwrap() / l as f64).unwrap();
    let norm = mean.sqr().unwrap().sum_keepdim(1).unwrap().sqrt().unwrap();
    mean.broadcast_div(&norm).unwrap().to_vec2::<f32>().unwrap()
}

fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    d / (na.sqrt() * nb.sqrt())
}

fn ids_for(b: usize, l: usize, vocab: usize, salt: u32) -> Vec<u32> {
    let mut x = 0x9E37_79B9u32 ^ salt.wrapping_mul(2654435761);
    (0..b * l)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            5 + x % (vocab as u32 - 5)
        })
        .collect()
}

fn time_it(reps: usize, mut f: impl FnMut()) -> f64 {
    f(); // warm-up
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() / reps as f64
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::PathBuf::from(args.get(1).expect("usage: cpu-embed-bench <granite dir> [--threads N] [--quick]"));
    let quick = args.iter().any(|a| a == "--quick");
    let threads = args.iter().position(|a| a == "--threads").and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok())
        .unwrap_or_else(candle_core::utils::get_num_threads);
    let pool_rayon = rayon::ThreadPoolBuilder::new().num_threads(threads).build()?;
    let model = Model::load(&dir)?;
    let mock = dir.join("MOCK").exists();
    println!(
        "[CPU-3-2] model {} ({}) · arch {} · threads {} · candle threads {}",
        dir.display(), if mock { "mock" } else { "real" }, std::env::consts::ARCH, threads, candle_core::utils::get_num_threads()
    );

    // Workloads (B rows × L tokens = one exact-length micro-batch of the CPU dynamic mode).
    let workloads: Vec<(&str, usize, usize)> = vec![
        ("short phrase B=1 L=12 (single call)", 1, 12),
        ("short phrases B=85 L=12 (token_budget 1024)", 85, 12),
        ("query n-grams B=64 L=6", 64, 6),
        ("mid texts B=16 L=32", 16, 32),
        ("mid texts B=8 L=64", 8, 64),
        ("mid text B=1 L=96", 1, 96),
        ("long text B=1 L=256", 1, 256),
        ("long texts B=4 L=256", 4, 256),
        ("chunk B=2 L=512", 2, 512),
    ];
    let reps = |b: usize, l: usize| if quick { 2 } else { (4000 / (b * l)).clamp(3, 40) };
    let vocab = model.cfg.vocab_size;

    // ── 1. profile of the candle path (the app's ops) ──
    println!("\n## 1. profile — candle path (op categories, % of forward time)");
    pool_rayon.install(|| -> Result<()> {
        for &(name, b, l) in &workloads {
            let ids = Tensor::from_vec(ids_for(b, l, vocab, 1), b * l, &Device::Cpu)?;
            PROF.lock().unwrap().clear();
            PROF_ON.store(true, std::sync::atomic::Ordering::Relaxed);
            let r = reps(b, l);
            let t = time_it(r, || {
                model.forward_candle(&ids, b, l).unwrap();
            });
            PROF_ON.store(false, std::sync::atomic::Ordering::Relaxed);
            let g = PROF.lock().unwrap();
            let total: f64 = g.values().map(|v| v.0).sum();
            let ops: usize = g.values().map(|v| v.1).sum::<usize>() / (r + 1);
            let mut parts: Vec<String> = g.iter().map(|(k, v)| format!("{k} {:.0}%", 100.0 * v.0 / total)).collect();
            parts.sort();
            println!("- {name}: {:.2} ms/forward · {:.1} µs/token · {} timed op groups/forward | {}", t * 1e3, t * 1e6 / (b * l) as f64, ops, parts.join(" · "));
        }
        Ok(())
    })?;

    // ── 2. GEMM blocking sweep (Linear shapes of one layer) ──
    println!("\n## 2. GEMM: candle/gemm vs packed custom kernel (per call, µs) · MC/KC/NC sweep");
    let shapes = [("Wqkv 384→1152", 384usize, 1152usize), ("attn Wo 384→384", 384, 384), ("Wi 384→3072", 384, 3072), ("mlp Wo 1536→384", 1536, 384)];
    let blockings = [
        Blocking { mc: 64, kc: 1 << 20, nc: 64 },
        Blocking { mc: 128, kc: 1 << 20, nc: 64 },
        Blocking { mc: 256, kc: 1 << 20, nc: 128 },
        Blocking { mc: 128, kc: 256, nc: 64 },
        Blocking { mc: 512, kc: 1 << 20, nc: 256 },
    ];
    pool_rayon.install(|| -> Result<()> {
        for &m in &[12usize, 256, 1020] {
            for &(name, k, n) in &shapes {
                let x = ids_for(m, k, 1000, 7).into_iter().map(|v| v as f32 / 1000.0 - 0.5).collect::<Vec<f32>>();
                let w = ids_for(n, k, 1000, 9).into_iter().map(|v| (v as f32 / 1000.0 - 0.5) * 0.1).collect::<Vec<f32>>();
                let xt = Tensor::from_slice(&x, (m, k), &Device::Cpu)?;
                let wt = Tensor::from_slice(&w, (n, k), &Device::Cpu)?;
                let lin = candle_nn::Linear::new(wt, None);
                let r = if quick { 3 } else { (20_000_000 / (m * k * n)).clamp(5, 200) };
                let t_candle = time_it(r, || {
                    lin.forward(&xt).unwrap();
                });
                let reference = lin.forward(&xt)?.flatten_all()?.to_vec1::<f32>()?;
                let pw = PackedW::pack(&w, n, k);
                let mut out = vec![0f32; m * n];
                let mut best = (f64::MAX, Blocking::default());
                let mut cells = Vec::new();
                for blk in blockings {
                    let t = time_it(r, || kern::gemm(&x, m, &pw, &mut out, blk, Epilogue::Store, threads));
                    cells.push(format!("{}/{}/{} {:.0}", blk.mc, if blk.kc > k { k } else { blk.kc }, blk.nc, t * 1e6));
                    if t < best.0 {
                        best = (t, blk);
                    }
                }
                kern::gemm(&x, m, &pw, &mut out, best.1, Epilogue::Store, threads);
                let maxd = out.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                println!(
                    "- M={m:4} {name:16}: candle {:.0} µs · custom best {:.0} µs ({:.2}×) · max|Δ| vs candle {:.1e} · [MC/KC/NC µs] {}",
                    t_candle * 1e6, best.0 * 1e6, t_candle / best.0, maxd, cells.join(", ")
                );
            }
        }
        Ok(())
    })?;

    // ── 3. bit-identity contract of the custom kernel (M, position, threads) ──
    println!("\n## 3. custom kernel bit-identity (row = f(row) only)");
    {
        let (k, n, mmax) = (384usize, 1152usize, 300usize);
        let x = ids_for(mmax, k, 1000, 3).into_iter().map(|v| v as f32 / 1000.0 - 0.5).collect::<Vec<f32>>();
        let w = ids_for(n, k, 1000, 4).into_iter().map(|v| (v as f32 / 1000.0 - 0.5) * 0.1).collect::<Vec<f32>>();
        let pw = PackedW::pack(&w, n, k);
        let mut full = vec![0f32; mmax * n];
        pool_rayon.install(|| kern::gemm(&x, mmax, &pw, &mut full, Blocking::default(), Epilogue::Store, threads));
        let mut bad = 0usize;
        for m in 1..=mmax {
            for (ti, blk) in [(1usize, Blocking { mc: 64, kc: 128, nc: 32 }), (threads, Blocking { mc: 7, kc: 1 << 20, nc: 512 })] {
                let mut out = vec![0f32; m * n];
                kern::gemm(&x[..m * k], m, &pw, &mut out, blk, Epilogue::Store, ti);
                if out.iter().zip(&full).any(|(a, b)| a.to_bits() != b.to_bits()) {
                    bad += 1;
                }
            }
        }
        // position: reversed rows
        let xr: Vec<f32> = (0..mmax).rev().flat_map(|r| x[r * k..(r + 1) * k].to_vec()).collect();
        let mut outr = vec![0f32; mmax * n];
        kern::gemm(&xr, mmax, &pw, &mut outr, Blocking::default(), Epilogue::Store, threads);
        let pos_bad = (0..mmax).filter(|&r| outr[(mmax - 1 - r) * n..(mmax - r) * n].iter().zip(&full[r * n..(r + 1) * n]).any(|(a, b)| a.to_bits() != b.to_bits())).count();
        println!("- M=1..=300 × (1 thread, MC64/KC128/NC32) and ({threads} threads, MC7/NC512): mismatching runs {bad} · reversed-row mismatches {pos_bad}/300 (including M=1: no gemv special case)");
        assert_eq!(bad + pos_bad, 0, "custom kernel broke the bit-identity contract");
    }

    // ── 4. end-to-end: candle path vs custom path ──
    println!("\n## 4. end-to-end forward: candle path vs custom kernels (ms/forward) · accuracy · composition invariance");
    pool_rayon.install(|| -> Result<()> {
        let blk = Blocking::default();
        for &(name, b, l) in &workloads {
            let ids_v = ids_for(b, l, vocab, 11);
            let ids = Tensor::from_vec(ids_v.clone(), b * l, &Device::Cpu)?;
            let r = reps(b, l);
            let t_c = time_it(r, || {
                model.forward_candle(&ids, b, l).unwrap();
            });
            let t_k = time_it(r, || {
                model.forward_custom(&ids, b, l, blk, threads, false).unwrap();
            });
            let vc = pool(&model.forward_candle(&ids, b, l)?);
            let t_f = time_it(r, || {
                model.forward_custom(&ids, b, l, blk, threads, true).unwrap();
            });
            let vk = pool(&model.forward_custom(&ids, b, l, blk, threads, false)?);
            let vf = pool(&model.forward_custom(&ids, b, l, blk, threads, true)?);
            let min_cos_f = vc.iter().zip(&vf).map(|(a, b)| cos(a, b)).fold(1f64, f64::min);
            let min_cos = vc.iter().zip(&vk).map(|(a, b)| cos(a, b)).fold(1f64, f64::min);
            // composition: row 0 alone vs in the batch (custom path)
            let ids0 = Tensor::from_vec(ids_v[..l].to_vec(), l, &Device::Cpu)?;
            let v0 = pool(&model.forward_custom(&ids0, 1, l, blk, threads, false)?);
            let same0 = v0[0].iter().zip(&vk[0]).all(|(a, b)| a.to_bits() == b.to_bits());
            let f0 = pool(&model.forward_custom(&ids0, 1, l, blk, 1, true)?);
            let same0f = f0[0].iter().zip(&vf[0]).all(|(a, b)| a.to_bits() == b.to_bits());
            let c0 = pool(&model.forward_candle(&ids0, 1, l)?);
            let same0c = c0[0].iter().zip(&vc[0]).all(|(a, b)| a.to_bits() == b.to_bits());
            println!(
                "- {name}: candle {:.2} ms · custom GEMM+fused norm/GLU {:.2} ms ({:.2}×) · + fused attention {:.2} ms ({:.2}×) | min cos vs candle {:.7} / {:.7} | row0 alone (1 thread) == in batch: custom {} · +fused attn {} · candle {}",
                t_c * 1e3, t_k * 1e3, t_c / t_k, t_f * 1e3, t_c / t_f, min_cos, min_cos_f, same0, same0f, same0c
            );
        }
        Ok(())
    })?;
    // custom-path profile for the short-phrase batch
    pool_rayon.install(|| -> Result<()> {
        for &(name, b, l) in &[workloads[1], workloads[7]] {
            let ids = Tensor::from_vec(ids_for(b, l, vocab, 1), b * l, &Device::Cpu)?;
            PROF.lock().unwrap().clear();
            PROF_ON.store(true, std::sync::atomic::Ordering::Relaxed);
            let t = time_it(reps(b, l), || {
                model.forward_custom(&ids, b, l, Blocking::default(), threads, true).unwrap();
            });
            PROF_ON.store(false, std::sync::atomic::Ordering::Relaxed);
            let g = PROF.lock().unwrap();
            let total: f64 = g.values().map(|v| v.0).sum();
            let parts: Vec<String> = g.iter().map(|(k, v)| format!("{k} {:.0}%", 100.0 * v.0 / total)).collect();
            println!("- custom+fused-attn profile {name}: {:.2} ms | {}", t * 1e3, parts.join(" · "));
        }
        Ok(())
    })?;
    Ok(())
}

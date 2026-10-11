# Commerce · Trade & Logistics · Website Analytics

> **Local-first AI Operations Platform** — An open-source operational tool where a browser extension and desktop client work seamlessly together to structure, vectorize, and search e-commerce data, trade & logistics documents, and website behavior logs **completely offline**.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [AI Model Architecture](#ai-model-architecture)
- [Language-Specialized Engine (AlphaEdge Qwen3.5-4B)](#language-specialized-engine-alphaedge-qwen35-4b)
- [Memory Management (RAM / SSD Offloading)](#memory-management-ram--ssd-offloading)
- [Build & Run](#build--run)
- [License](#license)

---

## Overview

| Domain | Description |
|--------|------|
| **Commerce** | Automatically structures order, goods, tracking, coupon, and review pages of shopping malls and indexes them into a local vector DB. |
| **Trade & Logistics** | Extracts 55 types of trade forms (B/L, C/I, P/L, L/C, etc.) using SigLIP2 Vision + Qwen3.5 2B and builds a reference graph between documents. |
| **Website Analytics** | Collects user click, hover, and input events to summarize behavioral flows, preferences, and conversion funnels into natural language reports. |

All inference, embedding, and storage are performed **locally on the client**, with no external API calls. For non-Latin document languages, a **language-specialized Qwen3.5-4B engine from [alphaedge-ai](https://huggingface.co/alphaedge-ai)** is downloaded on demand, converted to a local Q4_K_M GGUF once, and routed into transliteration, synthesis, and field extraction steps when it is a better fit than the base models.

---

## Key Features

### Commerce Pipeline
- **Page Classification** — Automatically determines 6 types (`order / goods / tracking / review / coupon / event`) using vector cosine + structural signals (repeated rows, forms, headings).
- **List Extraction** — Boa engine DOM traversal → LLM field matching → mutually exclusive assignment (1:1) for column-value mapping.
- **Detail Extraction** — Structural label-value pairs + field-by-field extraction using Qwen3 0.6B.
- **Chunk Indexing** — Natural language segmentation → attribute tagging (PLINKO) → 384-dimensional tri-synthetic vectors → LanceDB `item_chunks`.
- **Transliteration Aliases** — Additionally stores bidirectional transliteration vectors for multilingual cross-lingual search. Uses the AlphaEdge Qwen3.5-4B model of the document language when ready, otherwise Qwen3.5 2B + pronunciation gate.
- **Language Routing** — Per-step router (`list_field / detail_field / list_synthesis / detail_synthesis`) decides between Qwen3 0.6B and the AlphaEdge 4B engine using script ratio, tokenizer fit, VRAM/RAM headroom, and observed pass rates.

### Trade & Logistics Pipeline
- **Document Classification** — SigLIP2 patches × text anchor cosine → 2-depth judgment for groups (7 types) → codes (55 types).
- **Heatmap Cropping** — Field anchors by category → patch cosine → connected components → exclusive assignment → pixel crop.
- **Precision Extraction** — Cropped image × schema prompt → Qwen3.5 2B JSON output.
- **Value Grounding Validation** — Extracted value × patch embedding cosine → hallucination filtering based on extreme value theory.
- **Document Relay** — Connects a graph of 55 forms using reference axes such as `reference_invoice / reference_bl / reference_lc`.

### Website Analytics Pipeline
- **Event Collection** — `content.js` loads `click / hover / change` outerHTML into D1.
- **Structuring** — The 0.6B model generates a natural language summary of behavioral intent and surrounding context.
- **Reporting** — Synthesizes multi-event sequences into reports along 3 axes: `cross_action_flow / intent_evolution / consistent_preferences`.

---

## Project Structure


```

cron-logis-center/
├── commerce/                           # Desktop client (Tauri v2 + Vite/TS)
│   ├── src/                            # Frontend (TypeScript)
│   ├── mobile/                         # Mobile companion app (Tauri)
│   ├── landing/                        # Landing page
│   ├── build_cargo*.bat / dev_cargo*.bat / test_cargo*.bat  # CUDA / ROCm / Vulkan scripts
│   ├── test_linux.sh                   # Linux integration tests (vulkan|cuda|rocm)
│   ├── crates/
│   │   ├── boa-engine/                 # Boa JS engine (local build)
│   │   ├── onnxruntime/                # ONNX Runtime bindings (local build)
│   │   ├── onnxruntime-sys/            # ONNX Runtime FFI
│   │   └── direct-storage/             # DirectStorage (Windows)
│   └── src-tauri/
│       ├── Cargo.toml                  # Rust dependencies & feature flags (cuda/rocm/vulkan/metal)
│       ├── tests/it/                   # Integration tests (cargo test --test it)
│       └── src/
│           ├── lib.rs                  # Tauri entry, IPC command registration, model downloads
│           ├── scheduler.rs            # Background task scheduler (main pipeline)
│           ├── scheduler/
│           │   ├── worker.rs           # Task worker loop
│           │   ├── trading.rs          # Trade & logistics pipeline
│           │   ├── translit.rs         # Transliteration alias generation
│           │   ├── indexing.rs         # Chunk indexing
│           │   ├── entity.rs           # Entity key normalization (commerce/trade)
│           │   ├── list_census.rs      # List page census
│           │   └── relay_ledger.rs     # Document relay ledger
│           ├── model.rs                # Model manager (ModelSize, residency budgets)
│           ├── model/
│           │   ├── lifecycle.rs        # Load/unload, VRAM crossover, lang_route()
│           │   ├── lang_llm.rs         # AlphaEdge Qwen3.5-4B: language codes, download, routing
│           │   ├── lang_gguf.rs        # safetensors → Q4_K_M GGUF runtime conversion
│           │   ├── inference.rs        # Generation entry points
│           │   ├── vision.rs           # Vision inference
│           │   ├── merge.rs            # Result merging
│           │   ├── research.rs         # Research flow
│           │   └── query_{commerce,shipping,analytic}.rs  # Domain search queries
│           ├── logic.rs                # Business logic (relay, state, trade rules)
│           ├── store.rs                # LanceDB vector store (envelope schema)
│           ├── automation.rs           # Chromium browser automation
│           ├── analytic.rs             # Website analytics tasks
│           ├── stanza.rs               # Stanza NLP (tokenization/POS/lemma/syntax)
│           ├── prompts.rs              # Prompt templates
│           ├── models/
│           │   ├── mod.rs
│           │   ├── common/             # Common layer (GateUpDownMLP, Attention, GGUF)
│           │   ├── qwen/               # Qwen VL (Vision + Language), KV offloading
│           │   ├── qwen3/              # Qwen3 0.6B (Text only)
│           │   ├── qwen3_5/            # Qwen3.5 2B / AlphaEdge 4B (shared runtime)
│           │   ├── qwen3vl/            # Qwen3 VL (Next-gen Vision)
│           │   ├── siglip2/            # SigLIP2 Vision Encoder
│           │   │   ├── mod.rs          # Model load (Text/Vision separated)
│           │   │   ├── vision.rs       # Vision Transformer
│           │   │   ├── text.rs         # Text encoder
│           │   │   ├── preprocessor.rs # NaFlex preprocessing
│           │   │   ├── vision_encoder.rs # Heatmap/Classification pipeline
│           │   │   ├── vision_crop.rs  # NMS crop + Tile splitting
│           │   │   ├── nms_arena.rs    # NMS arena allocator
│           │   │   ├── layout_cache.rs # Layout heatmap cache
│           │   │   ├── phrase_cache.rs # Text anchor embedding cache
│           │   │   ├── legibility.rs   # Legibility map (blur/margin detection)
│           │   │   ├── value_grounding.rs # Value grounding validation
│           │   │   └── tokenizer.rs    # SigLIP2 tokenizer
│           │   ├── granite/            # Granite MoE model
│           │   ├── embedding.rs        # granite-embedding-97m (384d)
│           │   └── vision_cache.rs     # ViT output disk cache
│           ├── utils/
│           │   ├── parsing.rs          # HTML→PUG conversion, schema management
│           │   ├── bias_schema.rs      # bias.json access layer
│           │   ├── bias.json / full_bias.json # Multilingual field anchor/bias dictionary
│           │   ├── ai_utils.rs         # Cosine, exclusive assignment, format gating
│           │   ├── score_dynamics.rs   # Adaptive baselines (SDS pass rates for routing)
│           │   ├── json_parse.rs       # LLM JSON parsing/recovery
│           │   ├── nl_convert.rs       # JSON→Natural language, chunk segmentation
│           │   ├── direct_loader.rs    # DirectStorage / io_uring KV block I/O
│           │   ├── device_utils.rs     # Device/backend detection
│           │   ├── resources.rs        # RAM/VRAM resource probes
│           │   ├── tensor_utils.rs     # Tensor helpers
│           │   ├── compression.rs      # Compression helpers
│           │   ├── crypto.rs           # Data / KV block encryption
│           │   ├── img_utils.rs        # Image resize/patch
│           │   ├── json_utils.rs       # JSON merge
│           │   ├── pug_utils.rs        # PUG grid parser
│           │   ├── hash.rs             # ID hashing, CRC32
│           │   ├── canonical.rs        # Type normalization rules
│           │   ├── time_guide.rs       # Time intent parsing
│           │   ├── lang_utils.rs       # Language detection
│           │   └── paths.rs / logger.rs / metrics.rs / network.rs / url_utils.rs / ...
│           ├── parsers/
│           │   └── mod.rs              # PDF/Excel/Word/HWP/CSV text extraction
│           ├── chat_template/          # ChatML template
│           ├── tokenizer/              # GGUF tokenizer wrapper
│           ├── position_embed/         # RoPE / sinusoidal positional encoding
│           ├── openai_types.rs         # OpenAI compatible type definitions
│           └── js_templates.rs         # Boa engine JS templates
├── apac/                               # Cloudflare Worker + D1 backend
├── proxy/                              # Cloudflare Worker + D1 backend (proxy)
├── SyntheticData/trading/              # Synthetic trade documents (YAML → PDF generators)
├── logic.md                            # Pipeline logic notes
└── readme.md

```

### Model File Paths


```

{AppData}/logis-center/models/
├── Qwen3-0.6B-Instruct-gguf/       # Text extraction (List/Detail)
│   ├── Qwen3-0.6B-Q8_0.gguf
│   ├── Qwen3-0.6B-Q4_K_M.gguf
│   ├── config.json
│   ├── tokenizer.json
│   └── generation_config.json
├── Qwen3.5-2B-Instruct-gguf/       # Precision extraction / Transliteration / Classification
│   ├── Qwen3.5-2B-Q8_0.gguf
│   ├── mmproj-BF16.gguf            # Vision projector
│   ├── config.json
│   ├── tokenizer.json
│   └── generation_config.json
├── granite-embedding-97m-multilingual-r2/  # 384d Multilingual Embedding
│   ├── model.safetensors
│   ├── config.json
│   └── tokenizer.json
├── siglip2-so400m-patch16-naflex/   # Vision similarity encoder
│   ├── model.safetensors
│   ├── config.json
│   ├── preprocessor_config.json
│   └── tokenizer.json
├── granite-4.0-h-350m/             # Granite MoE (Optional)
│   └── model.safetensors
├── Qwen3.5-4B-{code}-16384/        # AlphaEdge language model (on demand, e.g. kor / jpn / zho / rus)
│   ├── model.safetensors           # Original weights (~7.9GB, kept as-is)
│   ├── config.json
│   ├── tokenizer.json              # 16,384-token language-specific vocab
│   ├── tokenizer_config.json / chat_template.jinja / merges.txt / vocab.json
│   └── runtime-q4km/               # Generated locally on first use
│       ├── model-q4km.gguf         # Q4_K_M runtime weights
│       └── READY                   # Conversion recipe key (re-converts if it changes)
└── stanza/                          # NLP Pipeline
├── ko/
│   ├── vocab.json
│   ├── tokenizer.onnx
│   ├── pos.onnx
│   ├── lemma.onnx
│   └── depparse.onnx
├── en/
├── ja/
└── zh-hans/

```

---

## AI Model Architecture

### Model Roles

| Model | Size | Role | Inference Timing |
|------|------|------|-----------|
| **Qwen3 0.6B** | ~600MB (Q8) | List/detail field extraction, page classification, behavior structuring | Every task |
| **Qwen3.5 2B** | ~2.1GB (Q8) | Precision extraction, transliteration fallback, document classification tie-breaks | As needed |
| **AlphaEdge Qwen3.5-4B-{code}** | ~7.9GB download → Q4_K_M runtime | Language-specialized transliteration, list/detail synthesis and field extraction for non-Latin documents | On demand (routed) |
| **granite-embedding-97m** | ~380MB | 384-dimensional multilingual embeddings (vector search) | Indexing/Search |
| **SigLIP2** | ~2.2GB | Vision patch × text anchor cosine (Heatmap) | Trade documents |
| **Stanza** | ~50MB/lang | Tokenization, POS, Lemma, Dependency syntax | Transliteration preprocessing |

### Inference Pipeline Flow


```

[Task Queue] → [Scheduler]
├── Commerce: PUG conversion → Page classification → Field extraction → Synthesis → Translit → Chunk indexing
│             (each LLM step: lang_route() → AlphaEdge 4B or Qwen3 0.6B / Qwen3.5 2B)
├── Trading:  PDF split → Document classification → Heatmap → Crop → Precision extraction → Grounding validation
└── Analytic: Event collection → Structuring → Report synthesis → Vector indexing

```

### KV Cache Architecture


```

┌─────────────────────────────────────────────────────────┐
│                    KV Cache Hierarchy                   │
├─────────────┬───────────────────────────────────────────┤
│   VRAM      │ FP8 Compression (1byte/elem)              │
│   (GPU)     │ 1024 tokens per block × 28 layers         │
│             │ Vision token ratio metadata tag           │
├─────────────┼───────────────────────────────────────────┤
│   RAM       │ BF16/F32 (BitKV Metadata)                 │
│   (CPU)     │ 8-bit quantization + Scale metadata       │
│             │ Includes original shape recovery info     │
├─────────────┼───────────────────────────────────────────┤
│   SSD       │ SafeTensors (b{offset}/l{layer}.st)       │
│   (Disk)    │ Distributed storage per layer             │
│             │ layer{N}_meta.json index                  │
└─────────────┴───────────────────────────────────────────┘

```

---

## Language-Specialized Engine (AlphaEdge Qwen3.5-4B)

Base models (Qwen3 0.6B / Qwen3.5 2B) share a large multilingual vocabulary, so non-Latin text (Korean, Japanese, Thai, Cyrillic, Devanagari, …) is split into many tokens and transliteration quality drops. The project therefore uses **per-language Qwen3.5-4B models published by [alphaedge-ai](https://huggingface.co/alphaedge-ai)**, each with a compact **16,384-token vocabulary** tuned for one language.

| Item | Value | Code |
|------|-------|------|
| Repository | `alphaedge-ai/Qwen3.5-4B-{code}-16384` | `lang_llm::repo_id()` |
| Language codes | 83 ISO 639-3 codes (`kor`, `jpn`, `zho`, `rus`, `tha`, `hin`, …); ISO 639-1 / locale tags are mapped automatically (`ko` → `kor`, `zh-hans` → `zho`) | `lang_llm::alphaedge_code()` |
| Runtime format | safetensors → **Q4_K_M GGUF** (`runtime-q4km/model-q4km.gguf`), converted once in-process (no Python) | `lang_gguf::prepare()` |
| Inference | Text-only, shares the Qwen3.5 runtime (`Qwen3_5GenerateModel::init_from_gguf`) with a no-think ChatML template | `lang_gguf::load_from()` |
| Labels in logs | `[LANG-LLM]`, `[TRANSLIT ENGINE]`, `[LANG ROUTE]`, `[CROSSOVER]` | |

### Lifecycle

```
Document language detected
  └─ prefetch(doc_lang)                 # only for non-Latin languages
       └─ background download (resumable, 4 attempts/file, 10-min cooldown on failure)
            └─ Q4_K_M GGUF conversion   # once; READY file stores the recipe key
                 └─ is_ready(code) ✔ → engine becomes routable from the next cache miss
```

While the model is downloading or converting, every step keeps running on **Qwen3.5 2B + pronunciation gate** — tasks never block on the 4B engine.

### Routing (`ModelManager::lang_route` → `lang_llm::route_verdict`)

A step is sent to the 4B engine only if every gate passes; otherwise it falls back to the base model and the reason is logged.

| Gate | Rule |
|------|------|
| Step contract | `search.commerce_verify` stays on Qwen3 (must copy English identifiers verbatim) |
| Script | Document language must be non-Latin; ≥ 8 letters, native script ≥ 60 %, third scripts ≤ 2 % |
| Tokenizer fit | 0 characters lost on round-trip; token count ≤ 1.5 × the reference tokenizer |
| Resources | GPU only (CPU mode skips per-item steps); `list_field` / `detail_field` need > 6 GB free RAM; VRAM must hold the 4B runtime (synthesis steps may swap out the embedding model) |
| Quality (SDS) | Recent 4B pass rate must stay ≥ Qwen3 pass rate − 0.10 (or ≥ 0.55 with no baseline); closed routes are re-probed every 10th call |
| Session | One 4B language resident at a time; runtime failures disable it until restart, VRAM shortages cool down for 120 s |

Outcomes are recorded per step (`{track}.lang4b_pass.{step}` vs `{track}.qwen3_pass.{step}`) in `score_dynamics`, so routing adapts to observed quality.

---

## Memory Management — RAM / SSD Offloading

### Design Philosophy

Processing a 32,768+ token context with a 0.6B model requires gigabytes of memory just for the KV cache.
This project implements a **3-tier memory hierarchy** to handle long-context reasoning even on limited VRAM.

### Storage Formats by Tier

| Tier | Format | Size/Token | Purpose |
|------|------|-----------|------|
| VRAM | FP8 (F8E4M3) | ~4KB/token/layer | Active inference |
| RAM | BF16/F32 | ~8KB/token/layer | Recent past context |
| SSD | SafeTensors | ~8KB/token/layer | Long-term retention |

### KV Block Management


```

┌─────────────────────────────────────────────────┐
│             KVRegistry (Central Index)          │
├─────────────────────────────────────────────────┤
│  entries[0]: offset=0,    len=1024, loc=[SSD×28] │
│  entries[1]: offset=1024, len=1024, loc=[RAM×28] │
│  entries[2]: offset=2048, len=1024, loc=[VRAM×28]│
│  entries[3]: offset=3072, len=512,  loc=[VRAM×28]│ ← Active block
└─────────────────────────────────────────────────┘

```

- **Block Size**: 1024 tokens
- **Layer Count**: 28 (based on the 0.6B model)
- **Location Tracking**: `KVLocation::{VRAM, RAM, SSD, Loading, Streaming}`
- **Vision Tag**: `vision_token_ratio` — Metadata for image patch ratio (lossless)

### Offloading Triggers

| Condition | Action | Code Location |
|------|------|-----------|
| > 8 VRAM blocks | Evacuate oldest block → RAM | `evacuate_vram_to_ram_only()` |
| Prefill chunk boundary | Entire current layer blocks → RAM | `evacuate_layer_kv_to_cpu()` |
| Session end/Snapshot | All active blocks → SSD | `force_flush_all_active_blocks()` |
| Reuse needed during decoding | Sequentially restore SSD → RAM → VRAM | `batch_load_layer_kv()` |

### SSD Storage Path Structure


```

{AppData}/logis-center/kv_cache/
└── {session_id}/
├── inference/
│   ├── text/
│   │   ├── b0/
│   │   │   ├── l0.st        # Layer 0 KV (SafeTensors)
│   │   │   ├── l1.st
│   │   │   ├── ...
│   │   │   └── l27.st
│   │   ├── b1024/
│   │   │   └── ...
│   │   └── b2048/
│   │       └── ...
│   └── layer{N}_meta.json   # Block index per layer
└── reference/
└── text/
└── (Same structure)

```

### SafeTensors KV Storage Format


```

File: b{offset}/l{layer}.st
Tensors:

* "b{offset}_l{layer}_k_data"   : K Cache [1, kv_heads, seq_len, head_dim]
* "b{offset}_l{layer}_v_data"   : V Cache [1, kv_heads, seq_len, head_dim]
* "b{offset}_l{layer}_k_shape"  : Original shape [4] (u32)

```

### RAM Offloading (BitKV Compression)

```rust
// 8-bit Quantized Compression
struct BitKVMetadata {
    k_data: Tensor,          // 8-bit packed K
    v_data: Tensor,          // 8-bit packed V
    original_shape: Vec<usize>,  // Original shape for restoration
}

```

* **Compression**: `(value / scale).round() as i8` → 1byte/element
* **Restoration**: `packed as f32 * scale`
* **Location**: `RegistryEntry.bitkv_cache` (Arc<RwLock<Vec<Option>>>)

### Memory Slot Manager (SlotManager)

```rust
// VRAM Slot Management — 128 slot pool
pub struct MemorySlot {
    id: usize,
    state: AtomicU8,          // 0:Free, 1:Baking, 2:Ready, 3:Loading
    k_layers: Vec<Arc<Mutex<Option<Tensor>>>>,
    v_layers: Vec<Arc<Mutex<Option<Tensor>>>>,
    remaining_layers: AtomicUsize,
}

```

* **Slot Count**: 128 (Concurrent active block management)
* **State Transition**: Free → Computing → Transferring → Compressing → Saving → Ready
* **Backpressure**: Blocks if queue exceeds 64 (prevents deadlocks)

### Layer JIT Loading (mmap)

```rust
// Do not preload weights; read from mmap when needed
pub fn reload_layer(&mut self, layer_idx: usize) -> Result<()> {
    // Read directly from mmap (Page-fault based)
    // 1-byte dummy tensor → In-place replacement with actual weights
}

```

* **Benefits**: Reduces initial load time, unutilized layers don't occupy memory.
* **Path**: `QuantizedQwenVLTextModel.reload_layer()`
* **Ping-Ponging**: While computing the current layer, the next layer is preloaded in the background.

### Adaptive Resolution Based on Remaining VRAM

```rust
// Dynamically adjusts max resolution based on available VRAM during image processing
fn compute_adaptive_max_pixels(&self, config_max: u32) -> u32 {
    // mem(N) = A·N + B·N²  Inverse quadratic
    // The positive root of B·N² + A·N - usable = 0 is the safe patch count
    const A: f64 = 24_000.0;   // Linear term
    const B: f64 = 4.0;        // Quadratic term (Attention buffer)
    const RESERVE: u64 = 800MB; // Headroom
}

```

### Offloading Environment Variables & Settings

| Setting | Default | Description |
| --- | --- | --- |
| `hard_token_limit` | 4096 | KV cache max tokens (×40,000 = byte budget) |
| `pinned_layer_count` | 28 (GPU) / 0 (CPU) | Number of layers kept resident in VRAM |
| `DECODE_HEADROOM_TOKENS` | 2048 | Expected additional tokens during decoding |
| `IO_BACKPRESSURE_WATERMARK` | 64 | Threshold for I/O queue backpressure |
| `VISION_VRAM_RESERVE` | 800MB | VRAM reserved for vision processing |

### Offloading Usage Examples

```rust
// 1. Prefill (Processes long context in chunks)
model.prefill_only(text, cancel, session_id, kv_name).await?;
// → Processes in 2048 token chunks, moving past blocks → RAM/SSD at chunk boundaries

// 2. Decoding (Generates 1 token at a time, restoring past blocks from SSD if needed)
model.generate(params, cancel, session_id, kv_name, ignore, prejudice).await?;
// → If KV blocks run out, sequentially restores SSD → RAM → VRAM
// → If scheduled for VRAM residency, utilizes directly from VRAM without restoration

// 3. Save Session Snapshot
model.force_flush_all_active_blocks(session_id, kv_name).await?;
// → Saves all active blocks to SSD as SafeTensors

// 4. Restore Session
model.load_kv_cache(path, device, expected_len, refill_len, kv_name).await?;
// → Reads block index from SSD → Loads sequentially per layer

```

### Memory Optimization Checklist

* [x] Layer-wise JIT Loading (Unused layers occupy no memory)
* [x] KV Cache FP8 Compression (VRAM) / 8-bit BitKV (RAM)
* [x] Block-level SSD Offloading (1024 tokens/block)
* [x] Prefill chunk segmentation (2048 tokens/chunk)
* [x] Vision weights JIT load/unload (Saves 600MB on text-only paths)
* [x] Adaptive image resolution (Based on available VRAM)
* [x] I/O Backpressure (Blocks when queue > 64)
* [x] OS-level working set trimming (At every prefill boundary)
* [x] mmap-based weight access (Page faults)
* [x] Vision cache disk storage (Prevents re-inference of identical images)

---

## Build & Run

### Prerequisites

* **Rust** ≥ 1.75 (nightly)
* **Node.js** ≥ 18 (Frontend build)
* **Tauri CLI** v2
* GPU backend (choose one, selected by Cargo feature):

| Feature | Backend | Requirement |
| --- | --- | --- |
| `cuda` (default) | NVIDIA CUDA | CUDA Toolkit ≥ 12.0 |
| `rocm` | AMD ROCm / HIP | ROCm SDK (via `candle-rocm-vulkan` fork) |
| `vulkan` | Vulkan | Vulkan SDK |
| `metal` | Apple Metal | macOS |

### Build

```bash
# Frontend build
cd commerce && npm install && npm run build

# Rust Backend build (CUDA default)
cd src-tauri
cargo build --release
# Other backends
cargo build --release --no-default-features --features rocm     # or vulkan / metal
```

### Run

```bash
cargo tauri dev      # Development mode
or
.\dev_cargo.bat      # .\dev_cargo_rocm.bat / .\dev_cargo_vulkan.bat

cargo tauri build    # Production build
or
.\build_cargo.bat    # .\build_cargo_rocm.bat / .\build_cargo_vulkan.bat
```

### Test

```bash
# Windows
.\test_cargo.bat     # .\test_cargo_rocm.bat / .\test_cargo_vulkan.bat

# Linux (vulkan | cuda | rocm)
./test_linux.sh vulkan

# Frontend
npm test
```

### Model Download

Base models are downloaded from the Settings tab upon first launch.
For manual downloads, place them in the [Model File Paths](#model-file-paths) directory specified above.

AlphaEdge Qwen3.5-4B language models are **not** part of the initial download. They are fetched automatically in the background the first time a non-Latin document language needs them (about 7.9GB per language), then converted to `runtime-q4km/model-q4km.gguf`. To pre-install manually, download `alphaedge-ai/Qwen3.5-4B-{code}-16384` into `{AppData}/logis-center/models/Qwen3.5-4B-{code}-16384/`; the GGUF conversion runs on first use.

---

## License

Apache License 2.0

### Reference Model Licenses

| Project | Component | License |
| --- | --- | --- |
| [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | Multimodal | Apache-2.0 |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Language Model | Apache-2.0 |
| [alphaedge-ai/Qwen3.5-4B-{code}-16384](https://huggingface.co/alphaedge-ai) | Language-specialized Model (Qwen3.5-4B based) | See each model card (base model: Apache-2.0) |
| [ibm-granite/granite-4.0-h-350m](https://huggingface.co/ibm-granite/granite-4.0-h-350m) | Language Model | Apache-2.0 |
| [ibm-granite/granite-embedding-97m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2) | Embedding | Apache-2.0 |
| [google/siglip2-so400m-patch16-naflex](https://huggingface.co/google/siglip2-so400m-patch16-naflex) | Vision Encoder | Apache-2.0 |
| [stanfordnlp](https://huggingface.co/stanfordnlp) | NLP Model | Apache-2.0 |
| [jhqxxx/aha](https://github.com/jhqxxx/aha) | Qwen Inference Engine | Apache-2.0 |
| [ericlbuehler/mistral.rs](https://github.com/ericlbuehler/mistral.rs) | Granite Inference Engine | MIT |

### Open Source Libraries

#### Rust (`commerce/src-tauri`)

| Library | License |
| --- | --- |
| ahash | MIT OR Apache-2.0 |
| any_ascii | ISC |
| anyhow | MIT OR Apache-2.0 |
| arrow-array / arrow-schema | Apache-2.0 |
| base64 | MIT OR Apache-2.0 |
| boa_engine (local build) | MIT OR Unlicense |
| calamine | MIT |
| candle-core / candle-nn / candle-transformers / candle-flash-attn | MIT OR Apache-2.0 |
| candle-rocm-vulkan (Qntology fork, `rocm` feature) | MIT OR Apache-2.0 |
| chrono | MIT OR Apache-2.0 |
| chromiumoxide | MIT OR Apache-2.0 |
| csv | Unlicense OR MIT |
| dirs | MIT OR Apache-2.0 |
| drag | MIT OR Apache-2.0 |
| ego-tree | ISC OR MIT |
| encoding_rs | MIT OR Apache-2.0 |
| ethers-core / ethers-signers | MIT OR Apache-2.0 |
| fantoccini | MIT OR Apache-2.0 |
| flate2 | MIT OR Apache-2.0 |
| futures | MIT OR Apache-2.0 |
| half | MIT OR Apache-2.0 |
| image | MIT OR Apache-2.0 |
| io-uring (Linux) | MIT OR Apache-2.0 |
| lancedb | Apache-2.0 |
| libc | MIT OR Apache-2.0 |
| memmap2 | MIT OR Apache-2.0 |
| metal (macOS) | MIT OR Apache-2.0 |
| mimalloc | MIT |
| minijinja | Apache-2.0 |
| ndarray | MIT OR Apache-2.0 |
| num | MIT OR Apache-2.0 |
| nvml-wrapper | MIT OR Apache-2.0 |
| once_cell | MIT OR Apache-2.0 |
| onnxruntime (local build) | MIT OR Apache-2.0 |
| pdf-extract | MIT |
| quick-xml | MIT |
| rand / rand_chacha | MIT OR Apache-2.0 |
| rayon | MIT OR Apache-2.0 |
| rcgen | MIT OR Apache-2.0 |
| regex | MIT OR Apache-2.0 |
| reqwest | MIT OR Apache-2.0 |
| safetensors | Apache-2.0 |
| scraper | ISC |
| serde / serde_json | MIT OR Apache-2.0 |
| sysinfo | MIT |
| tauri / tauri-build | MIT OR Apache-2.0 |
| tauri-plugin-opener / -dialog / -fs / -shell | MIT OR Apache-2.0 |
| tokenizers | Apache-2.0 |
| tokio | MIT |
| url | MIT OR Apache-2.0 |
| uuid | MIT OR Apache-2.0 |
| webrtc | MIT OR Apache-2.0 |
| whatlang | MIT |
| which | MIT |
| windows / windows-sys | MIT OR Apache-2.0 |
| zip | MIT |

#### JavaScript / TypeScript (`commerce`, `commerce/mobile`, `apac`, `proxy`)

| Library | License |
| --- | --- |
| @tauri-apps/api / @tauri-apps/cli / @tauri-apps/plugin-* | MIT OR Apache-2.0 |
| vite | MIT |
| vitest | MIT |
| happy-dom | MIT |
| typescript | Apache-2.0 |
| wrangler | MIT OR Apache-2.0 |
| ethers | MIT |
| linkedom | ISC |
| pako | MIT AND Zlib |

### GPU Notices

#### CUDA Toolkit

This application utilizes the NVIDIA CUDA Toolkit.
Portions of this software are copyrighted by NVIDIA Corporation.
[NVIDIA CUDA Toolkit EULA](https://docs.nvidia.com/cuda/eula/index.html)

#### AMD ROCm

This application utilizes components from the AMD ROCm Platform.
Portions of this software are copyrighted by Advanced Micro Devices, Inc.
Licensed under the MIT License and/or Apache License 2.0.
[ROCm License](https://rocm.docs.amd.com/en/latest/about/license.html)

#### Microsoft DirectStorage

This application uses Microsoft DirectStorage 1.3.0 on Windows for fast KV cache block I/O.
See `commerce/src-tauri/dstorage.license.txt` for license terms.

---

## Contributing

Issues and Pull Requests are always welcome.
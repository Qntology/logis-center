# Commerce · Trade & Logistics · Website Analytics

> **Local-first AI Operations Platform** — An open-source operational tool where a browser extension and desktop client work seamlessly together to structure, vectorize, and search e-commerce data, trade & logistics documents, and website behavior logs **completely offline**.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [AI Model Architecture](#ai-model-architecture)
- [Memory Management (RAM / SSD Offloading)](#memory-management-ram--ssd-offloading)
- [Build & Run](#build--run)
- [License](#license)

---

## Overview

| Domain | Description |
|--------|------|
| **Commerce** | Automatically structures order, goods, tracking, coupon, and review pages of shopping malls (Cafe24, MakeShop, Shopify, etc.) and indexes them into a local vector DB. |
| **Trade & Logistics** | Extracts 55 types of trade forms (B/L, C/I, P/L, L/C, etc.) using SigLIP2 Vision + Qwen3.5 2B and builds a reference graph between documents. |
| **Website Analytics** | Collects user click, hover, and input events to summarize behavioral flows, preferences, and conversion funnels into natural language reports. |

All inference, embedding, and storage are performed **locally on the client**, with no external API calls.

---

## Key Features

### Commerce Pipeline
- **Page Classification** — Automatically determines 6 types (`order / goods / tracking / review / coupon / event`) using vector cosine + structural signals (repeated rows, forms, headings).
- **List Extraction** — Boa engine DOM traversal → LLM field matching → mutually exclusive assignment (1:1) for column-value mapping.
- **Detail Extraction** — Structural label-value pairs + field-by-field extraction using Qwen3 0.6B.
- **Chunk Indexing** — Natural language segmentation → attribute tagging (PLINKO) → 384-dimensional tri-synthetic vectors → LanceDB `item_chunks`.
- **Transliteration Aliases** — Additionally stores bidirectional transliteration vectors for multilingual cross-lingual search.

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
├── commerce/
│   └── src-tauri/
│       ├── Cargo.toml                  # Rust dependencies & build settings
│       └── src/
│           ├── lib.rs                  # Tauri entry, IPC command registration
│           ├── scheduler.rs            # Background task scheduler (main pipeline)
│           ├── logic.rs                # Business logic (relay, state, trade rules)
│           ├── store.rs                # LanceDB vector store (envelope schema)
│           ├── automation.rs           # Chromium browser automation
│           ├── analytic.rs             # Website analytics tasks
│           ├── stanza.rs               # Stanza NLP (tokenization/POS/lemma/syntax)
│           ├── models/
│           │   ├── mod.rs
│           │   ├── common/             # Common layer (GateUpDownMLP, Attention, etc.)
│           │   ├── qwen/               # Qwen VL 0.6B (Vision + Language)
│           │   │   ├── config.rs       # Model configuration parsing
│           │   │   ├── generate.rs     # Inference loop, KV cache management
│           │   │   ├── model.rs        # Model structure definition
│           │   │   ├── processor.rs    # Image preprocessing (resize/patch)
│           │   │   ├── quantized_model.rs  # GGUF quantized model + KV offloading
│           │   │   └── rope.rs         # RoPE positional encoding
│           │   ├── qwen3/              # Qwen3 0.6B (Text only)
│           │   │   ├── config.rs
│           │   │   ├── generate.rs
│           │   │   └── model.rs
│           │   ├── qwen3_5/            # Qwen3.5 2B (Transliteration/Precision extraction)
│           │   │   ├── config.rs
│           │   │   ├── generate.rs
│           │   │   └── model.rs
│           │   ├── qwen3vl/            # Qwen3 VL (Next-gen Vision)
│           │   │   ├── config.rs
│           │   │   ├── generate.rs
│           │   │   ├── model.rs
│           │   │   └── processor.rs
│           │   ├── siglip2/            # SigLIP2 Vision Encoder
│           │   │   ├── mod.rs          # Model load (Text/Vision separated)
│           │   │   ├── vision.rs       # Vision Transformer
│           │   │   ├── text.rs         # Text encoder
│           │   │   ├── preprocessor.rs # NaFlex preprocessing
│           │   │   ├── vision_encoder.rs # Heatmap/Classification pipeline
│           │   │   ├── vision_crop.rs  # NMS crop + Tile splitting
│           │   │   ├── legibility.rs   # Legibility map (blur/margin detection)
│           │   │   ├── value_grounding.rs # Value grounding validation
│           │   │   └── tokenizer.rs    # SigLIP2 tokenizer
│           │   ├── embedding/          # granite-embedding-97m (384d)
│           │   ├── vision_cache/       # ViT output disk cache
│           │   └── granite/            # Granite MoE model
│           ├── utils/
│           │   ├── parsing.rs          # HTML→PUG conversion, schema management
│           │   ├── bias_schema.rs      # bias.json access layer
│           │   ├── bias.json           # Multilingual field anchor/bias dictionary
│           │   ├── ai_utils.rs         # Cosine, exclusive assignment, format gating
│           │   ├── json_parse.rs       # LLM JSON parsing/recovery
│           │   ├── nl_convert.rs       # JSON→Natural language, chunk segmentation
│           │   ├── img_utils.rs        # Image resize/patch
│           │   ├── json_utils.rs       # JSON merge
│           │   ├── pug_utils.rs        # PUG grid parser
│           │   ├── hash.rs             # ID hashing, CRC32
│           │   ├── canonical.rs        # Type normalization rules
│           │   ├── time_guide.rs       # Time intent parsing
│           │   └── lang_utils.rs       # Language detection
│           ├── parsers/
│           │   └── mod.rs              # PDF/Excel/Word/HWP text extraction
│           ├── chat_template.rs        # ChatML template
│           ├── tokenizer.rs            # GGUF tokenizer wrapper
│           ├── openai_types.rs         # OpenAI compatible type definitions
│           └── js_templates.rs         # Boa engine JS templates
├── crates/
│   ├── boa-engine/                     # Boa JS engine (local build)
│   ├── onnxruntime/                    # ONNX Runtime (local build)
│   └── direct-storage/                 # DirectStorage (Windows)
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
| **Qwen3.5 2B** | ~2.1GB (Q8) | Precision extraction, transliteration generation, document classification tie-breaks | As needed |
| **granite-embedding-97m** | ~380MB | 384-dimensional multilingual embeddings (vector search) | Indexing/Search |
| **SigLIP2** | ~2.2GB | Vision patch × text anchor cosine (Heatmap) | Trade documents |
| **Stanza** | ~50MB/lang | Tokenization, POS, Lemma, Dependency syntax | Transliteration preprocessing |

### Inference Pipeline Flow


```

[Task Queue] → [Scheduler]
├── Commerce: PUG conversion → Page classification → Field extraction → Chunk indexing
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
* **CUDA Toolkit** ≥ 12.0 (Optional, for GPU acceleration)
* **Node.js** ≥ 18 (Frontend build)
* **Tauri CLI** v2

### Build

```bash
# Frontend build
cd commerce && npm install && npm run build

# Rust Backend build
cd src-tauri
cargo build --release

```

### Run

```bash
cargo tauri dev      # Development mode
cargo tauri build    # Production build

```

### Model Download

Model downloading is triggered from the Settings tab upon first launch.
For manual downloads, place them in the [Model File Paths](https://www.google.com/search?q=%23model-file-paths) directory specified above.

---

## License

Apache License 2.0

### Reference Model Licenses

| Project | Component | License |
| --- | --- | --- |
| [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | Multimodal | Apache-2.0 |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Language Model | Apache-2.0 |
| [ibm-granite/granite-4.0-h-350m](https://huggingface.co/ibm-granite/granite-4.0-h-350m) | Language Model | Apache-2.0 |
| [ibm-granite/granite-embedding-97m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2) | Embedding | Apache-2.0 |
| [google/siglip2-so400m-patch16-naflex](https://huggingface.co/google/siglip2-so400m-patch16-naflex) | Vision Encoder | Apache-2.0 |
| [stanfordnlp](https://huggingface.co/stanfordnlp) | NLP Model | Apache-2.0 |
| [jhqxxx/aha](https://github.com/jhqxxx/aha) | Qwen Inference Engine | Apache-2.0 |
| [ericlbuehler/mistral.rs](https://github.com/ericlbuehler/mistral.rs) | Granite Inference Engine | MIT |

### Open Source Libraries

| Library | License |
| --- | --- |
| ahash | MIT OR Apache-2.0 |
| anyhow | MIT OR Apache-2.0 |
| arrow-array / arrow-schema | Apache-2.0 |
| base64 | MIT OR Apache-2.0 |
| candle-core / candle-nn / candle-transformers | MIT OR Apache-2.0 |
| chrono | MIT OR Apache-2.0 |
| chromiumoxide | MIT OR Apache-2.0 |
| ego-tree | ISC OR MIT |
| encoding_rs | MIT OR Apache-2.0 |
| ethers-core / ethers-signers | MIT OR Apache-2.0 |
| fantoccini | MIT OR Apache-2.0 |
| flate2 | MIT OR Apache-2.0 |
| futures | MIT OR Apache-2.0 |
| half | MIT OR Apache-2.0 |
| image | MIT OR Apache-2.0 |
| lancedb | Apache-2.0 |
| libc | MIT OR Apache-2.0 |
| memmap2 | MIT OR Apache-2.0 |
| mimalloc | MIT |
| minijinja | Apache-2.0 |
| ndarray | MIT OR Apache-2.0 |
| num | MIT OR Apache-2.0 |
| nvml-wrapper | MIT OR Apache-2.0 |
| once_cell | MIT OR Apache-2.0 |
| onnxruntime | MIT OR Apache-2.0 |
| rand / rand_chacha | MIT OR Apache-2.0 |
| rayon | MIT OR Apache-2.0 |
| rcgen | MIT OR Apache-2.0 |
| regex | MIT OR Apache-2.0 |
| reqwest | MIT OR Apache-2.0 |
| safetensors | Apache-2.0 |
| scraper | ISC |
| serde / serde_json | MIT OR Apache-2.0 |
| pdf-extract | MIT |
| sysinfo | MIT |
| tauri | MIT OR Apache-2.0 |
| tokenizers | Apache-2.0 |
| tokio | MIT |
| url | MIT OR Apache-2.0 |
| uuid | MIT OR Apache-2.0 |
| webrtc | MIT OR Apache-2.0 |
| whatlang | MIT |
| which | MIT |
| windows / windows-sys | MIT OR Apache-2.0 |

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

---

## Contributing

Issues and Pull Requests are always welcome.
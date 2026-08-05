# 🎰 Logis-Center Extraction Pipeline: The Plinko Board

> **Plinko Metaphor**: A ball (data value) drops from the top of the board, is pulled or pushed by each peg (bias/prejudice magnet pairs), and finally lands in a bottom basket (category/field) – a pipeline simulation.

---

## Part 1: Document Entry and Page Classification (Layer 1 ~ Layer 3)

```text
=============================================================================
                       [ 🌐 URL / HTML DROP ZONE ]
│                 testshop.com/manage/goods?type=list
│
▼
=============================================================================
[ LAYER 1 ]          🎱 BALL DROP (HTML → PUG Line Conversion)
│
│   o   o   o   o   o   o   o   o   o   o   o   o
│    \   \   \   \  |  /   /   /   /   /   /   /
│     v   v   v   v  v  v   v   v   v   v   v   v
│
│   HTML → Clean PUG (4774 tokens) → FullContent Mode Upgrade 
│
=============================================================================
[ LAYER 2 ]          🌪️ GLOBAL NOISE FILTER (The Blackholes)
│
│   o    x    o    x    o    o    x    o    o    x
│  / \       / \       / \  / \       / \       / \
│ (Nav)    (Foot)    (Dup) (Dup)    (Ad)     (Menu)
│   x        x        x    x        x         x
│
│   [ 🚫 Noise dropped into blackholes (x) ]
│   🚫 'All' (6 occurrences cross noise) 
│   🚫 'Display' (15 occurrences noise) 
│   🚫 'admin', 'Sample Shop', 'Edit', 'Unlimited' (13 occurrences repeated single column)
│
│   [ 🛡️ Protected core data (o) ]
│   🛡️ 'Test Product Small Cherry...' (protected by dispersion pattern within the same structure)
│
=============================================================================
[ LAYER 3 ]          🎯 PAGE-TYPE CLASSIFICATION PLINKO
│                    (bias.json: ko.{type}.layout_*)
│
│        o              o              o
│       / \            / \            / \
│     (🧲) (🧲)     (🧲) (🧲)     (🧲) (🧲)     ← Category Anchors (N-pole magnets)
│     / \  / \      / \  / \      / \  / \
│    /   \/   \    /   \/   \    /   \/   \
│
│   [order]  [goods]  [tracking]  [review]  [coupon]  [event]
│              |★|
│              ▼
│     (Selected: goods basket / Score: 2.7249)
│
=============================================================================
```

---

## Part 2: Structure Determination and Core Extraction Engine (Layer 4 ~ Layer 7)

```text
=============================================================================
[ LAYER 4 ]          ⚙️ STRUCTURE ROUTING (is_detail determination)
│
│        ┌──────────────────┐     ┌──────────────────┐
│        │  🧲 LIST peg     │     │  🚫 FORM peg     │
│        │  Score: 1.4859   │     │  Score: 0.0000   │
│        │  rows:28 ✅      │     │                  │
│        └────────┬─────────┘     └──────────────────┘
│                 ▼
│         is_detail = false → List Path (enters list extraction path)
│
=============================================================================
[ LAYER 5 ]          🔍 SELECTOR LOCK (DOM Chunking)
│
│   Qwen3 Titles → ["Test Product 3", "Test Product - Flower..."]
│   Boa Engine CSS Selector → table#sodr_list.tablef tr.list0, tr.list1...
│   matchCount: 13 (each row separated into 13 independent item balls)
│
=============================================================================
[ LAYER 6 ]          🎰 FIELD EXTRACTION PLINKO (The Core Engine)
│
│   [ Balls: "13", "0", "Test Product 3", "Display", "admin", "24-12-26" ]
│    │  │       │            │          │         │
│    ▼  ▼       ▼            ▼          ▼         ▼
│   ┌─────────────────────────────────────────────────────────┐
│   │  1️⃣ FORMAT GATE (Physical entry filter)                 │
│   │  [Date]    [Numeric]   [Text]    [Enum]    [Link]       │
│   │  "24-12"✅  "0"✅      "13"⛔    "Display"✅ (none)⛔   │
│   └─────────────────────────────────────────────────────────┘
│    │  │       │            │          │
│    ▼  ▼       ▼            ▼          ▼
│   ┌─────────────────────────────────────────────────────────┐
│   │  2️⃣ DOUBLE CENTERING + EXCLUSIVE ASSIGN                 │
│   │                                                         │
│   │    (🧲 N-Pole: Bias / 🚫 S-Pole: Prejudice)             │
│   │                                                         │
│   │      o("0")         o("Display")    o("admin")          │
│   │     / \             / \             / \                 │
│   │  (🧲) (🚫)       (🧲) (🚫)       (🧲) (🚫)             │
│   │  /   \  \        /   \  \        /   \  \              │
│   │ [qty][price]   [status][curr]  [status][title]         │
│   │  ★               ★      ★        ★                      │
│   │                                                         │
│   │  * Exclusive assignment: the field with the largest margin preempts the ball 1:1 │
│   └─────────────────────────────────────────────────────────┘
│    │       │            │
│    ▼       ▼            ▼
│   ┌─────────────────────────────────────────────────────────┐
│   │  3️⃣ LLM CATCH + POST-FORMAT REJECT (Safety Net)         │
│   │                                                         │
│   │  "13"  → LLM → Title? ⛔ REJECT (numbers only)          │
│   │  "0"   → LLM → Title? ⛔ REJECT (numbers only)          │
│   │  "Test Product 3" → LLM → Title? ✅ PASS                │
│   └─────────────────────────────────────────────────────────┘
│    │
│    ▼
=============================================================================
[ LAYER 7 ]                      🛒 OUTPUT & DB SAVE
│
│   { 
│     "status": "progress", 
│     "title": "Test Product 3",
│     "quantity": "0", 
│     "currency": "0",
│     "general_insight": "The sales data shows a fluctuating trend...",
│     "traffic_insight": "There are multiple products available..." 
│   }
│
│   → DB (sales / items / tracking / users) tables Upsert
│   → Metrics Engine statistics update logic execution
│
=============================================================================
```

---

## Part 3: Magnet Placement, Physical Laws, and Real Bounce Cases

### 🧲 The Magnet Array: `bias.json` Magnet Layout

```text
=============================================================================
[ FIELD MAGNET ARRAY (ko.goods.*) ]
=============================================================================
🎱 Ball (value to extract)
│
├──────────────────────────────────────────────────────────────────┐
│  Peg 1: id,link (Identifier)                                     │
│  │ 🧲 bias(N-pole):   "Product Identifier, Product Number, PROD-001..." │
│  │ 🚫 prej(S-pole):   "Sales Status, Product Name, Registration Date, Quantity..." │
│  │ ⚙️ Physical law:  Must exist in the token pool within href to pass │
├──────────────────────────────────────────────────────────────────┤
│  Peg 2: title (Text)                                             │
│  │ 🧲 bias(N-pole):   "Product Name, Product Title, Premium Wireless Headphones..." │
│  │ 🚫 prej(S-pole):   "Identifier, Link, Registration Date, Quantity..." │
│  │ ⚙️ Physical law:  String of at least 2 characters required (bounced if only numbers) │
├──────────────────────────────────────────────────────────────────┤
│  Peg 3: registration_date (Date)                                 │
│  │ 🧲 bias(N-pole):   "Product Registration Date, 2026-03-15T12:00:00..." │
│  │ 🚫 prej(S-pole):   "Identifier, Product Name, Quantity..."   │
│  │ ⚙️ Physical law:  a-b-c date pattern required (e.g., "24-12-26") │
├──────────────────────────────────────────────────────────────────┤
│  Peg 4: quantity (Numeric)                                       │
│  │ 🧲 bias(N-pole):   "100, 50, 0, 999"                         │
│  │ 🚫 prej(S-pole):   "Identifier, Product Name, Registration Date..." │
│  │ ⚙️ Physical law:  Must contain a number                      │
├──────────────────────────────────────────────────────────────────┤
│  Peg 5: general_insight / traffic_insight (Synthesis)            │
│  │ 🧠 SYNTHESIS FIELD — No magnet!                              │
│  │ Cannot be reduced to a single line → LLM reads entire context and synthesizes sentences │
└──────────────────────────────────────────────────────────────────┘
```

### ⚖️ The 3 Laws of Plinko Physics

```text
=============================================================================
[ THE 3 LAWS OF PHYSICS ]
=============================================================================
│
│  Law 1: FORMAT GATE (Physical entry filter at peg)
│  ─────────────────────────────────────────────
│  Before measuring similarity (Vector), the value's appearance is validated.
│  • Date peg: "24-09-11" (✅ passes) / "615600" (❌ bounced)
│  • Text peg: "Test Product" (✅ passes) / "13" (❌ bounced)
│
│  Law 2: DOUBLE CENTERING (Gravity Correction)
│  ─────────────────────────────────────────────
│  Resolves the phenomenon where all vector values cluster in the 0.50-0.74 range.
│  • "Is this line scoring high on all fields?" (Subtract line mean)
│  • "Is this field scoring high on all lines?" (Subtract field mean)
│  • Use only the residual (Contrast) to determine true relative advantage.
│
│  Law 3: EXCLUSIVE ASSIGN (One ball – one basket)
│  ─────────────────────────────────────────────
│  Proceed with 1:1 preemption in order of largest Margin.
│  A ball that has entered one basket can never be taken by another basket.
│
=============================================================================
```

### ⚡ Real Bounces: Bounce Phenomena Observed in Actual Logs

```text
=============================================================================
[ LLM HALLUCINATION REJECT LOOPS ]
=============================================================================
[ CASE 1: title field (Item 1/13) ]
🎱 Ball drops → approaches title magnet
 │
 ├── 1st attempt: LLM returns "13"
 │   → 🚫 FORMAT REJECT triggered: Text type contains only numbers
 │   → Magnet strongly bounces the ball away
 │
 ├── 2nd attempt: LLM returns "0"
 │   → 🚫 FORMAT REJECT triggered: again numbers only
 │   → Added "13", "0" to ignore_list as blacklist and retried
 │
 └── 3rd attempt: LLM returns "Test Product 3"
     → ✅ FORMAT PASS: Contains letters, meets the 2+ character condition
     → Landed in basket 🧺

[ CASE 2: registration_date field (Item 3/13) ]
🎱 Ball drops → approaches Date magnet
 │
 ├── 1st attempt: LLM returns "1726045859" (raw timestamp value)
 │   → 🚫 FORMAT REJECT triggered: Not in YY-MM-DD (a-b-c) pattern
 │
 └── 2nd attempt: LLM returns "24-09-11"
     → ✅ FORMAT PASS: Date pattern satisfied
     → Landed in basket 🧺
=============================================================================
```

---

### 📝 Final One-Line Trajectory Summary (for Item 1/13)

> **"PUG Line 25 → [LAYER 3] confirmed as goods → [LAYER 4] confirmed as List → [LAYER 6] passed Format/Centering/LLM Reject gates → [LAYER 7] 4 field values + 2 synthesized sentences landed in baskets"**


---

# License

## Reference License

This project integrates and extends the following open-source works:

| Project | Component | License |
|-----------|---------|---------|
| [Qwen/Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) | Multimodal | Apache-2.0 |
| [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Language Model | Apache-2.0 |
| [ibm-granite/granite-4.0-h-350m](https://huggingface.co/ibm-granite/granite-4.0-h-350m) | Language Model | Apache-2.0 |
| [ibm-granite/granite-embedding-97m-multilingual-r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2) | Embedding Model | Apache-2.0 |
| [stanfordnlp](https://huggingface.co/stanfordnlp) | NLP Model | Apache-2.0 |
| [jhqxxx/aha](https://github.com/jhqxxx/aha) | Qwen Inference Engine | Apache-2.0 |
| [ericlbuehler/mistral.rs](https://github.com/ericlbuehler/mistral.rs) | Granite Inference Engine | MIT |

---

## Open Source Licenses

This project uses the following open-source libraries:

| Library | License |
|---------|---------|
| ahash | MIT OR Apache-2.0 |
| anyhow | MIT OR Apache-2.0 |
| arrow-array | Apache-2.0 |
| arrow-schema | Apache-2.0 |
| base64 | MIT OR Apache-2.0 |
| candle-core | MIT OR Apache-2.0 |
| candle-flash-attn | MIT OR Apache-2.0 |
| candle-nn | MIT OR Apache-2.0 |
| candle-transformers | MIT OR Apache-2.0 |
| chrono | MIT OR Apache-2.0 |
| chromiumoxide | MIT OR Apache-2.0 |
| direct-storage | MIT OR Apache-2.0 |
| dirs | MIT OR Apache-2.0 |
| drag | MIT OR Apache-2.0 |
| ego-tree | ISC OR MIT |
| encoding_rs | MIT OR Apache-2.0 |
| ethers-core | MIT OR Apache-2.0 |
| ethers-signers | MIT OR Apache-2.0 |
| fantoccini | MIT OR Apache-2.0 |
| flate2 | MIT OR Apache-2.0 |
| futures | MIT OR Apache-2.0 |
| half | MIT OR Apache-2.0 |
| image | MIT OR Apache-2.0 |
| io-uring | MIT OR Apache-2.0 |
| lancedb | Apache-2.0 |
| libc | MIT OR Apache-2.0 |
| memmap2 | MIT OR Apache-2.0 |
| metal | MIT OR Apache-2.0 |
| mimalloc | MIT |
| minijinja | Apache-2.0 |
| ndarray | MIT OR Apache-2.0 |
| num | MIT OR Apache-2.0 |
| nvml-wrapper | MIT OR Apache-2.0 |
| once_cell | MIT OR Apache-2.0 |
| onnxruntime | MIT OR Apache-2.0 |
| rand | MIT OR Apache-2.0 |
| rand_chacha | MIT OR Apache-2.0 |
| rayon | MIT OR Apache-2.0 |
| rcgen | MIT OR Apache-2.0 |
| regex | MIT OR Apache-2.0 |
| reqwest | MIT OR Apache-2.0 |
| safetensors | Apache-2.0 |
| scraper | ISC |
| serde | MIT OR Apache-2.0 |
| serde_json | MIT OR Apache-2.0 |
| sysinfo | MIT |
| tauri | MIT OR Apache-2.0 |
| tauri-build | MIT OR Apache-2.0 |
| tauri-plugin-dialog | MIT OR Apache-2.0 |
| tauri-plugin-fs | MIT OR Apache-2.0 |
| tauri-plugin-opener | MIT OR Apache-2.0 |
| tauri-plugin-shell | MIT OR Apache-2.0 |
| tokenizers | Apache-2.0 |
| tokio | MIT |
| url | MIT OR Apache-2.0 |
| uuid | MIT OR Apache-2.0 |
| webrtc | MIT OR Apache-2.0 |
| whatlang | MIT |
| which | MIT |
| windows | MIT OR Apache-2.0 |
| windows-sys | MIT OR Apache-2.0 |

---

## GPU Notices

### CUDA Toolkit Notice

This application utilizes the NVIDIA CUDA Toolkit.
Portions of this software are copyrighted by NVIDIA Corporation.
All rights reserved.

NVIDIA CUDA Toolkit End User License Agreement (EULA):
<https://docs.nvidia.com/cuda/eula/index.html>

### AMD ROCm Notice

This application utilizes components from the AMD ROCm Platform.
Portions of this software are copyrighted by Advanced Micro Devices, Inc.
All rights reserved.

Licensed under the MIT License and/or Apache License 2.0.
For more information and source code, please visit:
<https://rocm.docs.amd.com/en/latest/about/license.html>

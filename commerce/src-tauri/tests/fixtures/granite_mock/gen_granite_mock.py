"""Mock granite-embedding-97m-multilingual-r2 for the [EMBED-BATCHED] verification (MOCK weights, not real inference).

Usage (from any directory; the script never imports from its own folder):

    python -I gen_granite_mock.py <models_dir>                 # writes <models_dir>/granite-embedding-97m-multilingual-r2/
    python -I gen_granite_mock.py <models_dir> --train-tokenizer   # also re-trains tokenizer.json in this folder

Then point the tests at it:  LOGIS_MODELS_DIR=<models_dir> LOGIS_TEST_MODELS=1 LOGIS_TEST_EMBED_DEVICE=cpu ...

What it is:
  * the architecture models/embedding.rs loads (FloatModel): hidden 384, 12 layers, 12 heads (head_dim 32),
    intermediate 1536 (Wi = 384 -> 2*1536 gate|up, Wo = 1536 -> 384), RMSNorm, layers.0 without attn_norm,
    the same tensor names. Values the code does not read from the checkpoint (RoPE theta 150000, 512-token
    chunks) are the code's own constants.
  * vocab reduced to 32000 rows (the vocab only changes the embedding lookup, not the per-token compute).
  * seeded weights from an exact integer -> float mapping (no libm calls), so every platform writes the same
    bytes; the script prints the SHA-256 and checks it against MOCK_SHA256 below.
  * tokenizer.json (committed next to this script) is a byte-level BPE trained on the app's own bias
    dictionaries (src/utils/bias.json, full_bias.json) with a small merge budget, so Korean / English /
    Japanese phrases get realistic subword lengths (short bank phrases 2-64 tokens, layout texts 65-512,
    joined texts > 512 that exercise the 512-token chunking). [CLS] $A [SEP] post-processing like the real
    tokenizer's encode(text, add_special_tokens=true).

Real-model values that could not be fetched here (huggingface.co is blocked in the Linux VM) are taken from
what the code expects; the macOS workflow downloads the real config/weights separately for decision-level checks.
"""

import hashlib
import json
import math
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC_TAURI = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
MODEL_NAME = "granite-embedding-97m-multilingual-r2"

CONFIG = {
    "architectures": ["ModernBertModel"],
    "model_type": "modernbert",
    "_mock": "seeded random weights for verification only (tests/fixtures/granite_mock/gen_granite_mock.py)",
    "hidden_size": 384,
    "intermediate_size": 1536,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "norm_eps": 1e-5,
    "vocab_size": 32000,
    "pad_token_id": 0,
    "max_position_embeddings": 8192,
}
SEED = 20261013
TOKENIZER_VOCAB = 16000
# SHA-256 of model.safetensors written by this script (checked after writing; None = print only).
MOCK_SHA256 = "3c920073cfc6eed2dc25dfa3f811d93fdf7c3ad3df972733b50e8effeeac8b65"


def det_uniform(rng, n, half_width):
    """n values uniform in [-half_width, half_width), from 24-bit integers (exact in float64, platform-independent)."""
    raw = rng.bit_generator.random_raw(n)
    u = (raw >> np.uint64(40)).astype(np.float64) / float(1 << 24)  # exact
    return ((2.0 * u - 1.0) * half_width).astype(np.float32)


def tensor_specs(cfg):
    h, i, n = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    specs = [("embeddings.tok_embeddings.weight", (cfg["vocab_size"], h), "emb"),
             ("embeddings.norm.weight", (h,), "norm")]
    for l in range(n):
        p = f"layers.{l}"
        if l > 0:  # Granite (ModernBERT) layers.0 has no attn_norm (embedding.rs CRITICAL FIX)
            specs.append((f"{p}.attn_norm.weight", (h,), "norm"))
        specs += [
            (f"{p}.attn.Wqkv.weight", (3 * h, h), "lin"),
            (f"{p}.attn.Wo.weight", (h, h), "lin"),
            (f"{p}.mlp_norm.weight", (h,), "norm"),
            (f"{p}.mlp.Wi.weight", (2 * i, h), "lin"),
            (f"{p}.mlp.Wo.weight", (h, i), "lin"),
        ]
    specs.append(("final_norm.weight", (h,), "norm"))
    return specs


def build_weights(cfg):
    rng = np.random.Generator(np.random.PCG64(SEED))
    out = {}
    for name, shape, kind in tensor_specs(cfg):
        n = int(np.prod(shape))
        if kind == "emb":
            w = det_uniform(rng, n, math.sqrt(3.0))  # std 1
        elif kind == "norm":
            w = (1.0 + det_uniform(rng, n, 0.2)).astype(np.float32)
        else:
            fan_in = shape[1]
            w = det_uniform(rng, n, math.sqrt(3.0 / fan_in))  # std 1/sqrt(fan_in)
        out[name] = w.reshape(shape)
    return out


def save_safetensors(tensors, path, metadata):
    """Byte-deterministic safetensors writer (safetensors' own writer orders the header through a hash map).
    Header keys are sorted, tensors are laid out in sorted-name order, little-endian F32."""
    header = {"__metadata__": dict(sorted(metadata.items()))}
    offset = 0
    blobs = []
    for name in sorted(tensors):
        a = np.ascontiguousarray(tensors[name], dtype="<f4")
        b = a.tobytes()
        header[name] = {"dtype": "F32", "shape": list(a.shape), "data_offsets": [offset, offset + len(b)]}
        offset += len(b)
        blobs.append(b)
    h = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as fh:
        fh.write(len(h).to_bytes(8, "little"))
        fh.write(h)
        for b in blobs:
            fh.write(b)


def corpus_lines():
    lines = []

    def walk(v):
        if isinstance(v, str):
            s = v.strip()
            if s:
                lines.append(s)
        elif isinstance(v, dict):
            for k in sorted(v):
                walk(k)
                walk(v[k])
        elif isinstance(v, list):
            for x in v:
                walk(x)

    for f in ("bias.json", "full_bias.json"):
        with open(os.path.join(SRC_TAURI, "src", "utils", f), encoding="utf-8") as fh:
            walk(json.load(fh))
    return lines


def train_tokenizer(path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()
    specials = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(
        vocab_size=TOKENIZER_VOCAB, min_frequency=2, special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    )
    tok.train_from_iterator(corpus_lines(), trainer=trainer)
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))],
    )
    assert tok.token_to_id("[PAD]") == CONFIG["pad_token_id"]
    tok.save(path)
    print(f"tokenizer.json: vocab {tok.get_vocab_size()} -> {path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    out_root = os.path.abspath(sys.argv[1])
    tok_src = os.path.join(HERE, "tokenizer.json")
    if "--train-tokenizer" in sys.argv or not os.path.exists(tok_src):
        train_tokenizer(tok_src)

    out = os.path.join(out_root, MODEL_NAME)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(CONFIG, fh, indent=2)
    shutil.copyfile(tok_src, os.path.join(out, "tokenizer.json"))
    weights = build_weights(CONFIG)
    wpath = os.path.join(out, "model.safetensors")
    save_safetensors(weights, wpath, {"format": "pt", "mock": "1", "seed": str(SEED)})
    with open(wpath, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    with open(os.path.join(out, "MOCK"), "w") as fh:
        fh.write(f"mock granite (seed {SEED}) sha256 {digest}\n")
    n_params = sum(w.size for w in weights.values())
    print(f"mock {MODEL_NAME}: {n_params/1e6:.1f}M params -> {out}")
    print(f"model.safetensors sha256 {digest}")
    if MOCK_SHA256 is not None and digest != MOCK_SHA256:
        print(f"ERROR: expected sha256 {MOCK_SHA256}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Tiny random Qwen4-Exp reference fixtures for the candle port (run with `python -P gen_tiny_ref.py <out_dir>`).

Writes two fixtures (each well under 2 MB):

  <out>/            variant A = the validated tiny config of qwen4-spec section 13/G, saved with
                    save_pretrained (checkpoint key format: per-expert experts.{e}.*, 4 n-gram shards,
                    int64 PLE buffers, layer_types remapped to qwen_sparse_attention).
  <out>/variant_b/  variant B = VLM-style checkpoint: config {"model_type":"qwen4_exp","text_config":{..}}
                    with "full_attention" layer types, keys prefixed model.language_model.*, runtime key
                    format (fused experts.gate_up_proj / down_proj, single ngram_embedding.weight),
                    sigmoid output gate, hc_count 3, 2 QSA layers, 2 PLE layers, eos list, Dk != Dv,
                    conv kernel 3, norm_topk_prob False, partial_rotary_factor absent (-> 1.0).

ref.safetensors in each folder holds:
  input_ids [1,T] i64, schedule [n] i64 (incremental chunk sizes), full_logits [1,T,V] f32 (one full
  forward, no cache), inc_logits [1,T,V] f32 (prefill + incremental forwards with the HF cache),
  hidden_{i} [1,T,hc*H] decoder outputs (full forward), mixer [1,T,H], ngram_ids_{i} [T,G] i64 for PLE
  layers, qsa_mask_{i} [T,T] u8 (1 = selected) for QSA layers.

Seeds are searched so that no QSA top-k boundary tie exists (score[k-1] == score[k]) in either the
full or the incremental pass: torch.topk tie order is implementation defined, so such positions
would not be a fair comparison.
"""

import json
import math
import os
import sys

import torch
from safetensors.torch import save_file

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpForCausalLM,
    Qwen4ExpTextQSAIndexer,
    apply_rotary_pos_emb,
)


def cfg_a():
    return dict(
        vocab_size=512, hidden_size=64, num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                         "mrope_section": [2, 1, 1], "mrope_interleaved": True},
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=48,
        norm_topk_prob=True,
        hc_count=2, hc_lowrank=16,
        ple_layer_ids=[2], ple_embed_dim=32, ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=2,
        ngram_vocab_size_base=97, make_ngram_vocab_size_divisible_by=128, seed=1234, split_ngram_parts=4,
        indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=8, indexer_compress_ratio=4,
        eos_token_id=2, pad_token_id=0, bos_token_id=1, rms_norm_eps=1e-6, max_position_embeddings=256,
        initializer_range=0.2, tie_word_embeddings=False, output_gate_type="silu",
    )


def cfg_b():
    return dict(
        vocab_size=256, hidden_size=48, num_hidden_layers=4,
        layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"],
        num_attention_heads=4, num_key_value_heads=1, head_dim=16,
        rope_parameters={"rope_type": "default", "rope_theta": 50000.0},
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=24,
        linear_conv_kernel_dim=3,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=24, shared_expert_intermediate_size=32,
        norm_topk_prob=False,
        hc_count=3, hc_lowrank=8,
        ple_layer_ids=[3, 1], ple_embed_dim=24, ple_conv_kernel_size=3, ngram_size=3, heads_per_ngram=3,
        ngram_vocab_size_base=50, make_ngram_vocab_size_divisible_by=64, seed=77, split_ngram_parts=1,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=16, indexer_budget=6, indexer_compress_ratio=3,
        eos_token_id=[5, 7], pad_token_id=0, bos_token_id=1, rms_norm_eps=1e-5, max_position_embeddings=128,
        initializer_range=0.2, tie_word_embeddings=False, output_gate_type="sigmoid",
    )


def build(kw, seed):
    cfg = Qwen4ExpTextConfig(**kw)
    torch.manual_seed(seed)
    try:
        model = Qwen4ExpForCausalLM._from_config(cfg, attn_implementation="eager")
    except Exception:
        cfg._attn_implementation = "eager"
        model = Qwen4ExpForCausalLM(cfg)
    model = model.float().eval()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("A_log") or name.endswith("dt_bias"):
                continue
            p.add_(torch.randn_like(p) * 0.1)
    return cfg, model


class Recorder:
    """Forward hooks: decoder outputs, mixer output, PLE n-gram ids, QSA masks, indexer inputs."""

    def __init__(self, model):
        self.data = {}
        self.indexer_inputs = {}
        self.handles = []
        m = model.model
        for i, layer in enumerate(m.layers):
            self.handles.append(layer.register_forward_hook(self._store(f"hidden_{i}")))
            if layer.ple is not None:
                emb = layer.ple.ple_embedding.ngram_embedding
                self.handles.append(emb.register_forward_hook(self._store_input(f"ngram_ids_{i}")))
            if hasattr(layer, "self_attn"):
                idx = layer.self_attn.indexer
                self.handles.append(idx.register_forward_hook(self._store_mask(f"qsa_mask_{i}")))
                self.handles.append(idx.register_forward_pre_hook(self._store_indexer_in(i), with_kwargs=True))
        self.handles.append(m.hyper_connection_mixer.register_forward_hook(self._store("mixer")))

    def _store(self, key):
        def hook(_m, _i, out):
            self.data[key] = out.detach().clone()
        return hook

    def _store_input(self, key):
        def hook(_m, inp, _out):
            self.data[key] = inp[0].detach().clone()[0]
        return hook

    def _store_mask(self, key):
        def hook(_m, _i, out):
            mask = out if out.dtype == torch.bool else (out == 0)
            self.data[key] = mask[0, 0].to(torch.uint8).clone()
        return hook

    def _store_indexer_in(self, layer_idx):
        def hook(_m, args, kwargs):
            hs = args[0] if args else kwargs["hidden_states"]
            pe = args[1] if len(args) > 1 else kwargs["position_embeddings"]
            self.indexer_inputs.setdefault(layer_idx, []).append((hs.detach().clone(), tuple(x.detach().clone() for x in pe)))
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def boundary_ties(indexer: Qwen4ExpTextQSAIndexer, calls, total_len):
    """Count queries whose top-k boundary is tied, replaying the indexer math from the captured inputs."""
    ties = 0
    raw_all = []
    q_all = []
    for hs, (full_cos, full_sin) in calls:
        t = hs.shape[1]
        qk = indexer.index_qk_proj(hs)
        q, tk = torch.split(qk, [indexer.index_n_heads * indexer.index_head_dim, indexer.index_head_dim], dim=-1)
        q = q.reshape(1, t, -1, indexer.index_head_dim)
        q = indexer.q_layernorm(q)
        q = apply_rotary_pos_emb(q, cos=full_cos[:, -t:], sin=full_sin[:, -t:], unsqueeze_dim=2)
        raw_all.append(tk.reshape(1, t, -1))
        q_all.append(q)
        cos_full, sin_full = full_cos, full_sin
    raw = torch.cat(raw_all, dim=1)[0]
    q = torch.cat(q_all, dim=1)[0]
    c = indexer.compress_ratio
    k = indexer.block_topk
    for p in range(total_len):
        nb = (p + 1) // c
        if nb <= k:
            continue
        groups = raw[: nb * c].view(nb, c, -1)
        pooled = indexer.k_layernorm(groups.float().mean(dim=1).to(raw.dtype))
        starts = torch.arange(nb) * c
        kb = apply_rotary_pos_emb(pooled.unsqueeze(1), cos=cos_full[0].index_select(0, starts),
                                  sin=sin_full[0].index_select(0, starts)).squeeze(1)
        s = torch.relu(q[p].float() @ kb.float().T).sum(0) / math.sqrt(indexer.index_head_dim)
        srt = torch.sort(s, descending=True).values
        if torch.isclose(srt[k - 1], srt[k], rtol=0, atol=1e-6):
            ties += 1
    return ties


def make_ids(vocab, n, eos_positions, eos_ids, gen):
    ids = torch.randint(3, vocab, (1, n), generator=gen)
    for pos, e in zip(eos_positions, eos_ids):
        ids[0, pos] = e
    return ids


def run_variant(kw, out_dir, n_tokens, eos_positions, eos_ids, schedule, vlm_layout):
    os.makedirs(out_dir, exist_ok=True)
    for seed in range(0, 50):
        cfg, model = build(kw, seed)
        gen = torch.Generator().manual_seed(1000 + seed)
        ids = make_ids(kw["vocab_size"], n_tokens, eos_positions, eos_ids, gen)

        rec = Recorder(model)
        with torch.no_grad():
            full = model(input_ids=ids, use_cache=False).logits
        full_data = dict(rec.data)
        full_calls = dict(rec.indexer_inputs)
        rec.close()

        rec2 = Recorder(model)
        inc = []
        with torch.no_grad():
            past = None
            pos = 0
            for chunk in schedule:
                out = model(input_ids=ids[:, pos:pos + chunk], past_key_values=past, use_cache=True)
                past = out.past_key_values
                inc.append(out.logits)
                pos += chunk
        assert pos == n_tokens
        inc = torch.cat(inc, dim=1)
        inc_calls = dict(rec2.indexer_inputs)
        rec2.close()

        ties = 0
        for li, calls in full_calls.items():
            ties += boundary_ties(model.model.layers[li].self_attn.indexer, calls, n_tokens)
        for li, calls in inc_calls.items():
            ties += boundary_ties(model.model.layers[li].self_attn.indexer, calls, n_tokens)
        d = (full - inc).abs().max().item()
        agree = (full.argmax(-1) == inc.argmax(-1)).float().mean().item()
        print(f"[{os.path.basename(out_dir) or out_dir}] seed {seed}: boundary ties {ties}, |full-inc| {d:.3e}, argmax agree {agree:.3f}, logit scale {full.abs().max().item():.3f}")
        if ties == 0:
            break
    else:
        raise SystemExit("no tie-free seed found")

    # Selected-token counts of the first QSA layer (for the report).
    for key in sorted(k for k in full_data if k.startswith("qsa_mask_")):
        counts = full_data[key].sum(-1).tolist()
        print(f"  {key} selected counts: {counts}")

    if vlm_layout:
        sd = {}
        for name, t in model.state_dict().items():
            new = name.replace("model.", "model.language_model.", 1) if name.startswith("model.") else name
            sd[new] = t.detach().contiguous().clone()
        save_file(sd, os.path.join(out_dir, "model.safetensors"))
        text = cfg.to_dict()
        text["layer_types"] = ["full_attention" if t == "qwen_sparse_attention" else t for t in text["layer_types"]]
        text["model_type"] = "qwen4_exp_text"
        for k in ("architectures", "transformers_version", "_attn_implementation_autoset"):
            text.pop(k, None)
        top = {"architectures": ["Qwen4ExpForConditionalGeneration"], "model_type": "qwen4_exp",
               "text_config": text, "tie_word_embeddings": False}
        with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(top, f, indent=2)
    else:
        model.save_pretrained(out_dir, safe_serialization=True)
        gc = os.path.join(out_dir, "generation_config.json")
        if os.path.exists(gc):
            os.remove(gc)

    ref = {
        "input_ids": ids.to(torch.int64).contiguous(),
        "schedule": torch.tensor(schedule, dtype=torch.int64),
        "full_logits": full.float().contiguous(),
        "inc_logits": inc.float().contiguous(),
    }
    for k, v in full_data.items():
        if k.startswith("ngram_ids_"):
            ref[k] = v.to(torch.int64).contiguous()
        elif k.startswith("qsa_mask_"):
            ref[k] = v.contiguous()
        else:
            ref[k] = v.float().contiguous()
    save_file(ref, os.path.join(out_dir, "ref.safetensors"))
    for fn in sorted(os.listdir(out_dir)):
        p = os.path.join(out_dir, fn)
        if os.path.isfile(p):
            print(f"  {fn}: {os.path.getsize(p)} bytes")


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "qwen4_exp_tiny"
    torch.set_num_threads(4)
    run_variant(cfg_a(), out, 32, [9, 20], [2, 2], [20] + [1] * 12, vlm_layout=False)
    run_variant(cfg_b(), os.path.join(out, "variant_b"), 40, [4, 17, 30], [5, 7, 5], [1, 5, 10] + [1] * 24,
                vlm_layout=True)


if __name__ == "__main__":
    main()

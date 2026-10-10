#!/usr/bin/env python3
"""Write tests/fixtures/ngram_cases.json: n-gram (PLE) row ids from the real HF hashing.

Builds `Qwen4ExpTextNGramEmbedding` for decoder layer 1 under the meta device, swaps its
320M-row embedding for a module that returns the row ids, and runs it on CPU. Multi-call cases
feed chunks through a real `DynamicCache`, exercising HF's previous_context path (step-by-step
decode).

    tools/.venv/bin/python tools/reference/make_ngram_fixture.py \
        [--hf /mnt/data/fngine/hf] [--pack /mnt/data/fngine/pack] [--out tests/fixtures/ngram_cases.json]
"""
import argparse
import json
import os
import random

import torch
from torch import nn
from transformers import DynamicCache
from transformers.models.qwen4_exp import modeling_qwen4_exp as M
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig


class RowIds(nn.Module):
    """Stands in for nn.Embedding: returns the looked-up ids as [..., 16, 1]."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, ids):
        return ids.unsqueeze(-1)


def build(hf_dir):
    text = json.load(open(os.path.join(hf_dir, "config.json")))["text_config"]
    cfg = Qwen4ExpTextConfig(**text)
    decoder_layer = cfg.ple_layer_ids[0] - 1  # PLE runs inside decoder layer index (id - 1)
    with torch.device("meta"):
        mod = M.Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, decoder_layer, ple_layer_index=0)
    mod.ngram_embedding = RowIds()
    mod.layer_multipliers = M._build_layer_multipliers(mod.unigram_vocab_size, mod.ngram_size, 0, mod.seed)
    mod.ngram_heads_vocab_sizes = torch.tensor(mod.head_vocab_sizes, dtype=torch.long)
    mod.ngram_heads_offsets = torch.tensor(mod.head_offsets, dtype=torch.long)
    return cfg, mod, decoder_layer


def run(cfg, mod, tokens, chunks):
    """Row ids [T][16] for tokens fed as consecutive calls of the given chunk sizes."""
    assert sum(chunks) == len(tokens)
    cache = DynamicCache(config=cfg) if len(chunks) > 1 else None
    out, pos = [], 0
    for n in chunks:
        ids = torch.tensor([tokens[pos:pos + n]], dtype=torch.long)
        out.append(mod(ids, cache)[0])
        pos += n
    rows = torch.cat(out, dim=0)
    assert rows.shape == (len(tokens), 16)
    return rows.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="/mnt/data/fngine/hf")
    ap.add_argument("--pack", default="/mnt/data/fngine/pack")
    ap.add_argument("--out", default=os.path.normpath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "tests", "fixtures", "ngram_cases.json")))
    args = ap.parse_args()

    cfg, mod, layer = build(args.hf)
    eos = mod.eos_token_id
    params = dict(multipliers=mod.layer_multipliers.tolist(), head_sizes=list(mod.head_vocab_sizes),
                  head_offsets=list(mod.head_offsets), eos=eos)
    manifest = os.path.join(args.pack, "manifest.json")
    if os.path.exists(manifest):
        ng = json.load(open(manifest))["ngram"]
        for k in ("multipliers", "head_sizes", "head_offsets", "eos"):
            assert ng[k] == params[k], f"pack manifest ngram.{k} differs from HF: {ng[k]} vs {params[k]}"

    rng = random.Random(1234)
    V = cfg.vocab_size

    def toks(n):
        return [rng.randrange(V) for _ in range(n)]

    cases = []

    def add(name, tokens, chunks=None):
        chunks = chunks or [len(tokens)]
        rows = run(cfg, mod, tokens, chunks)
        if len(chunks) > 1:  # the cache path must agree with one full call
            assert rows == run(cfg, mod, tokens, [len(tokens)]), name
        cases.append(dict(name=name, tokens=tokens, chunks=chunks, row_ids=rows))

    add("sequence_start", [0, V - 1, 151644, 872, 198] + toks(11))
    t = toks(14)
    t[4] = eos
    t[9] = eos
    add("eos_mid_sequence", t)
    add("eos_at_start", [eos] + toks(9))
    t = toks(16)
    t[3:6] = [eos, eos, eos]
    t[10] = eos
    t[-1] = eos
    add("repeated_eos", t)
    # Step-by-step decode through HF's cache: first call T=1 (EOS padding of the cached context),
    # then single tokens and small verify-like chunks, EOS inside and at chunk boundaries.
    t = toks(40)
    for p in (2, 7, 8, 15, 23, 31):
        t[p] = eos
    chunks = [1, 1, 3, 1, 2, 1, 5, 1, 1, 4, 9, 1, 2, 1, 3]
    chunks.append(len(t) - sum(chunks))
    add("decode_steps", t, chunks)
    t = toks(30)
    t[19] = eos
    add("prefill_then_decode", t, [17, 1, 1, 1, 1, 1, 8])

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dict(source="transformers Qwen4ExpTextNGramEmbedding (decoder layer %d)" % layer,
                       params=params, cases=cases), f, separators=(",", ":"))
        f.write("\n")
    print("wrote %s: %d cases" % (args.out, len(cases)))


if __name__ == "__main__":
    main()

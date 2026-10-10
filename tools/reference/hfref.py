"""Hugging Face reference for Qwen3.8-Flash-Next built from fngine's inputs.

Dense weights: /mnt/data/fngine/hf/dense-*.safetensors (BF16 from the official repo).
Routed experts: GGUF shard 1 (dequantized on GPU with tools/iqformats.py).
N-gram table: GGUF shard 2 (IQ4_NL rows, dequantized on demand).

The forward is transformers' own `Qwen4ExpTextModel` code: either `model(...)` directly (used with a cache for
generation) or `RefModel.forward_layerwise`, which mirrors `Qwen4ExpTextModel.forward` but runs layer by layer over
several sequences so that every layer's 512 experts are dequantized once.
"""
import glob
import json
import math
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import iqformats  # noqa: E402
from gguf import GGUFFile  # noqa: E402

from transformers.masking_utils import create_causal_mask  # noqa: E402
from transformers.models.qwen4_exp import modeling_qwen4_exp as M  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402

_create_recurrent_mask = M.create_recurrent_attention_mask


class GGUFRowEmbedding(nn.Module):
    """Stand-in for the 102 GB n-gram nn.Embedding: gathers IQ4_NL rows from GGUF shard 2."""

    def __init__(self, gguf2: GGUFFile, device):
        super().__init__()
        t = gguf2.tensors["per_layer_token_embd.weight"]
        assert t.type_name == "IQ4_NL"
        self.dim, self.rows = t.ne[0], t.ne[1]
        self.row_bytes = iqformats.row_bytes("IQ4_NL", self.dim)
        self.table = np.asarray(gguf2.tensor_bytes(t.name)).reshape(self.rows, self.row_bytes)
        # NGramEmbedding.forward reads `.weight.device` to pick the execution device
        self.weight = nn.Buffer(torch.empty(0, dtype=torch.bfloat16, device=device), persistent=False)

    def forward(self, ids):
        flat = ids.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)
        u = uniq.cpu().numpy()
        assert u.min() >= 0 and u.max() < self.rows
        raw = torch.from_numpy(np.ascontiguousarray(self.table[u]))
        rows = iqformats.dequantize_rows("IQ4_NL", raw, device=ids.device).to(self.weight.dtype)
        return rows[inv].reshape(*ids.shape, self.dim)


class GGUFExperts(nn.Module):
    """Replacement for Qwen4ExpTextExperts with identical forward semantics; weights come from GGUF shard 1."""

    def __init__(self, layer, gguf1: GGUFFile, cache, num_experts=512, hidden=2560, inter=640,
                 dtype=torch.bfloat16, ablate=None):
        super().__init__()
        self.layer, self.cache, self.dtype = layer, cache, dtype
        self.num_experts, self.hidden, self.inter = num_experts, hidden, inter
        self.act_fn = F.silu
        # Abliteration of the routed experts' down projections, W -= scale * r r^T W, applied to their
        # weighted sum (equivalent, since the sum is linear): (unit direction r [hidden], scale).
        self.ablate = ablate
        self.parts = {}
        for p in ("gate", "up", "down"):
            t = gguf1.tensors["blk.%d.ffn_%s_exps.weight" % (layer, p)]
            per = t.nbytes // num_experts
            self.parts[p] = (t.type_name, np.asarray(gguf1.tensor_bytes(t.name)).reshape(num_experts, per),
                             (inter, hidden) if p != "down" else (hidden, inter))
        self.full = None  # (gate_up [E,1280,2560], down [E,2560,640]) when load_all() was called

    def _deq(self, p, lo, hi, device, out=None):
        name, raw, shape = self.parts[p]
        if out is None:
            out = torch.empty((hi - lo, *shape), dtype=self.dtype, device=device)
        step = 32
        for a in range(lo, hi, step):
            b = min(hi, a + step)
            r = torch.from_numpy(np.ascontiguousarray(raw[a:b])).to(device)
            out[a - lo:b - lo] = iqformats.dequantize(name, r).reshape(b - a, *shape)
        return out

    def load_all(self, device):
        E, I = self.num_experts, self.inter
        gu = torch.empty((E, 2 * I, self.hidden), dtype=self.dtype, device=device)
        self._deq("gate", 0, E, device, gu[:, :I])
        self._deq("up", 0, E, device, gu[:, I:])
        self.full = (gu, self._deq("down", 0, E, device))

    def release(self):
        self.full = None

    def weights(self, e, device):
        if self.full is not None:
            return self.full[0][e], self.full[1][e]
        key = (self.layer, e)
        w = self.cache.get(key)
        if w is None:
            gu = torch.cat([self._deq("gate", e, e + 1, device)[0], self._deq("up", e, e + 1, device)[0]], 0)
            w = (gu, self._deq("down", e, e + 1, device)[0])
            self.cache.put(key, w)
        return w

    # Same loop as Qwen4ExpTextExperts.forward (modeling_qwen4_exp.py:933-957).
    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit.tolist():
            expert_idx = expert_idx[0]
            gate_up_w, down_w = self.weights(expert_idx, hidden_states.device)
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, gate_up_w).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, down_w)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        if self.ablate is not None:
            r, scale = self.ablate
            r = r.to(device=final_hidden_states.device, dtype=torch.float32)
            y = final_hidden_states.float()
            y = y - scale * (y @ r)[..., None] * r
            final_hidden_states = y.to(final_hidden_states.dtype)
        return final_hidden_states


class ExpertLRU:
    def __init__(self, budget_bytes):
        self.budget, self.used, self.d = budget_bytes, 0, OrderedDict()
        self.hits = self.misses = 0

    def get(self, k):
        w = self.d.get(k)
        if w is not None:
            self.d.move_to_end(k)
            self.hits += 1
        else:
            self.misses += 1
        return w

    def put(self, k, w):
        nb = sum(t.numel() * t.element_size() for t in w)
        while self.d and self.used + nb > self.budget:
            _, old = self.d.popitem(last=False)
            self.used -= sum(t.numel() * t.element_size() for t in old)
        self.d[k] = w
        self.used += nb

    def clear(self):
        self.d.clear()
        self.used = 0


_orig_indexer_forward = M.Qwen4ExpTextQSAIndexer.forward


def _fast_indexer_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
    """When no query sees more than block_topk complete blocks, the indexer selects every visible token,
    i.e. its mask equals the causal mask. Falls back to the real indexer otherwise."""
    kv_len = attention_mask.shape[-1]
    if kv_len // self.compress_ratio <= self.block_topk:
        if past_key_values is not None:  # keep the cache's raw keys consistent
            return _orig_indexer_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values)
        if attention_mask.dtype == torch.bool:
            return attention_mask.clone()
        return torch.where(attention_mask == 0, attention_mask.new_zeros(()), torch.finfo(attention_mask.dtype).min)
    return _orig_indexer_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values)


def set_fast_indexer(on: bool):
    M.Qwen4ExpTextQSAIndexer.forward = _fast_indexer_forward if on else _orig_indexer_forward


class RefModel:
    def __init__(self, hf_dir, gguf1, gguf2, device="cuda", expert_cache_gib=6.0, fast_indexer=False,
                 attn="eager", dtype=torch.bfloat16, abliteration=None):
        """dtype=float32 upcasts every weight exactly (BF16 source, GGUF dequant) and runs activations in FP32.
        In that mode dense weights live on the CPU and decoder layers are streamed to the GPU one at a time
        (forward_layerwise only).

        abliteration: optional dict(dense=<safetensors of edited dense tensors, HF names>,
        directions=<float tensor [48, hidden]>, scales=<float tensor [48]>). The edited dense tensors
        replace the official ones, and each layer's routed-expert sum gets W -= scale * r r^T W."""
        self.device = torch.device(device)
        self.dtype = dtype
        self.stream = dtype != torch.bfloat16
        text_config = json.load(open(os.path.join(hf_dir, "config.json")))["text_config"]
        cfg = Qwen4ExpTextConfig(**text_config)
        cfg._attn_implementation = attn
        self.cfg = cfg
        self.g1, self.g2 = GGUFFile(gguf1), GGUFFile(gguf2)
        self.cache = ExpertLRU(int(expert_cache_gib * (1 << 30)))
        set_fast_indexer(fast_indexer)

        prev = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with torch.device("meta"):
                model = M.Qwen4ExpTextModel(cfg)
                lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        finally:
            torch.set_default_dtype(prev)
        home = torch.device("cpu") if self.stream else self.device
        for i, layer in enumerate(model.layers):
            ablate = None
            if abliteration is not None:
                ablate = (abliteration["directions"][i].float(), float(abliteration["scales"][i]))
            layer.mlp.experts = GGUFExperts(i, self.g1, self.cache, dtype=dtype, ablate=ablate)
            if layer.ple is not None:
                layer.ple.ple_embedding.ngram_embedding = GGUFRowEmbedding(self.g2, home)
        model.to_empty(device=home)
        lm_head.to_empty(device=home)
        with self.device:
            model.rotary_emb = M.Qwen4ExpTextRotaryEmbedding(cfg)
        for i, layer in enumerate(model.layers):  # restore the placeholder buffer we just emptied
            if layer.ple is not None:
                layer.ple.ple_embedding.ngram_embedding.weight = torch.empty(0, dtype=dtype, device=home)
        self._load_dense(hf_dir, model, lm_head,
                         overrides=abliteration["dense"] if abliteration is not None else None)
        model.hyper_connection_mixer.to(self.device)
        model.eval()
        self.model, self.lm_head = model, lm_head
        self.revision = None
        mf = os.path.join(hf_dir, "fetch-manifest.json")
        if os.path.exists(mf):
            self.revision = json.load(open(mf))["revision"]

    @torch.no_grad()
    def _load_dense(self, hf_dir, model, lm_head, overrides=None):
        want = dict(model.state_dict())
        want = {("model." + k): v for k, v in want.items()}
        want["lm_head.weight"] = lm_head.weight
        loaded = set()
        for f in sorted(glob.glob(os.path.join(hf_dir, "dense-*.safetensors"))):
            with safe_open(f, "pt") as s:
                for name in s.keys():
                    if name.startswith("mtp."):
                        continue
                    key = "model." + name[len("model.language_model."):] if name.startswith(
                        "model.language_model.") else name
                    if key not in want:
                        raise KeyError("unexpected tensor %s" % name)
                    t = s.get_tensor(name)
                    dst = want[key]
                    if tuple(t.shape) != tuple(dst.shape):
                        raise ValueError("%s: shape %s != %s" % (name, tuple(t.shape), tuple(dst.shape)))
                    if t.dtype != dst.dtype and not (t.dtype == torch.bfloat16 and dst.dtype == torch.float32):
                        raise ValueError("%s: dtype %s != %s" % (name, t.dtype, dst.dtype))
                    dst.copy_(t)
                    loaded.add(key)
        missing = sorted(set(want) - loaded)
        if missing:
            raise KeyError("parameters not loaded: %s" % missing)
        if overrides is not None:
            replaced = 0
            with safe_open(overrides, "pt") as s:
                for name in s.keys():
                    key = "model." + name[len("model.language_model."):] if name.startswith(
                        "model.language_model.") else name
                    if key not in want:
                        raise KeyError("override %s matches no parameter" % name)
                    want[key].copy_(s.get_tensor(name))
                    replaced += 1
            self.replaced_dense = replaced

    # ------------------------------------------------------------------ forward helpers

    def logits(self, last_hidden):
        if self.lm_head.weight.device != last_hidden.device:
            self.lm_head.to(last_hidden.device)
        return self.lm_head(last_hidden).float()

    @torch.no_grad()
    def _prepare(self, ids):
        """Lines 1419-1480 of Qwen4ExpTextModel.forward for one sequence, no cache."""
        model, cfg = self.model, self.cfg
        input_ids = ids.view(1, -1).to(self.device)
        inputs_embeds = model.embed_tokens(input_ids.to(model.embed_tokens.weight.device)).to(self.device)
        ple_input_ids = input_ids
        position_ids = torch.arange(inputs_embeds.shape[1], device=self.device)
        position_ids = position_ids.view(1, 1, -1).expand(4, 1, -1)
        text_position_ids, position_ids = position_ids[0], position_ids[1:]
        mask_kwargs = dict(config=cfg, inputs_embeds=inputs_embeds, attention_mask=None, past_key_values=None,
                           position_ids=text_position_ids, allow_is_causal_skip=False)
        attn_mask = create_causal_mask(**mask_kwargs)
        conv_mask = _create_recurrent_mask(**mask_kwargs)
        if conv_mask is not None:
            eos = cfg.eos_token_id[0] if isinstance(cfg.eos_token_id, list) else cfg.eos_token_id
            ple_input_ids = torch.where(conv_mask.bool(), ple_input_ids, eos)
        pos_emb = model.rotary_emb(inputs_embeds, position_ids)
        hidden = inputs_embeds.repeat(1, 1, cfg.hc_count)
        return hidden, dict(position_embeddings=pos_emb, attention_mask=attn_mask, conv_mask=conv_mask,
                            past_key_values=None, ple_input_ids=ple_input_ids)

    @torch.no_grad()
    def forward_layerwise(self, seqs, layer_cb=None, on_final=None):
        """Run sequences (list of 1-D int tensors) layer by layer; streams stay on CPU between layers.
        layer_cb(seq_idx, layer_idx_or_-1, hidden_gpu) observes streams (-1 = embedding).
        on_final(seq_idx, mixer_output_gpu) receives the hyper_connection_mixer output [1,T,2560]."""
        model = self.model
        self.cache.clear()
        torch.cuda.empty_cache()
        hs = []
        for s, ids in enumerate(seqs):
            h, _ = self._prepare(ids)
            if layer_cb:
                layer_cb(s, -1, h)
            hs.append(h.cpu())
        for li, layer in enumerate(model.layers):
            if self.stream:
                layer.to(self.device)
            layer.mlp.experts.load_all(self.device)
            for s, ids in enumerate(seqs):
                _, kw = self._prepare(ids)
                h = layer(hs[s].to(self.device), **kw)
                if layer_cb:
                    layer_cb(s, li, h)
                hs[s] = h.cpu()
                del h, kw
            layer.mlp.experts.release()
            if self.stream:
                layer.to("cpu")
            torch.cuda.empty_cache()
        for s in range(len(seqs)):
            out = model.hyper_connection_mixer(hs[s].to(self.device))
            on_final(s, out)

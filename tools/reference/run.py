#!/usr/bin/env python3
"""Reference outputs for fngine validation.

    tools/.venv/bin/python tools/reference/run.py golden    --out /mnt/data/fngine/ref [--only G0,G1,G2]
    tools/.venv/bin/python tools/reference/run.py calibrate --out /mnt/data/fngine/ref
    tools/.venv/bin/python tools/reference/run.py check-indexer
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import corpus  # noqa: E402
import hfref  # noqa: E402
import prompts  # noqa: E402

HF = "/mnt/data/fngine/hf"
GGUF_DIR = "/home/obelisk/Projects/exploresrata/Strata-data/models/IQ3_S"
GGUF1 = os.path.join(GGUF_DIR, "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf")
GGUF2 = os.path.join(GGUF_DIR, "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf")
G0_NEW = 384
EOS = (248046, 248044)


def log(*a):
    print("[%s]" % time.strftime("%H:%M:%S"), *a, flush=True)


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(HF)


def mkout(out):
    os.makedirs(out, exist_ok=True)
    subprocess.run(["chattr", "+m", out], stderr=subprocess.DEVNULL, check=False)


ABLITERATION = None  # dict for hfref.RefModel(abliteration=...), set by --abliteration


def abliteration_from(work):
    """Edited dense tensors + per-layer routed-expert directions written by the abliteration
    tooling (abliterated-dense.safetensors under work/hf, refusal-direction.json under work)."""
    spec = json.load(open(os.path.join(work, "refusal-direction.json")))
    layers = sorted(spec["layers"], key=lambda e: e["layer"])
    return dict(dense=os.path.join(work, "hf", "abliterated-dense.safetensors"),
                directions=torch.tensor([e["unit_direction"] for e in layers], dtype=torch.float64),
                scales=torch.tensor([e.get("scale", spec["scale"]) for e in layers], dtype=torch.float64),
                source=spec.get("source"))


def load(**kw):
    t = time.time()
    R = hfref.RefModel(HF, GGUF1, GGUF2, fast_indexer=True, abliteration=ABLITERATION, **kw)
    log("model loaded in %.1fs, %.2f GiB on GPU" % (time.time() - t, torch.cuda.memory_allocated() / 2**30))
    return R


@torch.no_grad()
def greedy(R, ids, n_new):
    t = time.time()
    out = R.model(input_ids=torch.tensor([ids], device=R.device), use_cache=True)
    cache = out.past_key_values
    prompt_logits = R.logits(out.last_hidden_state[0, -1:])[0]
    nxt = int(prompt_logits.argmax())
    gen = [nxt]
    log("G0 prompt forward (%d tokens, cached) %.1fs" % (len(ids), time.time() - t))
    t = time.time()
    while len(gen) < n_new and gen[-1] not in EOS:
        out = R.model(input_ids=torch.tensor([[gen[-1]]], device=R.device), past_key_values=cache, use_cache=True)
        gen.append(int(R.logits(out.last_hidden_state[0, -1:])[0].argmax()))
        if len(gen) % 32 == 0:
            log("  generated %d tokens, %.2fs/token, lru hit %d miss %d" % (
                len(gen), (time.time() - t) / (len(gen) - 1), R.cache.hits, R.cache.misses))
    del cache
    R.cache.clear()
    torch.cuda.empty_cache()
    return gen, prompt_logits


def topk_summary(R, ids, stride, attn):
    """Teacher-forced pass; returns dict of top-32/logsumexp/full-logit samples."""
    T = len(ids)
    res = {}

    def on_final(_, h):
        vals, idx, lse, full, fpos = [], [], [], [], []
        for a in range(0, T, 256):
            lg = R.logits(h[0, a:a + 256])
            v, i = lg.topk(32, dim=-1)
            vals.append(v.cpu()); idx.append(i.int().cpu()); lse.append(torch.logsumexp(lg, -1).cpu())
            sel = [p for p in range(a, min(T, a + 256)) if p % stride == 0]
            full.append(lg[[p - a for p in sel]].half().cpu()); fpos += sel
        res.update(top32_values=torch.cat(vals), top32_ids=torch.cat(idx), logsumexp=torch.cat(lse),
                   full_logits=torch.cat(full), full_positions=torch.tensor(fpos, dtype=torch.int32))

    prev = R.cfg._attn_implementation
    R.cfg._attn_implementation = attn
    try:
        R.forward_layerwise([torch.tensor(ids)], on_final=on_final)
    finally:
        R.cfg._attn_implementation = prev
    res["ids"] = torch.tensor(ids, dtype=torch.int32)
    return res


def describe(tensors):
    return {k: dict(dtype=str(v.dtype).replace("torch.", ""), shape=list(v.shape)) for k, v in tensors.items()}


def golden(out, only, dtype="bfloat16"):
    import transformers
    from safetensors.torch import load_file
    fp32 = dtype == "float32"
    sfx = "_fp32" if fp32 else ""
    mkout(out)
    tok = tokenizer()
    R = load(dtype=torch.float32 if fp32 else torch.bfloat16)
    mpath = os.path.join(out, "golden-manifest.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    if ABLITERATION is not None:
        manifest.update(abliteration=dict(source=ABLITERATION["source"], dense=ABLITERATION["dense"],
                                          replaced_dense_tensors=getattr(R, "replaced_dense", 0),
                                          routed_expert_scale_mean=float(ABLITERATION["scales"].mean())))
    manifest.update(model="Qwen/Qwen3.8-Flash-Next", revision=R.revision, transformers=transformers.__version__,
                    torch=torch.__version__, experts="GGUF " + os.path.basename(GGUF1),
                    ngram="GGUF " + os.path.basename(GGUF2),
                    dtype="bfloat16 (files without suffix); float32 (files with _fp32 suffix: weights upcast "
                          "exactly from BF16/GGUF dequant, FP32 activations, same token ids)",
                    files=manifest.get("files", {}), timings=manifest.get("timings", {}))

    if "G0" in only:
        t0 = time.time()
        if fp32:  # same ids as the BF16 golden (its greedy continuation)
            ref = load_file(os.path.join(out, "g0.safetensors"))
            ids, plen = ref["ids"].tolist(), int(ref["prompt_len"][0])
            assert ids[:plen] == list(prompts.g0_ids(tok))
        else:
            p = list(prompts.g0_ids(tok))
            gen, gen_prompt_logits = greedy(R, p, G0_NEW)
            manifest["timings"]["G0_generate_s"] = time.time() - t0
            text = tok.decode(gen)
            with open(os.path.join(out, "g0_continuation.txt"), "w") as f:
                f.write(text)
            log("G0 continuation (%d tokens):\n%s" % (len(gen), text))
            ids, plen = p + gen, len(p)
        T = len(ids)
        sdt = torch.float32 if fp32 else torch.bfloat16
        streams = torch.empty((49, T, 10240), dtype=sdt)
        logits = {}

        def cb(_, li, h):
            streams[li + 1] = h[0].cpu()

        def fin(_, h):
            logits["x"] = torch.cat([R.logits(h[0, a:a + 256]).cpu() for a in range(0, T, 256)])

        t1 = time.time()
        R.forward_layerwise([torch.tensor(ids)], layer_cb=cb, on_final=fin)
        lg = logits["x"]
        g0 = dict(ids=torch.tensor(ids, dtype=torch.int32), prompt_len=torch.tensor([plen], dtype=torch.int32),
                  logits=lg.contiguous())
        save_file(g0, os.path.join(out, "g0%s.safetensors" % sfx))
        save_file(dict(streams=streams), os.path.join(out, "g0_streams%s.safetensors" % sfx))
        manifest["files"]["g0%s.safetensors" % sfx] = describe(g0)
        manifest["files"]["g0_streams%s.safetensors" % sfx] = dict(
            describe(dict(streams=streams)),
            note="index 0 = embedding replicated x4; index k = output of decoder layer k-1 (streams s*2560+d)")
        if not fp32:
            # consistency: cached generation vs teacher-forced pass
            d_prompt = (lg[plen - 1] - gen_prompt_logits.cpu()).abs().max().item()
            agree = (lg[plen - 1:T - 1].argmax(-1) == torch.tensor(gen[:T - plen])).float().mean().item()
            log("G0 TF: prompt-end logit maxdiff vs cached %.4g, TF argmax == generated %.4f" % (d_prompt, agree))
            manifest["files"]["g0_continuation.txt"] = "decoded greedy continuation (%d tokens)" % len(gen)
            manifest["g0"] = dict(prompt=prompts.G0_USER, prompt_len=plen, generated=len(gen),
                                  gen_vs_tf_prompt_logit_maxdiff=d_prompt, gen_vs_tf_argmax_agreement=agree)
        manifest["timings"]["G0%s_teacher_forced_s" % sfx] = time.time() - t1
        log("G0%s teacher-forced: %d tokens in %.1fs" % (sfx, T, time.time() - t1))
        del streams, lg, logits

    for name, fn, stride, attn in (("G1", prompts.g1_ids, 8, "eager"), ("G2", prompts.g2_ids, 32, "sdpa")):
        if name not in only:
            continue
        t1 = time.time()
        ids = list(fn(tok))
        if fp32:
            assert load_file(os.path.join(out, name.lower() + ".safetensors"))["ids"].tolist() == ids
        res = topk_summary(R, ids, stride, attn)
        fname = name.lower() + sfx + ".safetensors"
        save_file(res, os.path.join(out, fname))
        manifest["files"][fname] = describe(res)
        manifest[name.lower() + sfx] = dict(tokens=len(ids), full_logit_stride=stride, attention=attn,
                                            indexer="dense-equivalent (T<=2051)" if len(ids) <= 2051
                                            else "real HF indexer")
        manifest["timings"][name + sfx + "_s"] = time.time() - t1
        log("%s%s: %d tokens in %.1fs" % (name, sfx, len(ids), time.time() - t1))
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)


def calibrate(out):
    mkout(out)
    tok = tokenizer()
    seqs = corpus.build(tok)
    total = sum(len(i) for _, i in seqs)
    log("corpus: %d sequences, %d tokens" % (len(seqs), total))
    R = load()
    counts = torch.zeros((48, 512), dtype=torch.int64, device=R.device)
    hooks = []
    for li, layer in enumerate(R.model.layers):
        def hook(_m, _inp, outp, li=li):
            counts[li] += torch.bincount(outp[2].reshape(-1), minlength=512)
        hooks.append(layer.mlp.gate.register_forward_hook(hook))
    t = time.time()
    R.forward_layerwise([torch.tensor(i) for _, i in seqs], on_final=lambda s, h: None)
    for h in hooks:
        h.remove()
    c = counts.cpu().numpy()
    assert c.sum() == 48 * 10 * total, (c.sum(), 48 * 10 * total)
    np.save(os.path.join(out, "routing-counts.npy"), c)
    with open(os.path.join(out, "routing-corpus.json"), "w") as f:
        json.dump(dict(sequences=[dict(path=p, tokens=len(i)) for p, i in seqs], total_tokens=total), f, indent=1)
    k = max(1, 512 // 100)
    share = np.sort(c, axis=1)[:, ::-1][:, :k].sum(1) / c.sum(1)
    log("routing counts sum %d == 48*10*%d; calibration %.1fs" % (c.sum(), total, time.time() - t))
    log("per-layer share of routing mass in top 1%% (%d experts): %s" % (k, np.round(share, 3).tolist()))


@torch.no_grad()
def check_indexer():
    tok = tokenizer()
    ids = prompts.g1_ids(tok)[:600]
    R = load()
    _, kw = R._prepare(torch.tensor(ids))
    h = R.model.embed_tokens(torch.tensor([ids], device=R.device))
    idx = R.model.layers[3].self_attn.indexer
    a = hfref._orig_indexer_forward(idx, h, kw["position_embeddings"], kw["attention_mask"], None)
    b = hfref._fast_indexer_forward(idx, h, kw["position_embeddings"], kw["attention_mask"], None)
    log("indexer check (layer 3, %d tokens): equal=%s" % (len(ids), torch.equal(a, b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["golden", "calibrate", "check-indexer"])
    ap.add_argument("--out", default="/mnt/data/fngine/ref")
    ap.add_argument("--only", default="G0,G1,G2")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--abliteration", help="work dir with hf/abliterated-dense.safetensors and "
                                           "refusal-direction.json: goldens of the abliterated model")
    a = ap.parse_args()
    global ABLITERATION
    if a.abliteration:
        ABLITERATION = abliteration_from(a.abliteration)
    if a.cmd == "golden":
        golden(a.out, a.only.split(","), a.dtype)
    elif a.cmd == "calibrate":
        calibrate(a.out)
    else:
        check_indexer()


if __name__ == "__main__":
    main()

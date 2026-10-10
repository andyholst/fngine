"""Helpers to load golden reference files and compare logits.

    from compare import load_golden, kld_top1
    g = load_golden("/mnt/data/fngine/ref", "g1", "fp32")
    kld, top1, per_pos = kld_top1(my_logits, g["full_logits"])

    python tools/reference/compare.py noise-floor [--ref /mnt/data/fngine/ref]   # FP32 vs BF16 golden
"""
import argparse
import json
import os

import torch
from safetensors.torch import load_file

NAMES = ("g0", "g1", "g2")


def golden_path(ref_dir, name, dtype="bf16", streams=False):
    assert name in NAMES and dtype in ("bf16", "fp32")
    base = name + ("_streams" if streams else "") + ("_fp32" if dtype == "fp32" else "")
    return os.path.join(ref_dir, base + ".safetensors")


def load_golden(ref_dir, name, dtype="bf16", streams=False):
    """Returns the tensors of g0/g1/g2 (bf16 = HF BF16 run, fp32 = HF FP32 run).
    g0: ids, prompt_len, logits [T,V] f32 (+ streams [49,T,10240] when streams=True).
    g1/g2: ids, top32_values, top32_ids, logsumexp, full_logits [n,V] f16, full_positions."""
    d = load_file(golden_path(ref_dir, name, dtype))
    if streams:
        d["streams"] = load_file(golden_path(ref_dir, name, dtype, streams=True))["streams"]
    return d


@torch.no_grad()
def kld_top1(logits_a, logits_b, device=None, chunk=64):
    """KL(softmax(a) || softmax(b)) per position in FP64 and top-1 agreement.
    Returns (mean_kld, top1_agreement, per_position_kld [n] float64 CPU)."""
    assert logits_a.shape == logits_b.shape
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    kl, agree = [], 0
    for i in range(0, logits_a.shape[0], chunk):
        a = logits_a[i:i + chunk].to(device, torch.float64)
        b = logits_b[i:i + chunk].to(device, torch.float64)
        la, lb = a.log_softmax(-1), b.log_softmax(-1)
        kl.append((la.exp() * (la - lb)).sum(-1).cpu())
        agree += (a.argmax(-1) == b.argmax(-1)).sum().item()
    kl = torch.cat(kl)
    return kl.mean().item(), agree / logits_a.shape[0], kl


@torch.no_grad()
def stream_cosine_distance(sa, sb, device=None):
    """Per layer: (1 - cosine over the whole [T,10240] stream, max over positions of 1 - per-position cosine)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for k in range(sa.shape[0]):
        a, b = sa[k].to(device, torch.float64), sb[k].to(device, torch.float64)
        whole = 1 - (a * b).sum() / (a.norm() * b.norm())
        per = 1 - (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))
        out.append((whole.item(), per.max().item()))
    return out


def noise_floor(ref_dir):
    res = {}
    a, b = load_golden(ref_dir, "g0", "fp32", streams=True), load_golden(ref_dir, "g0", "bf16", streams=True)
    assert torch.equal(a["ids"], b["ids"])
    kld, top1, per = kld_top1(a["logits"], b["logits"])
    cos = stream_cosine_distance(a["streams"], b["streams"])
    res["g0"] = dict(positions=int(a["ids"].numel()), mean_kld=kld, max_kld=per.max().item(), top1=top1,
                     stream_cosine_distance_whole_max=max(c[0] for c in cos),
                     stream_cosine_distance_per_position_max=max(c[1] for c in cos),
                     per_layer=[dict(index=k, whole=c[0], worst_position=c[1]) for k, c in enumerate(cos)])
    for name in ("g1", "g2"):
        a, b = load_golden(ref_dir, name, "fp32"), load_golden(ref_dir, name, "bf16")
        assert torch.equal(a["ids"], b["ids"]) and torch.equal(a["full_positions"], b["full_positions"])
        kld, top1, per = kld_top1(a["full_logits"], b["full_logits"])
        top1_all = (a["top32_ids"][:, 0] == b["top32_ids"][:, 0]).float().mean().item()
        res[name] = dict(full_logit_positions=int(a["full_positions"].numel()), mean_kld=kld,
                         max_kld=per.max().item(), top1_full_positions=top1, top1_all_positions=top1_all,
                         positions=int(a["ids"].numel()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["noise-floor"])
    ap.add_argument("--ref", default="/mnt/data/fngine/ref")
    a = ap.parse_args()
    res = noise_floor(a.ref)
    mpath = os.path.join(a.ref, "golden-manifest.json")
    m = json.load(open(mpath))
    m["noise_floor"] = dict(description="FP32 HF reference vs BF16 HF reference (KL(fp32||bf16), FP64 math)",
                            **res)
    with open(mpath, "w") as f:
        json.dump(m, f, indent=1)
    show = {k: {kk: vv for kk, vv in v.items() if kk != "per_layer"} for k, v in res.items()}
    print(json.dumps(show, indent=1))
    print("g0 per-layer stream cosine distance (whole, worst position):")
    for r in res["g0"]["per_layer"]:
        print("  %2d %.3e %.3e" % (r["index"], r["whole"], r["worst_position"]))


if __name__ == "__main__":
    main()

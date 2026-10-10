#!/usr/bin/env python3
"""Verify an fngine pack against its sources. Exit 0 iff every check passes.

    tools/.venv/bin/python tools/verify_pack.py --pack /mnt/data/fngine/pack --gguf SHARD1 --ngram-gguf SHARD2 \
        [--hf /mnt/data/fngine/hf] [--hf-repo Qwen/Qwen3.8-Flash-Next] [--revision SHA]

Checks: expert blobs byte-equal the GGUF slices (all 48x512, zero padding); MTP IQ4_NL rows vs HF BF16 (cosine);
every dense tensor vs the HF source after the documented transform (FP8 within E4M3 rounding of dense_bf16.bin);
ngram.bin header + random rows vs GGUF; mapping sanity vs BF16 originals range-fetched from the HF repo.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp8row  # noqa: E402
import iqformats  # noqa: E402
from fetch_hf import HF, get, shard_header  # noqa: E402
from gguf import GGUFFile  # noqa: E402
from safetensors import safe_open  # noqa: E402

HFP = "model.language_model."
FAILS: list[str] = []


def check(ok, msg):
    if not ok:
        FAILS.append(msg)
        print("FAIL: " + msg, flush=True)
    return ok


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def pread(fd, n, off):
    buf = np.empty(n, dtype=np.uint8)
    got = os.preadv(fd, [memoryview(buf)], off)
    if got != n:
        raise IOError(f"short read {got}/{n} at {off}")
    return buf


def bf16_from_bytes(b):
    return torch.from_numpy(np.ascontiguousarray(b).view(np.int16).copy()).view(torch.bfloat16)


# ----------------------------------------------------------------------------- HF expectations (independent of convert.py)

class HFLocal:
    def __init__(self, hf_dir):
        man = json.load(open(os.path.join(hf_dir, "fetch-manifest.json")))
        self.manifest = man
        self.path = {t["name"]: os.path.join(hf_dir, t["out"]) for t in man["tensors"]}
        self.h = {}

    def get(self, name):
        p = self.path[name]
        if p not in self.h:
            self.h[p] = safe_open(p, framework="pt")
        return self.h[p].get_tensor(name)

    def get_slice(self, name):
        p = self.path[name]
        if p not in self.h:
            self.h[p] = safe_open(p, framework="pt")
        return self.h[p].get_slice(name)


def hf_prefix(pack_prefix):
    """'L7' -> 'model.language_model.layers.7', 'mtp' -> 'mtp.layers.0', '' (global) -> model.language_model."""
    if pack_prefix.startswith("L"):
        return f"{HFP}layers.{int(pack_prefix[1:])}"
    return "mtp.layers.0"


def expected(name, hf, dev):
    """Expected pack tensor (bf16 for fp8/bf16 tensors, float32 for f32 tensors) from the HF sources."""
    g = lambda n: hf.get(n).to(dev)  # noqa: E731
    one_plus = lambda t: 1.0 + t.float()  # noqa: E731

    def gr_up(src):
        r = torch.arange(10240, device=dev)
        out = torch.zeros(10240, 384, dtype=torch.bfloat16, device=dev)
        out[:, :320] = src[(r % 4) * 2560 + r // 4]      # stored row d*4+s <- source row s*2560+d
        return out

    if name == "embed":
        return g(HFP + "embed_tokens.weight")
    if name == "lm_head":
        return g("lm_head.weight")
    m = re.fullmatch(r"(mtp\.)?mixer\.(hc_norm|down|up)", name)
    if m:
        base = ("mtp." if m.group(1) else HFP) + "hyper_connection_mixer."
        part = m.group(2)
        if part == "hc_norm":
            return one_plus(g(base + "hc_norm.weight"))
        if part == "down":
            return g(base + "input_mix_weight_down.weight")
        return gr_up(g(base + "input_mix_weight_up.weight"))
    if name.startswith("mtp.") and name.split(".")[1] in ("fc_embedding", "fc_hidden"):
        return g(name + ".weight")
    if name in ("mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden"):
        return one_plus(g(name + ".weight"))
    pre, blk, part = name.split(".", 2)
    src = hf_prefix(pre)
    if blk in ("attn_gr", "mlp_gr"):
        s = f"{src}.{blk[:-3]}_hyper_connection."
        return {"hc_norm": lambda: one_plus(g(s + "hc_norm.weight")),
                "down": lambda: g(s + "input_mix_weight_down.weight"),
                "up": lambda: gr_up(g(s + "input_mix_weight_up.weight")),
                "inject": lambda: g(s + "block_inject_weight.weight")}[part]()
    if blk == "gdn":
        s = f"{src}.linear_attn."
        return {"qkvz": lambda: torch.cat([g(s + "in_proj_qkv.weight"), g(s + "in_proj_z.weight")]),
                "ab": lambda: torch.cat([g(s + "in_proj_a.weight"), g(s + "in_proj_b.weight")]),
                "conv": lambda: g(s + "conv1d.weight").float()[:, 0, :],
                "A_log": lambda: g(s + "A_log").float(),
                "dt_bias": lambda: g(s + "dt_bias").float(),
                "norm": lambda: g(s + "norm.weight").float(),
                "out": lambda: g(s + "out_proj.weight")}[part]()
    if blk == "qsa":
        s = f"{src}.self_attn."
        return {"qkv": lambda: torch.cat([g(s + "q_proj.weight"), g(s + "k_proj.weight"), g(s + "v_proj.weight")]),
                "q_norm": lambda: one_plus(g(s + "q_norm.weight")),
                "k_norm": lambda: one_plus(g(s + "k_norm.weight")),
                "o": lambda: g(s + "o_proj.weight"),
                "idx_qk": lambda: g(s + "indexer.index_qk_proj.weight"),
                "idx_q_norm": lambda: one_plus(g(s + "indexer.q_layernorm.weight")),
                "idx_k_norm": lambda: one_plus(g(s + "indexer.k_layernorm.weight"))}[part]()
    if blk == "moe":
        s = f"{src}.mlp."
        return {"router": lambda: torch.cat([g(s + "gate.weight"), g(s + "shared_expert_gate.weight")]),
                "shared_gate_up": lambda: torch.cat([g(s + "shared_expert.gate_proj.weight"),
                                                     g(s + "shared_expert.up_proj.weight")]),
                "shared_down": lambda: g(s + "shared_expert.down_proj.weight")}[part]()
    if blk == "ple" and pre == "L1":
        s = f"{src}.ple."
        return {"key": lambda: g(s + "key_proj.weight"), "value": lambda: g(s + "value_proj.weight"),
                "norm_key": lambda: one_plus(g(s + "norm_key.weight")),
                "norm_query": lambda: one_plus(g(s + "norm_query.weight")),
                "norm_conv": lambda: one_plus(g(s + "norm_conv.weight")),
                "conv": lambda: g(s + "conv1d.weight").float()[:, 0, :]}[part]()
    raise KeyError(name)


def expected_names():
    n = ["embed", "lm_head", "mixer.hc_norm", "mixer.down", "mixer.up"]
    gr = lambda p: [f"{p}.{k}" for k in ("hc_norm", "down", "up", "inject")]  # noqa: E731
    qsa = ["qkv", "q_norm", "k_norm", "o", "idx_qk", "idx_q_norm", "idx_k_norm"]
    moe = ["router", "shared_gate_up", "shared_down"]
    for i in range(48):
        n += gr(f"L{i}.attn_gr") + gr(f"L{i}.mlp_gr")
        if (i + 1) % 4 == 0:
            n += [f"L{i}.qsa.{k}" for k in qsa]
        else:
            n += [f"L{i}.gdn.{k}" for k in ("qkvz", "ab", "conv", "A_log", "dt_bias", "norm", "out")]
        n += [f"L{i}.moe.{k}" for k in moe]
    n += [f"L1.ple.{k}" for k in ("key", "value", "norm_key", "norm_query", "norm_conv", "conv")]
    n += ["mtp.fc_embedding", "mtp.fc_hidden", "mtp.pre_fc_norm_embedding", "mtp.pre_fc_norm_hidden"]
    n += gr("mtp.attn_gr") + gr("mtp.mlp_gr") + [f"mtp.qsa.{k}" for k in qsa] + [f"mtp.moe.{k}" for k in moe]
    n += ["mtp.mixer.hc_norm", "mtp.mixer.down", "mtp.mixer.up"]
    return n


def verify_dense(pack, man, hf, dev):
    print("== dense tensors", flush=True)
    names = expected_names()
    check(sorted(names) == sorted(man["tensors"]), "dense tensor name set differs from spec")
    fp8_names = [n for n, t in man["tensors"].items() if t["format"] == "fp8_e4m3_row"]
    check(sorted(fp8_names) == sorted(man["tensors_bf16"]), "tensors_bf16 names != fp8 tensor names")
    fd = os.open(os.path.join(pack, "dense.bin"), os.O_RDONLY)
    fd16 = os.open(os.path.join(pack, "dense_bf16.bin"), os.O_RDONLY)
    sizes = {"dense": os.fstat(fd).st_size, "dense_bf16": os.fstat(fd16).st_size}
    worst_fp8 = 0.0
    try:
        for name in names:
            t = man["tensors"].get(name)
            if t is None:
                continue
            check(t["offset"] % 4096 == 0 and (t["scale_offset"] is None or t["scale_offset"] % 4096 == 0),
                  f"{name}: unaligned")
            check(t["offset"] + t["bytes"] <= sizes["dense"], f"{name}: beyond dense.bin")
            shape = tuple(t["shape"])
            exp = expected(name, hf, dev)
            if not check(tuple(exp.shape) == shape, f"{name}: shape {tuple(exp.shape)} != {shape}"):
                continue
            raw = pread(fd, t["bytes"], t["offset"])
            if t["format"] == "f32":
                got = torch.from_numpy(raw.view(np.float32).copy()).reshape(shape).to(dev)
                check(exp.dtype == torch.float32 and torch.equal(got, exp), f"{name}: f32 values differ")
            elif t["format"] == "bf16":
                got = bf16_from_bytes(raw).reshape(shape).to(dev)
                check(torch.equal(got.view(torch.int16), exp.view(torch.int16)), f"{name}: bf16 values differ")
            else:
                b = man["tensors_bf16"][name]
                check(b["offset"] % 4096 == 0 and b["offset"] + b["bytes"] <= sizes["dense_bf16"],
                      f"{name}: bad dense_bf16 region")
                ref = bf16_from_bytes(pread(fd16, b["bytes"], b["offset"])).reshape(shape).to(dev)
                check(torch.equal(ref.view(torch.int16), exp.view(torch.int16)),
                      f"{name}: dense_bf16 copy != HF after transform")
                codes = torch.from_numpy(raw).reshape(shape).to(dev)
                scales = bf16_from_bytes(pread(fd, t["scale_bytes"], t["scale_offset"])).to(dev)
                check(t["scale_bytes"] == 2 * shape[0], f"{name}: scale_bytes")
                ok, zero_ok = True, True
                for r0 in range(0, shape[0], 8192):
                    c, sc = codes[r0:r0 + 8192], scales[r0:r0 + 8192]
                    x = ref[r0:r0 + 8192].float()
                    tol = x.abs() * 2.0 ** -4 + sc.float()[:, None] * 2.0 ** -9
                    err = (fp8row.dequantize_rows(c, sc) - x).abs()
                    ok &= bool((err <= tol).all())
                    worst_fp8 = max(worst_fp8, float((err / tol.clamp_min(1e-38)).max()))
                    zero_rows = x.abs().amax(dim=1) == 0
                    zero_ok &= bool((sc.view(torch.int16)[zero_rows] == 0).all())
                check(ok, f"{name}: fp8 dequant outside E4M3 rounding")
                check(zero_ok, f"{name}: zero row with nonzero scale")
    finally:
        os.close(fd)
        os.close(fd16)
    print(f"  {len(names)} tensors ({len(fp8_names)} fp8); worst fp8 err/tol = {worst_fp8:.3f}", flush=True)


def verify_experts(pack, man, g: GGUFFile):
    print("== experts.bin (all 48x512 blobs)", flush=True)
    ex = man["experts"]
    fd = os.open(os.path.join(pack, "experts.bin"), os.O_RDONLY)
    gfd = os.open(g.path, os.O_RDONLY)
    off_expect = 0
    t0 = time.time()
    try:
        check(os.fstat(fd).st_size == sum(512 * L["blob_stride"] for L in ex["layers"]), "experts.bin size")
        for l, L in enumerate(ex["layers"]):
            check(L["offset"] == off_expect and L["offset"] % 4096 == 0, f"layer {l} offset")
            off_expect += 512 * L["blob_stride"]
            ti = {k: g.tensors[f"blk.{l}.ffn_{k}_exps.weight"] for k in ("gate", "up", "down")}
            check(ti["gate"].type_name == L["gu_type"] and ti["down"].type_name == L["down_type"]
                  and iqformats.TYPE_IDS[L["gu_type"]] == L["gu_type_id"]
                  and iqformats.TYPE_IDS[L["down_type"]] == L["down_type_id"], f"layer {l} types")
            gb, db = L["gate_bytes"], L["down_bytes"]
            check(gb == 640 * iqformats.row_bytes(L["gu_type"], 2560) and L["up_bytes"] == gb
                  and db == 2560 * iqformats.row_bytes(L["down_type"], 640)
                  and L["blob_bytes"] == 2 * gb + db and L["blob_stride"] == (L["blob_bytes"] + 4095) // 4096 * 4096,
                  f"layer {l} sizes")
            stride = L["blob_stride"]
            src = {k: pread(gfd, ti[k].nbytes, ti[k].offset).reshape(512, -1) for k in ti}
            blobs = pread(fd, 512 * stride, L["offset"]).reshape(512, stride)
            ok = (np.array_equal(blobs[:, :gb], src["gate"]) and np.array_equal(blobs[:, gb:2 * gb], src["up"])
                  and np.array_equal(blobs[:, 2 * gb:2 * gb + db], src["down"])
                  and not blobs[:, 2 * gb + db:].any())
            check(ok, f"layer {l}: expert blobs differ from GGUF or padding nonzero")
    finally:
        os.close(fd)
        os.close(gfd)
    print(f"  48 layers byte-equal ({time.time() - t0:.0f}s)", flush=True)


def verify_mtp(pack, man, hf, rng):
    print("== mtp_experts.bin (64 sampled rows vs HF BF16)", flush=True)
    m = man["mtp_experts"]
    stride = m["blob_stride"]
    check(os.path.getsize(os.path.join(pack, "mtp_experts.bin")) == 512 * stride, "mtp_experts.bin size")
    gu = hf.get_slice("mtp.layers.0.mlp.experts.gate_up_proj")
    dn = hf.get_slice("mtp.layers.0.mlp.experts.down_proj")
    fd = os.open(os.path.join(pack, "mtp_experts.bin"), os.O_RDONLY)
    cos = []
    try:
        for _ in range(64):
            e = int(rng.integers(512))
            kind = int(rng.integers(3))
            if kind < 2:
                r = int(rng.integers(640))
                rb = 2560 // 32 * 18
                off = e * stride + kind * m["gate_bytes"] + r * rb
                ref = gu[e:e + 1, kind * 640 + r:kind * 640 + r + 1].float().reshape(-1).numpy()
            else:
                r = int(rng.integers(2560))
                rb = 640 // 32 * 18
                off = e * stride + 2 * m["gate_bytes"] + r * rb
                ref = dn[e:e + 1, r:r + 1].float().reshape(-1).numpy()
            got = iqformats.dequantize("IQ4_NL", pread(fd, rb, off)).numpy()
            c = cosine(got, ref)
            cos.append(c)
            check(c >= 0.99, f"mtp expert {e} {['gate', 'up', 'down'][kind]} row {r}: cosine {c:.4f}")
    finally:
        os.close(fd)
    print(f"  cosine min {min(cos):.5f} mean {np.mean(cos):.5f}", flush=True)


def verify_ngram(pack, man, g2: GGUFFile, ple_meta, rng):
    print("== ngram.bin (header + 10,000 random rows)", flush=True)
    t = g2.tensors["per_layer_token_embd.weight"]
    ng = man["ngram"]
    rows = t.ne[1]
    path = os.path.join(pack, "ngram.bin")
    fd = os.open(path, os.O_RDONLY)
    gfd = os.open(g2.path, os.O_RDONLY)
    try:
        hdr = pread(fd, 4096, 0).tobytes()
        magic, (ver, nrows, rb, rpp, dim, typ) = hdr[:4], struct.unpack_from("<IQIIII", hdr, 4)
        check(magic == b"FNGR" and ver == 1 and nrows == rows and rb == 90 and rpp == 45 and dim == 160 and typ == 20
              and not any(hdr[32:]), f"ngram header {hdr[:32]!r}")
        check(ng["rows"] == rows and ng["row_bytes"] == 90 and ng["rows_per_page"] == 45, "manifest ngram fields")
        pages = (rows + 44) // 45
        check(os.fstat(fd).st_size == 4096 + pages * 4096, "ngram.bin size")
        sample = np.concatenate([rng.integers(0, rows, 9998), [0, rows - 1]])
        bad = 0
        for r in sample:
            r = int(r)
            a = pread(fd, 90, 4096 + (r // 45) * 4096 + (r % 45) * 90)
            b = pread(gfd, 90, t.offset + r * 90)
            bad += not np.array_equal(a, b)
        check(bad == 0, f"ngram: {bad} of {len(sample)} rows differ")
        tail = pread(fd, 46, 4096 + int(rng.integers(pages)) * 4096 + 45 * 90)
        check(not tail.any(), "ngram page tail nonzero")
    finally:
        os.close(fd)
        os.close(gfd)
    print(f"  {len(sample)} rows checked", flush=True)
    for k in ("multipliers", "head_sizes", "head_offsets"):
        key = {"multipliers": "layer_multipliers", "head_sizes": "head_vocab_sizes", "head_offsets": "head_offsets"}[k]
        check(list(ng[k]) == [int(v) for v in np.asarray(ple_meta[f"qwen4exp.ple.{key}"]).reshape(-1)],
              f"manifest ngram {k}")


class Remote:
    def __init__(self, repo, rev, hf_dir):
        self.base = f"{HF}/{repo}/resolve/{rev}/"
        self.wmap = json.load(open(os.path.join(hf_dir, "model.safetensors.index.json")))["weight_map"]
        self.hdrs = {}

    def info(self, name):
        shard = self.wmap[name]
        if shard not in self.hdrs:
            self.hdrs[shard] = shard_header(self.base, shard)
        start, hdr = self.hdrs[shard]
        t = hdr[name]
        assert t["dtype"] == "BF16", t
        return shard, start + t["data_offsets"][0], t["shape"]

    def row(self, name, flat_row, row_elems):
        shard, base, _ = self.info(name)
        off = base + flat_row * row_elems * 2
        data = get(self.base + shard, off, off + row_elems * 2 - 1)
        return bf16_from_bytes(np.frombuffer(data, dtype=np.uint8)).float().numpy()


def verify_mapping(pack, man, rng, remote: Remote):
    print("== mapping sanity vs HF BF16 originals (HTTP range fetch)", flush=True)
    # n-gram: 128 shards concatenated along dim 0 in numeric order
    shard_names = sorted((n for n in remote.wmap if re.fullmatch(
        re.escape(HFP) + r"layers\.1\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight", n)),
        key=lambda n: int(re.search(r"shard_(\d+)", n).group(1)))
    counts = [remote.info(n)[2][0] for n in shard_names]
    check(all(remote.info(n)[2][1] == 160 for n in shard_names), "ngram shard dim != 160")
    starts = np.concatenate([[0], np.cumsum(counts)])
    rows = man["ngram"]["rows"]
    check(len(shard_names) == 128 and starts[-1] == rows, f"ngram shards: {len(shard_names)} shards, {starts[-1]} rows")
    fd = os.open(os.path.join(pack, "ngram.bin"), os.O_RDONLY)
    try:
        for r in rng.integers(0, rows, 32):
            r = int(r)
            j = int(np.searchsorted(starts, r, side="right") - 1)
            ref = remote.row(shard_names[j], r - int(starts[j]), 160)
            got = iqformats.dequantize("IQ4_NL", pread(fd, 90, 4096 + (r // 45) * 4096 + (r % 45) * 90)).numpy()
            c = cosine(got, ref)
            print(f"  ngram row {r:10d} (shard_{j}) cosine {c:.5f}", flush=True)
            check(c >= 0.98, f"ngram row {r}: cosine {c:.4f} < 0.98")
    finally:
        os.close(fd)
    fd = os.open(os.path.join(pack, "experts.bin"), os.O_RDONLY)
    try:
        for _ in range(8):
            l, e = int(rng.integers(48)), int(rng.integers(512))
            L = man["experts"]["layers"][l]
            kind = int(rng.integers(3))
            base = L["offset"] + e * L["blob_stride"]
            if kind < 2:
                r = int(rng.integers(640))
                rb = iqformats.row_bytes(L["gu_type"], 2560)
                got = iqformats.dequantize(L["gu_type"], pread(fd, rb, base + kind * L["gate_bytes"] + r * rb)).numpy()
                ref = remote.row(f"{HFP}layers.{l}.mlp.experts.gate_up_proj", e * 1280 + kind * 640 + r, 2560)
            else:
                r = int(rng.integers(2560))
                rb = iqformats.row_bytes(L["down_type"], 640)
                got = iqformats.dequantize(L["down_type"], pread(fd, rb, base + 2 * L["gate_bytes"] + r * rb)).numpy()
                ref = remote.row(f"{HFP}layers.{l}.mlp.experts.down_proj", e * 2560 + r, 640)
            c = cosine(got, ref)
            print(f"  expert L{l:2d} e{e:3d} {['gate', 'up', 'down'][kind]:4s} row {r:4d} "
                  f"({L['gu_type'] if kind < 2 else L['down_type']}) cosine {c:.5f}", flush=True)
            check(c >= 0.90, f"expert L{l} e{e} {kind} row {r}: cosine {c:.4f} < 0.90")
    finally:
        os.close(fd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ngram-gguf", required=True)
    ap.add_argument("--hf", default="/mnt/data/fngine/hf")
    ap.add_argument("--hf-repo", default="Qwen/Qwen3.8-Flash-Next")
    ap.add_argument("--revision")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", default="", help="comma list of: dense,experts,mtp,ngram,mapping")
    a = ap.parse_args()
    skip = set(filter(None, a.skip.split(",")))
    t0 = time.time()
    man = json.load(open(os.path.join(a.pack, "manifest.json")))
    check(man.get("format") == "fngine-pack" and man.get("version") == 1, "manifest format/version")
    hf = HFLocal(a.hf)
    rev = a.revision or hf.manifest["revision"]
    g = GGUFFile(a.gguf)
    g2 = GGUFFile(a.ngram_gguf)
    ple_meta = {k: v for k, v in g.metadata.items() if k.startswith("qwen4exp.ple.")}
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "dense" not in skip:
        verify_dense(a.pack, man, hf, dev)
    if "experts" not in skip:
        verify_experts(a.pack, man, g)
    if "mtp" not in skip:
        verify_mtp(a.pack, man, hf, rng)
    if "ngram" not in skip:
        verify_ngram(a.pack, man, g2, ple_meta, rng)
    if "mapping" not in skip:
        verify_mapping(a.pack, man, rng, Remote(a.hf_repo, rev, a.hf))
    print(f"\n{'PASS' if not FAILS else 'FAIL'}: {len(FAILS)} failures, {time.time() - t0:.0f}s")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()

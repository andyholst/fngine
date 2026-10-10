#!/usr/bin/env python3
"""Build the fngine pack (format v1, see the pack spec) from the GSQ-RCO GGUF shards and the fetched HF BF16 tensors.

    tools/.venv/bin/python tools/convert.py --gguf SHARD1 --ngram-gguf SHARD2 --hf /mnt/data/fngine/hf \
        --out /mnt/data/fngine/pack [--profile counts.npy] [--skip-ngram] [--only dense,experts,mtp,ngram,manifest]

Parts not selected by --only (or --skip-ngram) are not rewritten; their existing files must already have the
expected size. --profile always (re)writes profile.bin. The manifest is written when `manifest` is selected.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fp8row  # noqa: E402
import iqformats  # noqa: E402
from gguf import GGUFFile  # noqa: E402

ALIGN = 4096
HFP = "model.language_model."
N_LAYERS = 48
N_EXPERTS = 512
HIDDEN = 2560
INTER = 640
HC = 4
GR_LOWRANK = 320
GR_LOWRANK_PAD = 384
MTP_STRIDE = 2764800
NGRAM_ROW_BYTES = 90
NGRAM_ROWS_PER_PAGE = 45
NGRAM_DIM = 160
EOS = 248044
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja",
                   "config.json", "generation_config.json"]
PARTS = ["dense", "experts", "mtp", "ngram", "manifest"]
IO_CHUNK = 64 << 20


def align(x: int, a: int = ALIGN) -> int:
    return (x + a - 1) // a * a


def is_qsa(i: int) -> bool:
    return (i + 1) % 4 == 0


# ----------------------------------------------------------------------------- dense tensor table

@dataclass(frozen=True)
class Entry:
    name: str
    fmt: str            # fp8_e4m3_row | bf16 | f32
    shape: tuple
    srcs: tuple         # HF tensor names
    xform: str          # cat | grup | onepw | w | conv


def _gr(out, src):
    return [Entry(f"{out}.hc_norm", "f32", (HC * HIDDEN,), (f"{src}.hc_norm.weight",), "onepw"),
            Entry(f"{out}.down", "fp8_e4m3_row", (GR_LOWRANK, HC * HIDDEN), (f"{src}.input_mix_weight_down.weight",), "cat"),
            Entry(f"{out}.up", "fp8_e4m3_row", (HC * HIDDEN, GR_LOWRANK_PAD), (f"{src}.input_mix_weight_up.weight",), "grup")]


def _layer_gr(out, src):
    return _gr(out, src) + [Entry(f"{out}.inject", "bf16", (HC, HC * HIDDEN), (f"{src}.block_inject_weight.weight",), "cat")]


def _qsa(out, src):
    return [Entry(f"{out}.qkv", "fp8_e4m3_row", (13312, HIDDEN),
                  (f"{src}.q_proj.weight", f"{src}.k_proj.weight", f"{src}.v_proj.weight"), "cat"),
            Entry(f"{out}.q_norm", "f32", (256,), (f"{src}.q_norm.weight",), "onepw"),
            Entry(f"{out}.k_norm", "f32", (256,), (f"{src}.k_norm.weight",), "onepw"),
            Entry(f"{out}.o", "fp8_e4m3_row", (HIDDEN, 6144), (f"{src}.o_proj.weight",), "cat"),
            Entry(f"{out}.idx_qk", "bf16", (640, HIDDEN), (f"{src}.indexer.index_qk_proj.weight",), "cat"),
            Entry(f"{out}.idx_q_norm", "f32", (128,), (f"{src}.indexer.q_layernorm.weight",), "onepw"),
            Entry(f"{out}.idx_k_norm", "f32", (128,), (f"{src}.indexer.k_layernorm.weight",), "onepw")]


def _moe(out, src):
    return [Entry(f"{out}.router", "bf16", (N_EXPERTS + 1, HIDDEN),
                  (f"{src}.gate.weight", f"{src}.shared_expert_gate.weight"), "cat"),
            Entry(f"{out}.shared_gate_up", "fp8_e4m3_row", (2 * INTER, HIDDEN),
                  (f"{src}.shared_expert.gate_proj.weight", f"{src}.shared_expert.up_proj.weight"), "cat"),
            Entry(f"{out}.shared_down", "fp8_e4m3_row", (HIDDEN, INTER), (f"{src}.shared_expert.down_proj.weight",), "cat")]


def dense_table() -> list[Entry]:
    t = [Entry("embed", "bf16", (248320, HIDDEN), (HFP + "embed_tokens.weight",), "cat"),
         Entry("lm_head", "fp8_e4m3_row", (248320, HIDDEN), ("lm_head.weight",), "cat")]
    t += _gr("mixer", HFP + "hyper_connection_mixer")
    for i in range(N_LAYERS):
        L, src = f"L{i}", f"{HFP}layers.{i}"
        t += _layer_gr(f"{L}.attn_gr", f"{src}.attn_hyper_connection")
        t += _layer_gr(f"{L}.mlp_gr", f"{src}.mlp_hyper_connection")
        if is_qsa(i):
            t += _qsa(f"{L}.qsa", f"{src}.self_attn")
        else:
            a = f"{src}.linear_attn"
            t += [Entry(f"{L}.gdn.qkvz", "fp8_e4m3_row", (16384, HIDDEN), (f"{a}.in_proj_qkv.weight", f"{a}.in_proj_z.weight"), "cat"),
                  Entry(f"{L}.gdn.ab", "bf16", (96, HIDDEN), (f"{a}.in_proj_a.weight", f"{a}.in_proj_b.weight"), "cat"),
                  Entry(f"{L}.gdn.conv", "f32", (10240, 4), (f"{a}.conv1d.weight",), "conv"),
                  Entry(f"{L}.gdn.A_log", "f32", (48,), (f"{a}.A_log",), "w"),
                  Entry(f"{L}.gdn.dt_bias", "f32", (48,), (f"{a}.dt_bias",), "w"),
                  Entry(f"{L}.gdn.norm", "f32", (128,), (f"{a}.norm.weight",), "w"),
                  Entry(f"{L}.gdn.out", "fp8_e4m3_row", (HIDDEN, 6144), (f"{a}.out_proj.weight",), "cat")]
        t += _moe(f"{L}.moe", f"{src}.mlp")
        if i == 1:
            p = f"{src}.ple"
            t += [Entry("L1.ple.key", "fp8_e4m3_row", (10240, HIDDEN), (f"{p}.key_proj.weight",), "cat"),
                  Entry("L1.ple.value", "fp8_e4m3_row", (HIDDEN, HIDDEN), (f"{p}.value_proj.weight",), "cat"),
                  Entry("L1.ple.norm_key", "f32", (10240,), (f"{p}.norm_key.weight",), "onepw"),
                  Entry("L1.ple.norm_query", "f32", (10240,), (f"{p}.norm_query.weight",), "onepw"),
                  Entry("L1.ple.norm_conv", "f32", (10240,), (f"{p}.norm_conv.weight",), "onepw"),
                  Entry("L1.ple.conv", "f32", (10240, 4), (f"{p}.conv1d.weight",), "conv")]
    t += [Entry("mtp.fc_embedding", "fp8_e4m3_row", (HIDDEN, HIDDEN), ("mtp.fc_embedding.weight",), "cat"),
          Entry("mtp.fc_hidden", "fp8_e4m3_row", (HIDDEN, HIDDEN), ("mtp.fc_hidden.weight",), "cat"),
          Entry("mtp.pre_fc_norm_embedding", "f32", (HIDDEN,), ("mtp.pre_fc_norm_embedding.weight",), "onepw"),
          Entry("mtp.pre_fc_norm_hidden", "f32", (HC * HIDDEN,), ("mtp.pre_fc_norm_hidden.weight",), "onepw")]
    t += _layer_gr("mtp.attn_gr", "mtp.layers.0.attn_hyper_connection")
    t += _layer_gr("mtp.mlp_gr", "mtp.layers.0.mlp_hyper_connection")
    t += _qsa("mtp.qsa", "mtp.layers.0.self_attn")
    t += _moe("mtp.moe", "mtp.layers.0.mlp")
    t += _gr("mtp.mixer", "mtp.hyper_connection_mixer")
    return t


ELEM_BYTES = {"fp8_e4m3_row": 1, "bf16": 2, "f32": 4}


def dense_layout(table):
    """-> (tensors, tensors_bf16, dense_size, dense_bf16_size) manifest dicts."""
    tensors, tensors_bf16 = {}, {}
    cur = cur16 = 0
    for e in table:
        n = int(np.prod(e.shape))
        off = align(cur)
        cur = off + n * ELEM_BYTES[e.fmt]
        rec = {"file": "dense", "format": e.fmt, "shape": list(e.shape), "offset": off, "bytes": n * ELEM_BYTES[e.fmt],
               "scale_offset": None, "scale_bytes": 0}
        if e.fmt == "fp8_e4m3_row":
            rec["scale_offset"] = align(cur)
            rec["scale_bytes"] = 2 * e.shape[0]
            cur = rec["scale_offset"] + rec["scale_bytes"]
            o16 = align(cur16)
            tensors_bf16[e.name] = {"file": "dense_bf16", "format": "bf16", "shape": list(e.shape), "offset": o16,
                                    "bytes": 2 * n}
            cur16 = o16 + 2 * n
        tensors[e.name] = rec
    return tensors, tensors_bf16, align(cur), align(cur16)


class HFTensors:
    def __init__(self, hf_dir):
        man = json.load(open(os.path.join(hf_dir, "fetch-manifest.json")))
        self.manifest = man
        self.where = {t["name"]: os.path.join(hf_dir, t["out"]) for t in man["tensors"]}
        self.meta = {t["name"]: t for t in man["tensors"]}
        self._open = {}

    def handle(self, name):
        path = self.where[name]
        if path not in self._open:
            self._open[path] = safe_open(path, framework="pt")
        return self._open[path]

    def get(self, name) -> torch.Tensor:
        return self.handle(name).get_tensor(name)

    def get_slice(self, name):
        return self.handle(name).get_slice(name)


def materialize(e: Entry, hf: HFTensors, device) -> torch.Tensor:
    """bf16 (for fp8/bf16 entries) or float32 (f32 entries) tensor of shape e.shape on device."""
    srcs = [hf.get(s).to(device) for s in e.srcs]
    for s, x in zip(e.srcs, srcs):
        if x.dtype != torch.bfloat16:
            raise ValueError(f"{s}: expected BF16, got {x.dtype}")
    if e.xform == "cat":
        y = torch.cat(srcs, dim=0) if len(srcs) > 1 else srcs[0]
    elif e.xform == "grup":
        (x,) = srcs
        assert tuple(x.shape) == (HC * HIDDEN, GR_LOWRANK), (e.name, x.shape)
        y = torch.zeros(HC * HIDDEN, GR_LOWRANK_PAD, dtype=torch.bfloat16, device=device)
        y[:, :GR_LOWRANK] = x.view(HC, HIDDEN, GR_LOWRANK).permute(1, 0, 2).reshape(HC * HIDDEN, GR_LOWRANK)
    elif e.xform == "onepw":
        y = 1.0 + srcs[0].float()
    elif e.xform == "w":
        y = srcs[0].float()
    elif e.xform == "conv":
        y = srcs[0].float().reshape(srcs[0].shape[0], -1)
    else:
        raise ValueError(e.xform)
    if tuple(y.shape) != tuple(e.shape):
        raise ValueError(f"{e.name}: shape {tuple(y.shape)} != {e.shape}")
    want = torch.float32 if e.fmt == "f32" else torch.bfloat16
    assert y.dtype == want, (e.name, y.dtype)
    return y.contiguous()


def tensor_bytes(t: torch.Tensor) -> memoryview:
    t = t.contiguous().cpu()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    elif t.dtype == torch.float8_e4m3fn:
        t = t.view(torch.uint8)
    return memoryview(t.numpy()).cast("B")


class AlignedWriter:
    def __init__(self, path):
        self.path = path
        self.part = path + ".part"
        self.f = open(self.part, "wb", buffering=0)
        self.pos = 0

    def write_at(self, off, data):
        assert off >= self.pos, (off, self.pos)
        if off > self.pos:
            self.f.write(bytes(off - self.pos))
        mv = memoryview(data).cast("B")
        for i in range(0, len(mv), IO_CHUNK):
            self.f.write(mv[i:i + IO_CHUNK])
        self.pos = off + len(mv)

    def finish(self, size):
        if size > self.pos:
            self.f.write(bytes(size - self.pos))
            self.pos = size
        assert self.pos == size, (self.path, self.pos, size)
        self.f.close()
        os.replace(self.part, self.path)


def write_dense(table, tensors, tensors_bf16, dense_size, bf16_size, hf, out, device):
    w = AlignedWriter(os.path.join(out, "dense.bin"))
    w16 = AlignedWriter(os.path.join(out, "dense_bf16.bin"))
    for e in table:
        y = materialize(e, hf, device)
        rec = tensors[e.name]
        if e.fmt == "fp8_e4m3_row":
            codes, scales = [], []
            for r0 in range(0, y.shape[0], 16384):
                c, s = fp8row.quantize_rows(y[r0:r0 + 16384])
                codes.append(c.cpu())
                scales.append(s.cpu())
            w.write_at(rec["offset"], tensor_bytes(torch.cat(codes)))
            w.write_at(rec["scale_offset"], tensor_bytes(torch.cat(scales)))
            w16.write_at(tensors_bf16[e.name]["offset"], tensor_bytes(y))
        else:
            w.write_at(rec["offset"], tensor_bytes(y))
        del y
    w.finish(dense_size)
    w16.finish(bf16_size)


# ----------------------------------------------------------------------------- experts

def expert_layout(g: GGUFFile):
    layers, off = [], 0
    for l in range(N_LAYERS):
        ti = {k: g.tensors[f"blk.{l}.ffn_{k}_exps.weight"] for k in ("gate", "up", "down")}
        gu, dn = ti["gate"].type_name, ti["down"].type_name
        if ti["up"].type_name != gu:
            raise ValueError(f"layer {l}: gate type {gu} != up type {ti['up'].type_name}")
        if ti["gate"].ne != [HIDDEN, INTER, N_EXPERTS] or ti["up"].ne != [HIDDEN, INTER, N_EXPERTS] \
                or ti["down"].ne != [INTER, HIDDEN, N_EXPERTS]:
            raise ValueError(f"layer {l}: unexpected expert shapes")
        gb = INTER * iqformats.row_bytes(gu, HIDDEN)
        db = HIDDEN * iqformats.row_bytes(dn, INTER)
        for k, b in (("gate", gb), ("up", gb), ("down", db)):
            if ti[k].nbytes != N_EXPERTS * b:
                raise ValueError(f"layer {l} {k}: nbytes {ti[k].nbytes} != 512*{b}")
        blob = 2 * gb + db
        stride = align(blob)
        layers.append({"gu_type": gu, "down_type": dn, "gu_type_id": iqformats.TYPE_IDS[gu],
                       "down_type_id": iqformats.TYPE_IDS[dn], "gate_bytes": gb, "up_bytes": gb, "down_bytes": db,
                       "blob_bytes": blob, "blob_stride": stride, "offset": off})
        off += N_EXPERTS * stride
    return layers, off


def expert_src_offsets(g: GGUFFile, l: int, e: int, lay):
    return [(g.tensors[f"blk.{l}.ffn_{k}_exps.weight"].offset + e * lay[f"{k}_bytes"], lay[f"{k}_bytes"])
            for k in ("gate", "up", "down")]


def write_experts(g: GGUFFile, layers, size, out):
    w = AlignedWriter(os.path.join(out, "experts.bin"))
    fd = os.open(g.path, os.O_RDONLY)
    try:
        for l, lay in enumerate(layers):
            stride = lay["blob_stride"]
            group = max(1, IO_CHUNK // stride)
            buf = np.zeros((group, stride), dtype=np.uint8)
            for e0 in range(0, N_EXPERTS, group):
                n = min(group, N_EXPERTS - e0)
                for i in range(n):
                    pos = 0
                    for src, nb in expert_src_offsets(g, l, e0 + i, lay):
                        got = os.preadv(fd, [memoryview(buf[i, pos:pos + nb])], src)
                        if got != nb:
                            raise IOError(f"short read at {src}")
                        pos += nb
                w.write_at(lay["offset"] + e0 * stride, buf[:n])
            print(f"  experts layer {l:2d} {lay['gu_type']:7s}/{lay['down_type']:6s} stride {stride}", flush=True)
    finally:
        os.close(fd)
    w.finish(size)


# ----------------------------------------------------------------------------- MTP experts (IQ4_NL)

def write_mtp_experts(hf: HFTensors, out, device, group=32):
    gu = hf.get_slice("mtp.layers.0.mlp.experts.gate_up_proj")
    dn = hf.get_slice("mtp.layers.0.mlp.experts.down_proj")
    if list(gu.get_shape()) != [N_EXPERTS, 2 * INTER, HIDDEN] or list(dn.get_shape()) != [N_EXPERTS, HIDDEN, INTER]:
        raise ValueError("unexpected MTP expert shapes")
    w = AlignedWriter(os.path.join(out, "mtp_experts.bin"))
    for e0 in range(0, N_EXPERTS, group):
        e1 = min(N_EXPERTS, e0 + group)
        a = gu[e0:e1].to(device).float()                                   # [n,1280,2560]
        d = dn[e0:e1].to(device).float()                                   # [n,2560,640]
        qa = iqformats.quantize_iq4_nl(a.reshape(-1, HIDDEN)).reshape(e1 - e0, -1)   # gate rows then up rows
        qd = iqformats.quantize_iq4_nl(d.reshape(-1, INTER)).reshape(e1 - e0, -1)
        blob = torch.cat([qa, qd], dim=1).cpu().numpy()
        assert blob.shape[1] == MTP_STRIDE
        w.write_at(e0 * MTP_STRIDE, blob)
        print(f"  mtp experts {e1}/{N_EXPERTS}", flush=True)
    w.finish(N_EXPERTS * MTP_STRIDE)


# ----------------------------------------------------------------------------- n-gram table

def ngram_info(g2: GGUFFile):
    t = g2.tensors["per_layer_token_embd.weight"]
    if t.type_name != "IQ4_NL" or t.ne[0] != NGRAM_DIM:
        raise ValueError(f"unexpected n-gram tensor {t}")
    rows = t.ne[1]
    if t.nbytes != rows * NGRAM_ROW_BYTES:
        raise ValueError("n-gram row size mismatch")
    pages = (rows + NGRAM_ROWS_PER_PAGE - 1) // NGRAM_ROWS_PER_PAGE
    return t, rows, ALIGN + pages * ALIGN


def ngram_header(rows: int) -> bytes:
    import struct
    h = b"FNGR" + struct.pack("<IQIIII", 1, rows, NGRAM_ROW_BYTES, NGRAM_ROWS_PER_PAGE, NGRAM_DIM,
                              iqformats.TYPE_IDS["IQ4_NL"])
    return h + bytes(ALIGN - len(h))


def write_ngram(g2: GGUFFile, out):
    t, rows, size = ngram_info(g2)
    w = AlignedWriter(os.path.join(out, "ngram.bin"))
    w.write_at(0, ngram_header(rows))
    per_page = NGRAM_ROWS_PER_PAGE * NGRAM_ROW_BYTES
    group = IO_CHUNK // ALIGN
    inb = np.zeros((group, per_page), dtype=np.uint8)
    outb = np.zeros((group, ALIGN), dtype=np.uint8)
    fd = os.open(g2.path, os.O_RDONLY)
    pages = (rows + NGRAM_ROWS_PER_PAGE - 1) // NGRAM_ROWS_PER_PAGE
    t0 = time.time()
    try:
        for p0 in range(0, pages, group):
            n = min(group, pages - p0)
            r0 = p0 * NGRAM_ROWS_PER_PAGE
            nrows = min(rows - r0, n * NGRAM_ROWS_PER_PAGE)
            nb = nrows * NGRAM_ROW_BYTES
            flat = inb.reshape(-1)
            got = os.preadv(fd, [memoryview(flat[:nb])], t.offset + r0 * NGRAM_ROW_BYTES)
            if got != nb:
                raise IOError("short n-gram read")
            flat[nb:n * per_page] = 0
            outb[:n, :per_page] = inb[:n]
            w.write_at(ALIGN + p0 * ALIGN, outb[:n])
            if (p0 // group) % 64 == 0:
                print(f"  ngram {p0 / pages * 100:5.1f}% {time.time() - t0:6.0f}s", flush=True)
    finally:
        os.close(fd)
    w.finish(size)


# ----------------------------------------------------------------------------- misc

def sha256_head(path, n=16 << 20):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read(n)).hexdigest()


def check_ple(g: GGUFFile, hf: HFTensors):
    src = HFP + "layers.1.ple.ple_embedding."
    pairs = [("qwen4exp.ple.layer_multipliers", "layer_multipliers"),
             ("qwen4exp.ple.head_vocab_sizes", "ngram_heads_vocab_sizes"),
             ("qwen4exp.ple.head_offsets", "ngram_heads_offsets")]
    vals = {}
    for gk, hk in pairs:
        a = [int(v) for v in np.asarray(g.metadata[gk]).reshape(-1)]
        b = [int(v) for v in hf.get(src + hk).reshape(-1).tolist()]
        if a != b:
            sys.exit(f"ABORT: GGUF {gk}={a} differs from HF {src + hk}={b}")
        vals[hk] = a
    return vals


def expect_size(path, size, part):
    have = os.path.getsize(path) if os.path.exists(path) else None
    if have != size:
        sys.exit(f"ABORT: {path} has size {have}, expected {size}; rerun with --only including '{part}'")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--ngram-gguf", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile")
    ap.add_argument("--skip-ngram", action="store_true")
    ap.add_argument("--only", default=",".join(PARTS))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    only = {p.strip() for p in a.only.split(",") if p.strip()}
    if only - set(PARTS):
        sys.exit(f"unknown --only parts {sorted(only - set(PARTS))}")
    if a.skip_ngram:
        only.discard("ngram")
    t_start = time.time()
    os.makedirs(a.out, exist_ok=True)
    subprocess.run(["chattr", "+m", a.out], stderr=subprocess.DEVNULL, check=False)

    g = GGUFFile(a.gguf)
    g2 = GGUFFile(a.ngram_gguf)
    hf = HFTensors(a.hf)
    ple = check_ple(g, hf)
    cfg = json.load(open(os.path.join(a.hf, "config.json")))["text_config"]
    gen_path = os.path.join(a.hf, "generation_config.json")
    gen = json.load(open(gen_path)) if os.path.exists(gen_path) else None

    table = dense_table()
    used = {s for e in table for s in e.srcs}
    known = set(hf.where)
    if used - known:
        sys.exit(f"ABORT: missing HF tensors: {sorted(used - known)[:10]}")
    tensors, tensors_bf16, dense_size, bf16_size = dense_layout(table)
    layers, experts_size = expert_layout(g)
    _, ngram_rows, ngram_size = ngram_info(g2)
    mtp_size = N_EXPERTS * MTP_STRIDE
    files = {"dense": (os.path.join(a.out, "dense.bin"), dense_size),
             "dense_bf16": (os.path.join(a.out, "dense_bf16.bin"), bf16_size),
             "experts": (os.path.join(a.out, "experts.bin"), experts_size),
             "mtp_experts": (os.path.join(a.out, "mtp_experts.bin"), mtp_size),
             "ngram": (os.path.join(a.out, "ngram.bin"), ngram_size)}
    part_of = {"dense": "dense", "dense_bf16": "dense", "experts": "experts", "mtp_experts": "mtp", "ngram": "ngram"}
    for f, (path, size) in files.items():
        if part_of[f] not in only:
            expect_size(path, size, part_of[f])

    times = {}
    device = torch.device(a.device)
    if "dense" in only:
        t0 = time.time()
        print("writing dense.bin + dense_bf16.bin", flush=True)
        write_dense(table, tensors, tensors_bf16, dense_size, bf16_size, hf, a.out, device)
        times["dense"] = time.time() - t0
    if "experts" in only:
        t0 = time.time()
        print("writing experts.bin", flush=True)
        write_experts(g, layers, experts_size, a.out)
        times["experts"] = time.time() - t0
    if "mtp" in only:
        t0 = time.time()
        print("writing mtp_experts.bin", flush=True)
        write_mtp_experts(hf, a.out, device)
        times["mtp"] = time.time() - t0
    if "ngram" in only:
        t0 = time.time()
        print("writing ngram.bin", flush=True)
        write_ngram(g2, a.out)
        times["ngram"] = time.time() - t0

    profile = None
    if a.profile:
        counts = np.load(a.profile)
        if counts.shape != (N_LAYERS, N_EXPERTS) or (counts < 0).any():
            sys.exit(f"ABORT: profile must be non-negative [48,512], got {counts.shape}")
        path = os.path.join(a.out, "profile.bin")
        with open(path + ".part", "wb") as f:
            f.write(counts.astype("<u8").tobytes())
        os.replace(path + ".part", path)
        profile = {"file": "profile.bin", "source": os.path.abspath(a.profile), "total": int(counts.sum())}

    if "manifest" in only:
        tok_dir = os.path.join(a.out, "tokenizer")
        os.makedirs(tok_dir, exist_ok=True)
        tok = []
        for fn in TOKENIZER_FILES:
            src = os.path.join(a.hf, fn)
            if os.path.exists(src):
                shutil.copyfile(src, os.path.join(tok_dir, fn))
                tok.append("tokenizer/" + fn)
        man = {
            "format": "fngine-pack", "version": 1,
            "config": cfg,
            "generation_config": gen,
            "files": {"dense": "dense.bin", "dense_bf16": "dense_bf16.bin", "experts": "experts.bin",
                      "mtp_experts": "mtp_experts.bin", "ngram": "ngram.bin",
                      "profile": "profile.bin" if profile else None},
            "tensors": tensors,
            "tensors_bf16": tensors_bf16,
            "experts": {"n_layers": N_LAYERS, "n_experts": N_EXPERTS, "hidden": HIDDEN, "intermediate": INTER,
                        "layers": layers},
            "mtp_experts": {"n_experts": N_EXPERTS, "gu_type": "IQ4_NL", "down_type": "IQ4_NL",
                            "gate_bytes": INTER * HIDDEN // 32 * 18, "up_bytes": INTER * HIDDEN // 32 * 18,
                            "down_bytes": HIDDEN * INTER // 32 * 18, "blob_bytes": MTP_STRIDE,
                            "blob_stride": MTP_STRIDE, "offset": 0},
            "ngram": {"rows": ngram_rows, "row_bytes": NGRAM_ROW_BYTES, "rows_per_page": NGRAM_ROWS_PER_PAGE,
                      "header_bytes": ALIGN, "dim": NGRAM_DIM, "type": "IQ4_NL", "layer": 1,
                      "ngram_size": int(g.metadata["qwen4exp.ple.ngram_size"]),
                      "heads_per_ngram": int(g.metadata["qwen4exp.ple.heads_per_ngram"]),
                      "n_heads": len(ple["ngram_heads_vocab_sizes"]),
                      "multipliers": ple["layer_multipliers"], "head_sizes": ple["ngram_heads_vocab_sizes"],
                      "head_offsets": ple["ngram_heads_offsets"], "eos": int(g.metadata["qwen4exp.ple.eos_token_id"])},
            "profile": profile,
            "tokenizer": tok,
            "provenance": {
                "gguf": {"path": os.path.abspath(a.gguf), "size": os.path.getsize(a.gguf),
                         "sha256_first_16MiB": sha256_head(a.gguf)},
                "ngram_gguf": {"path": os.path.abspath(a.ngram_gguf), "size": os.path.getsize(a.ngram_gguf),
                               "sha256_first_16MiB": sha256_head(a.ngram_gguf)},
                "hf": {"repo": hf.manifest["repo"], "revision": hf.manifest["revision"]},
                "created": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "converter": "tools/convert.py"},
        }
        assert man["ngram"]["eos"] == EOS
        path = os.path.join(a.out, "manifest.json")
        with open(path + ".part", "w") as f:
            json.dump(man, f, indent=1)
        os.replace(path + ".part", path)

    print("\nsummary")
    for f, (path, size) in files.items():
        print(f"  {os.path.basename(path):18s} {size / 2**30:8.3f} GiB")
    hist = {}
    for lay in layers:
        hist[("gate/up", lay["gu_type"])] = hist.get(("gate/up", lay["gu_type"]), 0) + 1
        hist[("down", lay["down_type"])] = hist.get(("down", lay["down_type"]), 0) + 1
    print("  expert types: " + ", ".join(f"{k[0]} {k[1]} x{v}" for k, v in sorted(hist.items())))
    print(f"  dense tensors: {len(tensors)} ({len(tensors_bf16)} fp8)")
    print("  times: " + ", ".join(f"{k} {v:.0f}s" for k, v in times.items()) + f", total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()

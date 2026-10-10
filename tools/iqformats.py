# Modified for fngine from llama.cpp/ggml:ggml/src/ggml-quants.c @ 3cf03257f219afbe7334045ff7c6a06ac68c627d. See NOTICE.
"""Torch (CPU/CUDA) dequantizers for the ggml formats fngine reads, plus the ggml IQ4_NL quantizer.

Ported from the ggml reference code extracted in `third_party/ggml-iq/ggml-iq.c`; the
results are bit-identical to that C code (see tools/tests/test_iqformats.py).
"""
from __future__ import annotations

import functools
import pathlib
import re

import numpy as np
import torch

GGML_IQ_DIR = pathlib.Path(__file__).resolve().parent.parent / "third_party" / "ggml-iq"

TYPE_IDS = {"Q8_0": 8, "IQ3_XXS": 18, "IQ4_NL": 20, "IQ3_S": 21, "IQ2_S": 22, "IQ4_XS": 23, "Q2_0": 42}
TYPE_NAMES = {v: k for k, v in TYPE_IDS.items()}
BLOCK_ELEMS = {"IQ2_S": 256, "IQ3_XXS": 256, "IQ3_S": 256, "IQ4_XS": 256, "IQ4_NL": 32, "Q2_0": 64, "Q8_0": 32}
BLOCK_BYTES = {"IQ2_S": 82, "IQ3_XXS": 98, "IQ3_S": 110, "IQ4_XS": 136, "IQ4_NL": 18, "Q2_0": 18, "Q8_0": 34}

_TABLE_DTYPES = {"uint8_t": np.uint8, "int8_t": np.int8, "uint32_t": np.uint32, "uint64_t": np.uint64}


@functools.cache
def _tables_np() -> dict[str, np.ndarray]:
    text = (GGML_IQ_DIR / "ggml-iq-tables.h").read_text()
    out = {}
    for m in re.finditer(r"static const (\w+) (\w+)\[(\d+)\] = \{(.*?)\};", text, re.S):
        ctype, name, n, body = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        vals = [int(v, 0) for v in body.replace("\n", " ").split(",") if v.strip()]
        assert len(vals) == n, name
        out[name] = np.array(vals, dtype=_TABLE_DTYPES[ctype])
    return out


def tables() -> dict[str, np.ndarray]:
    """The ggml lookup tables as numpy arrays (exact C values)."""
    return dict(_tables_np())


@functools.cache
def _tables_torch(device: str) -> dict[str, torch.Tensor]:
    t = _tables_np()
    dev = torch.device(device)
    # grids as little-endian byte rows, exactly as the C code reads them via (const uint8_t *)
    return {
        "iq2s_grid": torch.from_numpy(t["iq2s_grid"].astype("<u8").view(np.uint8).reshape(-1, 8).copy()).to(dev).float(),
        "iq3xxs_grid": torch.from_numpy(t["iq3xxs_grid"].astype("<u4").view(np.uint8).reshape(-1, 4).copy()).to(dev).float(),
        "iq3s_grid": torch.from_numpy(t["iq3s_grid"].astype("<u4").view(np.uint8).reshape(-1, 4).copy()).to(dev).float(),
        "ksigns_iq2xs": torch.from_numpy(t["ksigns_iq2xs"].astype(np.int64)).to(dev),
        "kvalues_iq4nl": torch.from_numpy(t["kvalues_iq4nl"].astype(np.float32)).to(dev),
    }


def _tt(device: torch.device) -> dict[str, torch.Tensor]:
    return _tables_torch(str(device))


def _as_u8(raw, device=None) -> torch.Tensor:
    if isinstance(raw, (bytes, bytearray, memoryview)):
        raw = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    elif isinstance(raw, np.ndarray):
        raw = torch.from_numpy(np.ascontiguousarray(raw).reshape(-1).view(np.uint8))
    elif not isinstance(raw, torch.Tensor):
        raise TypeError(f"unsupported raw type {type(raw)}")
    raw = raw.reshape(-1)
    if raw.dtype != torch.uint8:
        raw = raw.contiguous().view(torch.uint8)
    if device is not None:
        raw = raw.to(device)
    return raw


def _f16(lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
    """fp16 bytes -> fp32 (exact; equals ggml_compute_fp16_to_fp32 for all non-NaN inputs)."""
    bits = lo.to(torch.int32) | (hi.to(torch.int32) << 8)
    return bits.to(torch.int16).view(torch.float16).float()


def _signs(v: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
    return torch.where(neg, -v, v)


def _bits8(b: torch.Tensor) -> torch.Tensor:
    """byte tensor [...] -> bool [..., 8] with bit j at index j."""
    sh = torch.arange(8, device=b.device, dtype=torch.int32)
    return ((b.to(torch.int32).unsqueeze(-1) >> sh) & 1).bool()


def _deq_iq2_s(b, T):
    d = _f16(b[:, 0], b[:, 1])                              # [B]
    qs = b[:, 2:34].reshape(-1, 8, 4).long()                # [B,ib32,l]
    sg = b[:, 34:66].reshape(-1, 8, 4)
    qh = b[:, 66:74].long()                                 # [B,ib32]
    sc = b[:, 74:82].to(torch.int32)                        # [B,ib32]
    db0 = d[:, None] * (0.5 + (sc & 0xF).float()) * 0.25
    db1 = d[:, None] * (0.5 + (sc >> 4).float()) * 0.25
    dl = torch.stack([db0, db0, db1, db1], dim=-1)          # [B,ib32,l]
    l = torch.arange(4, device=b.device)
    idx = qs | ((qh[:, :, None] >> (2 * l)) & 3) << 8
    grid = T["iq2s_grid"][idx]                              # [B,8,4,8]
    v = dl[..., None] * grid
    return _signs(v, _bits8(sg)).reshape(-1)


def _deq_iq3_xxs(b, T):
    d = _f16(b[:, 0], b[:, 1])
    qs = b[:, 2:66].reshape(-1, 8, 4, 2).long()             # [B,ib32,l,2]
    aux = b[:, 66:98].reshape(-1, 8, 4).to(torch.int64)
    aux = aux[..., 0] | aux[..., 1] << 8 | aux[..., 2] << 16 | aux[..., 3] << 24   # [B,ib32]
    db = d[:, None] * (0.5 + (aux >> 28).float()) * 0.5
    l = torch.arange(4, device=b.device)
    signs = T["ksigns_iq2xs"][(aux[:, :, None] >> (7 * l)) & 127]                 # [B,ib32,l]
    grid = T["iq3xxs_grid"][qs].reshape(qs.shape[0], 8, 4, 8)
    v = db[:, :, None, None] * grid
    return _signs(v, _bits8(signs)).reshape(-1)


def _deq_iq3_s(b, T):
    d = _f16(b[:, 0], b[:, 1])
    qs = b[:, 2:66].reshape(-1, 8, 4, 2).long()             # [B,ib32,l,2]
    qh = b[:, 66:74].long()                                 # [B,ib32]
    sg = b[:, 74:106].reshape(-1, 8, 4)
    sc = b[:, 106:110].to(torch.int32)
    s = torch.stack([sc & 0xF, sc >> 4], dim=-1).reshape(-1, 8)
    db = d[:, None] * (1 + 2 * s).float()
    bit = torch.arange(8, device=b.device).reshape(4, 2)    # bit 2l (first grid), 2l+1 (second)
    idx = qs | ((qh[:, :, None, None] >> bit) & 1) << 8
    grid = T["iq3s_grid"][idx].reshape(qs.shape[0], 8, 4, 8)
    v = db[:, :, None, None] * grid
    return _signs(v, _bits8(sg)).reshape(-1)


def _nl_pairs(qs, vals):
    """qs [..., 16] bytes -> values [..., 32]: low nibbles then high nibbles."""
    q = qs.long()
    return torch.cat([vals[q & 0xF], vals[q >> 4]], dim=-1)


def _deq_iq4_xs(b, T):
    d = _f16(b[:, 0], b[:, 1])
    sh = (b[:, 2].to(torch.int32) | b[:, 3].to(torch.int32) << 8)
    sl = b[:, 4:8].to(torch.int32)
    ib = torch.arange(8, device=b.device)
    ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF) | (((sh[:, None] >> (2 * ib)) & 3) << 4)
    dl = d[:, None] * (ls - 32).float()                     # [B,8]
    q = _nl_pairs(b[:, 8:136].reshape(-1, 8, 16), T["kvalues_iq4nl"])
    return (dl[..., None] * q).reshape(-1)


def _deq_iq4_nl(b, T):
    d = _f16(b[:, 0], b[:, 1])
    return (d[:, None] * _nl_pairs(b[:, 2:18], T["kvalues_iq4nl"])).reshape(-1)


def _deq_q2_0(b, T):
    d = _f16(b[:, 0], b[:, 1])
    sh = torch.arange(0, 8, 2, device=b.device, dtype=torch.int32)
    q = (b[:, 2:18].to(torch.int32)[..., None] >> sh) & 3   # [B,16,4] -> element 4*byte + k
    return (((q - 1).float()).reshape(-1, 64) * d[:, None]).reshape(-1)


def _deq_q8_0(b, T):
    d = _f16(b[:, 0], b[:, 1])
    return (b[:, 2:34].view(torch.int8).float() * d[:, None]).reshape(-1)


_DEQ = {"IQ2_S": _deq_iq2_s, "IQ3_XXS": _deq_iq3_xxs, "IQ3_S": _deq_iq3_s, "IQ4_XS": _deq_iq4_xs,
        "IQ4_NL": _deq_iq4_nl, "Q2_0": _deq_q2_0, "Q8_0": _deq_q8_0}


def _type_name(t) -> str:
    if isinstance(t, int):
        return TYPE_NAMES[t]
    return t.upper()


def dequantize(qtype, raw, n_elements: int | None = None, device=None) -> torch.Tensor:
    """Dequantize whole blocks of `qtype` (name or ggml type id) to float32 [n]."""
    name = _type_name(qtype)
    b = _as_u8(raw, device)
    bb = BLOCK_BYTES[name]
    if b.numel() % bb:
        raise ValueError(f"{name}: {b.numel()} bytes is not a whole number of {bb}-byte blocks")
    y = _DEQ[name](b.reshape(-1, bb), _tt(b.device))
    if n_elements is not None:
        if n_elements > y.numel():
            raise ValueError(f"{name}: need {n_elements} elements, raw holds {y.numel()}")
        y = y[:n_elements]
    return y


def row_bytes(qtype, row_elems: int) -> int:
    name = _type_name(qtype)
    assert row_elems % BLOCK_ELEMS[name] == 0
    return row_elems // BLOCK_ELEMS[name] * BLOCK_BYTES[name]


def dequantize_rows(qtype, raw2d, device=None) -> torch.Tensor:
    """raw2d [rows, row_bytes] -> float32 [rows, row_elems]."""
    rows = raw2d.shape[0]
    return dequantize(_type_name(qtype), raw2d, device=device).reshape(rows, -1)


# ---------------------------------------------------------------- IQ4_NL quantizer

def _best_index(vals: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Vectorized ggml best_index_int8(16, kvalues_iq4nl, x)."""
    mu = torch.searchsorted(vals, x.contiguous(), right=True).clamp(1, 15)
    lo, hi = vals[mu - 1], vals[mu]
    l = torch.where(x - lo < hi - x, mu - 1, mu)
    l = torch.where(x >= vals[15], 15, l)
    l = torch.where(x <= vals[0], 0, l)
    return l


def _seqsum(p: torch.Tensor) -> torch.Tensor:
    """Left-to-right float32 sum over the last dim (same rounding as the C loop)."""
    s = p[:, 0].clone()
    for j in range(1, p.shape[1]):
        s = s + p[:, j]
    return s


def _quantize_iq4_nl_blocks(x: torch.Tensor) -> torch.Tensor:
    vals = _tt(x.device)["kvalues_iq4nl"]
    v0 = vals[0]                                            # -127.0
    ntry = 7
    weight = x * x
    ax = x.abs()
    imax = ax.argmax(dim=1, keepdim=True)                   # first maximum, like the strict '>' scan
    amax = ax.gather(1, imax)[:, 0]
    mx = x.gather(1, imax)[:, 0]
    zero = amax < 1e-15
    mxs = torch.where(zero, torch.ones_like(mx), mx)        # avoid inf/nan noise in skipped blocks

    def sums(idv):
        q = vals[_best_index(vals, idv[:, None] * x)]
        sumqx = _seqsum(weight * q * x)
        sumq2 = _seqsum(weight * q * q)
        return sumqx, sumq2

    d = -mxs / v0
    sumqx, sumq2 = sums(1 / d)
    d = torch.where(sumq2 > 0, sumqx / sumq2, torch.zeros_like(sumqx))
    best = d * sumqx
    for itry in range(-ntry, ntry + 1):
        sumqx, sumq2 = sums((itry + v0) / mxs)
        upd = (sumq2 > 0) & (sumqx * sumqx > best * sumq2)
        dn = sumqx / sumq2
        d = torch.where(upd, dn, d)
        best = torch.where(upd, dn * sumqx, best)
    scale = torch.where(zero, torch.zeros_like(d), d)
    dh = scale.half()
    idv = torch.where(scale != 0, 1 / scale, torch.zeros_like(scale))
    L = _best_index(vals, idv[:, None] * x).to(torch.uint8)
    q4 = L[:, :16] | (L[:, 16:] << 4)
    dbytes = dh.view(torch.int16).to(torch.int32)
    out = torch.empty(x.shape[0], 18, dtype=torch.uint8, device=x.device)
    out[:, 0] = (dbytes & 0xFF).to(torch.uint8)
    out[:, 1] = ((dbytes >> 8) & 0xFF).to(torch.uint8)
    out[:, 2:] = q4
    return out


def quantize_iq4_nl(x: torch.Tensor, chunk_blocks: int = 1 << 21) -> torch.Tensor:
    """ggml quantize_iq4_nl with quant_weights=NULL: float32 [rows, K] -> uint8 [rows, K//32*18]."""
    if x.dim() != 2 or x.shape[1] % 32:
        raise ValueError("expected [rows, K] with K % 32 == 0")
    rows, k = x.shape
    blocks = x.to(torch.float32).reshape(-1, 32)
    out = torch.empty(blocks.shape[0], 18, dtype=torch.uint8, device=x.device)
    for s in range(0, blocks.shape[0], chunk_blocks):
        out[s:s + chunk_blocks] = _quantize_iq4_nl_blocks(blocks[s:s + chunk_blocks])
    return out.reshape(rows, k // 32 * 18)

#!/usr/bin/env python3
"""Minimal GGUF v3 reader (stdlib + numpy).

    python tools/gguf.py PATH [--tensors]
"""
import argparse
import struct
import sys
from collections import OrderedDict, namedtuple

import numpy as np

# ggml type id -> (name, block bytes, block elements); sizes per ggml-common.h static_asserts
GGML_TYPES = {
    0: ("F32", 4, 1), 1: ("F16", 2, 1), 8: ("Q8_0", 34, 32), 20: ("IQ4_NL", 18, 32), 42: ("Q2_0", 18, 64),
    22: ("IQ2_S", 82, 256), 18: ("IQ3_XXS", 98, 256), 21: ("IQ3_S", 110, 256), 23: ("IQ4_XS", 136, 256),
    24: ("I8", 1, 1), 25: ("I16", 2, 1), 26: ("I32", 4, 1), 27: ("I64", 8, 1), 28: ("F64", 8, 1),
    30: ("BF16", 2, 1), 12: ("Q4_K", 144, 256), 13: ("Q5_K", 176, 256), 14: ("Q6_K", 210, 256),
    15: ("Q8_K", 292, 256),
}

# GGUF metadata value types
_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_NP = {0: "<u1", 1: "<i1", 2: "<u2", 3: "<i2", 4: "<u4", 5: "<i4", 6: "<f4", 7: "<u1", 10: "<u8", 11: "<i8",
       12: "<f8"}

TensorInfo = namedtuple("TensorInfo", "name ne type_id type_name offset nbytes")


class _Reader:
    def __init__(self, buf):
        self.buf, self.pos = buf, 0

    def unpack(self, fmt):
        v = struct.unpack_from(fmt, self.buf, self.pos)[0]
        self.pos += struct.calcsize(fmt)
        return v

    def string(self):
        n = self.unpack("<Q")
        s = bytes(self.buf[self.pos:self.pos + n]).decode("utf-8", errors="replace")
        self.pos += n
        return s

    def value(self, vt):
        if vt == 8:
            return self.string()
        if vt == 9:
            et, n = self.unpack("<I"), self.unpack("<Q")
            if et in _NP:
                arr = np.frombuffer(self.buf, dtype=_NP[et], count=n, offset=self.pos).copy()
                self.pos += arr.nbytes
                return arr.astype(bool) if et == 7 else arr
            return [self.value(et) for _ in range(n)]
        return self.unpack(_SCALAR[vt])


def tensor_nbytes(ne, type_id):
    name, bs, be = GGML_TYPES[type_id]
    n = 1
    for d in ne:
        n *= d
    if ne[0] % be:
        raise ValueError("ne0 %d not a multiple of %s block %d" % (ne[0], name, be))
    return n // be * bs


class GGUFFile:
    def __init__(self, path):
        self.path = path
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        r = _Reader(self.mm)
        if bytes(self.mm[:4]) != b"GGUF":
            raise ValueError("%s: not a GGUF file" % path)
        r.pos = 4
        self.version = r.unpack("<I")
        if self.version != 3:
            raise ValueError("unsupported GGUF version %d" % self.version)
        n_tensors, n_kv = r.unpack("<Q"), r.unpack("<Q")
        self.metadata = OrderedDict()
        for _ in range(n_kv):
            k = r.string()
            self.metadata[k] = r.value(r.unpack("<I"))
        self.alignment = int(self.metadata.get("general.alignment", 32))
        raw = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.unpack("<I")
            ne = [r.unpack("<Q") for _ in range(nd)]
            raw.append((name, ne, r.unpack("<I"), r.unpack("<Q")))
        self.data_start = (r.pos + self.alignment - 1) // self.alignment * self.alignment
        self.tensors = OrderedDict()
        for name, ne, t, off in raw:
            tn = GGML_TYPES[t][0] if t in GGML_TYPES else "type%d" % t
            nb = tensor_nbytes(ne, t) if t in GGML_TYPES else None
            self.tensors[name] = TensorInfo(name, ne, t, tn, self.data_start + off, nb)

    def tensor_bytes(self, name):
        t = self.tensors[name]
        return self.mm[t.offset:t.offset + t.nbytes]


def _fmt(v, limit=8):
    if isinstance(v, np.ndarray):
        s = v[:limit].tolist()
        return "%s[%d] %s%s" % (v.dtype, len(v), s, " ..." if len(v) > limit else "")
    if isinstance(v, list):
        return "list[%d] %r%s" % (len(v), v[:limit], " ..." if len(v) > limit else "")
    if isinstance(v, str) and len(v) > 200:
        return repr(v[:200]) + "... (%d chars)" % len(v)
    return repr(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--tensors", action="store_true")
    a = ap.parse_args()
    g = GGUFFile(a.path)
    print("version %d  alignment %d  data_start %d  tensors %d" % (g.version, g.alignment, g.data_start, len(g.tensors)))
    for k, v in g.metadata.items():
        print("%s = %s" % (k, _fmt(v)))
    if a.tensors:
        for t in g.tensors.values():
            print("%-48s %-8s %-24s off=%d bytes=%s" % (t.name, t.type_name, t.ne, t.offset, t.nbytes))


if __name__ == "__main__":
    sys.exit(main())

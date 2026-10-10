import ctypes
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import iqformats as iq  # noqa: E402

NBLOCKS = 2000
DEQ_FN = {"IQ2_S": "iq2_s", "IQ3_XXS": "iq3_xxs", "IQ3_S": "iq3_s", "IQ4_XS": "iq4_xs",
          "IQ4_NL": "iq4_nl", "Q2_0": "q2_0", "Q8_0": "q8_0"}
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="session")
def clib(tmp_path_factory):
    so = tmp_path_factory.mktemp("ggml_iq") / "libggml_iq.so"
    src = iq.GGML_IQ_DIR / "ggml-iq.c"
    subprocess.run(["gcc", "-O2", "-fno-fast-math", "-ffp-contract=off", "-shared", "-fPIC",
                    "-o", str(so), str(src), "-lm"], check=True)
    lib = ctypes.CDLL(str(so))
    for name in DEQ_FN.values():
        f = getattr(lib, f"dequantize_row_{name}")
        f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
        f.restype = None
    lib.quantize_iq4_nl.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
    lib.quantize_iq4_nl.restype = ctypes.c_size_t
    return lib


def _random_blocks(name, rng):
    bb = iq.BLOCK_BYTES[name]
    raw = rng.integers(0, 256, size=(NBLOCKS, bb), dtype=np.uint8)
    mag = np.exp(rng.uniform(np.log(1e-4), np.log(8.0), size=NBLOCKS))
    d = (mag * rng.choice([-1.0, 1.0], size=NBLOCKS)).astype(np.float16)
    assert np.isfinite(d).all()
    raw[:, 0:2] = d.view(np.uint8).reshape(NBLOCKS, 2)
    return np.ascontiguousarray(raw)


def _c_dequant(lib, name, raw):
    n = raw.shape[0] * iq.BLOCK_ELEMS[name]
    out = np.empty(n, dtype=np.float32)
    getattr(lib, f"dequantize_row_{DEQ_FN[name]}")(raw.ctypes.data, out.ctypes.data, n)
    return torch.from_numpy(out)


def test_tables_match_header():
    t = iq.tables()
    assert t["iq2s_grid"].shape == (1024,) and t["iq3xxs_grid"].shape == (256,)
    assert t["iq3s_grid"].shape == (512,) and t["ksigns_iq2xs"].shape == (128,)
    assert list(t["kvalues_iq4nl"]) == [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", list(DEQ_FN))
def test_dequantize_exact(clib, name, device):
    rng = np.random.default_rng(1234 + iq.TYPE_IDS[name])
    raw = _random_blocks(name, rng)
    ref = _c_dequant(clib, name, raw)
    got = iq.dequantize(name, torch.from_numpy(raw).to(device)).cpu()
    assert got.dtype == torch.float32
    assert torch.equal(got, ref)
    rows = iq.dequantize_rows(iq.TYPE_IDS[name], torch.from_numpy(raw.reshape(100, -1)).to(device)).cpu()
    assert torch.equal(rows.reshape(-1), ref)


def test_quantize_iq4_nl_matches_c(clib):
    rng = np.random.default_rng(7)
    rows, k = 2000, 256
    normal = rng.standard_normal((rows // 2, k))
    heavy = rng.standard_t(2.0, size=(rows - rows // 2, k)) * np.exp(rng.uniform(-6, 3, size=(rows - rows // 2, 1)))
    x = np.ascontiguousarray(np.concatenate([normal, heavy]).astype(np.float32))
    x[3, :32] = 0.0  # an all-zero block
    ref = np.empty((rows, k // 32 * 18), dtype=np.uint8)
    clib.quantize_iq4_nl(x.ctypes.data, ref.ctypes.data, rows, k, None)

    got = iq.quantize_iq4_nl(torch.from_numpy(x)).numpy()
    assert got.shape == ref.shape
    gb, rb = got.reshape(-1, 18), ref.reshape(-1, 18)
    xb = x.reshape(-1, 32)
    mism = np.nonzero((gb != rb).any(axis=1))[0]
    assert len(mism) <= 0.001 * len(gb), f"{len(mism)} of {len(gb)} blocks differ"
    for i in mism:
        eg = ((iq.dequantize("IQ4_NL", gb[i]).numpy().astype(np.float64) - xb[i]) ** 2).sum()
        er = ((iq.dequantize("IQ4_NL", rb[i]).numpy().astype(np.float64) - xb[i]) ** 2).sum()
        assert abs(eg - er) <= 1e-6 * max(er, 1e-30), (i, eg, er)

    if torch.cuda.is_available():
        gc = iq.quantize_iq4_nl(torch.from_numpy(x).cuda()).cpu().numpy()
        assert (gc.reshape(-1, 18) != rb).any(axis=1).sum() <= 0.001 * len(gb)

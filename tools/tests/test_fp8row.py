import importlib.util
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import fp8row  # noqa: E402

CINF = "/home/obelisk/Projects/cinference/tools/convert/quantization/fp8_row.py"
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(scope="module")
def cinference():
    if not os.path.exists(CINF):
        pytest.skip("missing " + CINF)
    spec = importlib.util.spec_from_file_location("cinf_fp8_row", CINF)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass(slots=True) resolves its module
    spec.loader.exec_module(mod)
    return mod


def _cases():
    g = torch.Generator().manual_seed(0)
    n, k = 512, 384
    w = torch.randn(n, k, generator=g)
    w[0] = 0                                                          # zero row
    w[1] = torch.randn(k, generator=g) * 1.5e-36                      # BF16-subnormal row scale
    w[2, :] = 0
    w[2, 5] = -2e-36                                                  # single tiny value, subnormal scale
    w[3] = torch.randn(k, generator=g) * 5e37                         # near-max magnitudes
    w[4:64] *= torch.logspace(-30, 30, 60)[:, None]                   # wide dynamic range
    w[64:128] = torch.distributions.StudentT(1.5).sample((64, k))     # heavy tails
    w[128, ::2] = -0.0                                                # signed zeros mixed in
    return [w.bfloat16(), torch.randn(37, 2560, generator=g).bfloat16()]


@pytest.mark.parametrize("device", DEVICES)
def test_bit_identical_to_cinference(cinference, device):
    for w in _cases():
        ref = cinference.quantize_bf16_rows(w)
        codes, scales = fp8row.quantize_rows(w.to(device))
        assert torch.equal(codes.cpu(), ref.codes)
        assert torch.equal(scales.cpu().view(torch.int16), ref.scales.view(torch.int16))


@pytest.mark.parametrize("device", DEVICES)
def test_scale_underflow_rejected_like_cinference(cinference, device):
    # max_abs/448 rounds to BF16 0 -> scale 0x0001, whose reciprocal overflows float32: both reject.
    w = torch.zeros(2, 64)
    w[1, 3] = 1e-38
    w = w.bfloat16()
    with pytest.raises(ValueError):
        cinference.quantize_bf16_rows(w)
    with pytest.raises(ValueError):
        fp8row.quantize_rows(w.to(device))

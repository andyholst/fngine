# Modified for fngine from cinference:tools/convert/quantization/fp8_row.py @ 383e5db. See NOTICE.
"""Row-scaled FP8 (E4M3FN codes + BF16 row scales) quantization in torch (CPU or CUDA).

Bit-identical to cinference `quantize_bf16_rows` (proved by tools/tests/test_fp8row.py):
scale = bf16_rne(f32(f64(max_abs) / 448)), underflow -> 0x0001, zero rows -> scale 0 and codes 0,
normalized = f32(f64(x) * f64(f32(1 / f64(scale)))), clip to +-448, E4M3FN round-to-nearest-even.
"""
from __future__ import annotations

import torch

E4M3FN_MAX = 448.0


def _bf16_rne_words(v: torch.Tensor) -> torch.Tensor:
    """Nonnegative float32 -> BF16 words (int64), ties to even."""
    bits = v.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16


def quantize_rows(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 [N,K] -> (uint8 E4M3FN codes [N,K], bfloat16 scales [N]) on weight.device."""
    if weight.dtype != torch.bfloat16 or weight.dim() != 2:
        raise TypeError("FP8 quantization source must be a rank-two BF16 tensor")
    x = weight.float()
    if not torch.isfinite(x).all():
        raise ValueError("FP8 quantization source contains NaN or infinity")
    max_abs = x.abs().amax(dim=1)
    zero = max_abs == 0
    raw = (max_abs.double() / E4M3FN_MAX).float()
    words = _bf16_rne_words(raw)
    words = torch.where((words == 0) & ~zero, torch.ones_like(words), words)
    scale32 = (words.to(torch.int32) << 16).view(torch.float32)
    if ((~zero) & (~torch.isfinite(scale32) | (scale32 <= 0))).any():
        raise ValueError("FP8 quantization row scale is not finite and positive")
    recip = torch.where(zero, torch.zeros_like(scale32), (1.0 / scale32.double()).float())
    norm = (x.double() * recip.double()[:, None]).float()
    if not torch.isfinite(norm[~zero]).all():
        raise ValueError("FP8 quantization normalization produced a non-finite value")
    norm = norm.clamp(-E4M3FN_MAX, E4M3FN_MAX)
    codes = norm.to(torch.float8_e4m3fn).view(torch.uint8)
    codes = torch.where(zero[:, None], torch.zeros_like(codes), codes)
    scales = words.to(torch.int16).view(torch.bfloat16)  # words <= 0x7F80 fit in int16
    return codes, scales


def dequantize_rows(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """uint8 E4M3FN codes [N,K] + bfloat16 scales [N] -> float32 [N,K]."""
    return codes.view(torch.float8_e4m3fn).float() * scales.float()[:, None]

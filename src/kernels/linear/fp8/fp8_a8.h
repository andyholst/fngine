// Modified for fngine from cinference:src/ops/linear/fp8/fp8_a8_plan.h @ 383e5db. See NOTICE.
// fngine: raw-pointer workspace carving instead of arenas; the fused RMSNorm producer is dropped.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace fngine::kernels::detail {

struct Fp8A8Workspace {
    std::uint8_t* codes = nullptr;  // E4M3 [T][K]
    float* scales       = nullptr;  // per-token FP32 scale [T]
};

// Bytes for T tokens of width K: codes (256-aligned) then scales.
std::size_t fp8_a8_workspace_bytes(std::int32_t tokens, std::int32_t input_rows);

// Carves the workspace from `base` (256-byte aligned), checking `bytes`.
Fp8A8Workspace fp8_a8_workspace(void* base, std::size_t bytes, std::int32_t tokens,
                                std::int32_t input_rows);

// Per-token E4M3 quantization of x BF16 [T][K]: scale = max|x| / 448, codes = e4m3(x / scale)
// (RNE, saturating). K in {384, 640, 2560, 6144, 10240}.
void launch_fp8_a8_quantize(const __nv_bfloat16* x, std::int32_t tokens, std::int32_t input_rows,
                            Fp8A8Workspace workspace, cudaStream_t stream);

} // namespace fngine::kernels::detail

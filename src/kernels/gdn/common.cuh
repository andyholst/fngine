// Modified for fngine from cinference:src/ops/linear_attention/gated_delta_net/common.cuh @ 383e5db. See NOTICE.
#pragma once

// Gated DeltaNet geometry shared by the GDN kernels. Generic CUDA primitives live under
// kernels/common.

#include "kernels/common/bf16_vector.cuh"
#include "kernels/common/math.cuh"
#include "kernels/common/math.h"
#include "kernels/common/memory.cuh"
#include "kernels/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace fngine::kernels::gdn {

inline constexpr std::int32_t kStateDim  = 128;
inline constexpr std::int32_t kChunkSize = 64;

// fngine geometry (Qwen3.8-Flash-Next): 16 Q/K heads, 48 value heads, head dim 128; value head h
// reads Q/K head h / 3.
inline constexpr std::int32_t kQkHeadCount    = 16;
inline constexpr std::int32_t kValueHeadCount = 48;

} // namespace fngine::kernels::gdn

// Modified for fngine from cinference:src/ops/kernel/gdn_gating.cuh @ 383e5db. See NOTICE.
#pragma once

// Elementwise GDN gate prep over [T][48]. Input ab FP32 [T][96] = a(48) | b(48).
// Transcendentals use FP32 CUDA math functions, not polynomial approximations.

#include "kernels/common/math.cuh"
#include "kernels/common/pdl.cuh"

#include <cstdint>

namespace fngine::kernels::gdn {

// Programmatic dependent (pdl.cuh): the parameters load before pdl_wait(), ab after it.
__global__ void gating_kernel(const float* __restrict__ ab, const float* __restrict__ A_log,
                              const float* __restrict__ dt_bias, float* __restrict__ g,
                              float* __restrict__ beta, std::int64_t n) {
    pdl_trigger();
    const std::int64_t i = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (i >= n) { return; }
    const std::int64_t t = i / 48;
    const int h          = static_cast<int>(i - t * 48);
    const float a_log    = A_log[h];
    const float bias     = dt_bias[h];
    pdl_wait();
    const float* ab_row  = pdl_coherent(ab) + t * 96;
    const float av       = ab_row[h];
    const float bv       = ab_row[48 + h];
    g[i]                 = -expf(a_log) * softplus(av + bias);
    beta[i]              = sigmoid(bv);
}

} // namespace fngine::kernels::gdn

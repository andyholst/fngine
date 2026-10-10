// Modified for fngine from cinference:src/ops/qk_rmsnorm_rope/qk_rmsnorm_rope.cu @ 383e5db. See NOTICE.
// fngine: 24 query / 2 key heads read in place from the fused qkv projection (query interleaved
// with the output gate per head), FP32 effective multipliers, HF float32 RoPE table.
#include "kernels/qsa/qsa.h"

#include "core/device.h"
#include "kernels/common/pdl.cuh"
#include "kernels/common/warp.cuh"
#include "kernels/qsa/qsa_common.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace fngine::kernels {
namespace {

constexpr int kHeadPairs   = qsa::kHeadDim / 2;
constexpr int kRotaryPairs = 16;  // lanes 0..15 hold dims 0..31, lanes 16..31 dims 32..63
constexpr int kRowsPerCta  = 4;
constexpr int kRowsPerToken = qsa::kHeads + qsa::kKvHeads;

// One warp owns one head row of one token, query heads first. Lane l holds bf16 pairs l + 32k,
// so the rotated dims [0, 64) are exactly the step-0 pairs: lane p < 16 holds first-half pair p
// and lane 16 + p its partner; each lane rotates its own pair from the rounded BF16 values.
// Programmatic dependent (pdl.cuh): the norm weights load before pdl_wait(), qkv after it.
__global__ __launch_bounds__(kRowsPerCta * 32) void qsa_qk_norm_rope_kernel(
    const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ q_weight,
    const float* __restrict__ k_weight, const std::int32_t* __restrict__ positions,
    __nv_bfloat162* __restrict__ q_out, __nv_bfloat162* __restrict__ k_out, std::int32_t tokens) {
    pdl_trigger();
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int row =
        static_cast<int>(blockIdx.x) * kRowsPerCta + static_cast<int>(threadIdx.x) / kWarpSize;
    const int token = row / kRowsPerToken;
    const int head  = row - token * kRowsPerToken;
    if (token >= tokens) return;
    const bool query = head < qsa::kHeads;
    const std::int64_t in_offset =
        static_cast<std::int64_t>(token) * qsa::kQkvLd +
        (query ? head * 2 * qsa::kHeadDim : qsa::kKeyCol + (head - qsa::kHeads) * qsa::kHeadDim);
    const auto* weight = reinterpret_cast<const float2*>(query ? q_weight : k_weight);

    float2 values[4];
    float2 weights[4];
#pragma unroll
    for (int step = 0; step < 4; ++step) weights[step] = weight[lane + step * kWarpSize];
    pdl_wait();
    const auto* x = reinterpret_cast<const __nv_bfloat162*>(pdl_coherent(qkv) + in_offset);
    float sum = 0.0f;
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        const int pair = lane + step * kWarpSize;
        values[step]   = __bfloat1622float2(x[pair]);
        sum = __fadd_rn(sum, __fmaf_rn(values[step].x, values[step].x,
                                       __fmul_rn(values[step].y, values[step].y)));
    }
    sum             = warp_sum(sum);
    const float inv = rsqrtf(sum / static_cast<float>(qsa::kHeadDim) + qsa::kEps);

    __nv_bfloat162 out[4];
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        out[step] = __floats2bfloat162_rn(values[step].x * inv * weights[step].x,
                                          values[step].y * inv * weights[step].y);
    }

    const int frequency = 2 * (lane & (kRotaryPairs - 1));
    float s0, c0, s1, c1;
    const std::int32_t position = positions[token];
    qsa_rope_sincos(position, frequency, &s0, &c0);
    qsa_rope_sincos(position, frequency + 1, &s1, &c1);
    const float2 own     = __bfloat1622float2(out[0]);
    const float2 partner = make_float2(__shfl_xor_sync(kFullWarpMask, own.x, kRotaryPairs),
                                       __shfl_xor_sync(kFullWarpMask, own.y, kRotaryPairs));
    const bool first_half = lane < kRotaryPairs;
    const float2 rotated_x =
        rope_rotate_pair(first_half ? own.x : partner.x, first_half ? partner.x : own.x, c0, s0);
    const float2 rotated_y =
        rope_rotate_pair(first_half ? own.y : partner.y, first_half ? partner.y : own.y, c1, s1);
    out[0] = first_half ? __floats2bfloat162_rn(rotated_x.x, rotated_y.x)
                        : __floats2bfloat162_rn(rotated_x.y, rotated_y.y);

    __nv_bfloat162* destination =
        query ? q_out + (static_cast<std::int64_t>(token) * qsa::kHeads + head) * kHeadPairs
              : k_out + (static_cast<std::int64_t>(token) * qsa::kKvHeads + head - qsa::kHeads) *
                            kHeadPairs;
#pragma unroll
    for (int step = 0; step < 4; ++step) destination[lane + step * kWarpSize] = out[step];
}

}  // namespace

void qsa_qk_norm_rope(const __nv_bfloat16* qkv, const float* q_norm, const float* k_norm,
                      const int32_t* positions, int32_t tokens, __nv_bfloat16* q_out,
                      __nv_bfloat16* k_out, cudaStream_t stream) {
    if (tokens <= 0) return;
    const int rows = tokens * kRowsPerToken;
    CUDA_CHECK(pdl_launch(qsa_qk_norm_rope_kernel, dim3((rows + kRowsPerCta - 1) / kRowsPerCta),
                          dim3(kRowsPerCta * 32), 0, stream, qkv, q_norm, k_norm, positions,
                          reinterpret_cast<__nv_bfloat162*>(q_out),
                          reinterpret_cast<__nv_bfloat162*>(k_out), tokens));
}

}  // namespace fngine::kernels

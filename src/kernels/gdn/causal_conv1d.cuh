// Modified for fngine from cinference:src/ops/kernel/causal_conv1d.cuh @ 383e5db. See NOTICE.
//
// Depthwise causal width-4 convolution with fused SiLU over the 10240 GDN conv channels.
// fngine layouts: input x = qkvz columns 0..10239 (row stride 16384), F32 weight [10240][4]
// channel-major (tap 3 multiplies x[t]), BF16 history [3][10240] oldest first, output BF16
// [T][10240]. Every form evaluates conv4_silu, so decode, record and prefill outputs agree bit
// for bit for equal inputs. SiLU is x / (1 + exp(-x)) in FP32.

#pragma once

#include "kernels/common/math.cuh"
#include "kernels/common/pdl.cuh"
#include "kernels/gdn/gdn.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace fngine::kernels::gdn {

inline constexpr int kConvChannelTile = 32;
inline constexpr std::int64_t kConvC  = gdn_dims::kConvChannels;
inline constexpr std::int64_t kXLd    = gdn_dims::kQkvzLd;

__device__ __forceinline__ float conv4_silu(float4 w, __nv_bfloat16 x0, __nv_bfloat16 x1,
                                            __nv_bfloat16 x2, __nv_bfloat16 x3) {
    float acc = w.x * __bfloat162float(x0);
    acc       = fmaf(w.y, __bfloat162float(x1), acc);
    acc       = fmaf(w.z, __bfloat162float(x2), acc);
    acc       = fmaf(w.w, __bfloat162float(x3), acc);
    return silu(acc);
}

__device__ __forceinline__ float4 conv_weight(const float* weight, std::int64_t c) {
    return load_vec<float4>(weight + c * 4);
}

__device__ __forceinline__ std::int64_t conv_history_offset(std::int32_t slot, std::int32_t layer) {
    return (static_cast<std::int64_t>(slot) * gdn_dims::kLayers + layer) *
           static_cast<std::int64_t>(gdn_dims::kConvLayerElems);
}

// T = 1 decode, in place on the selected slot's history. Programmatic dependent (pdl.cuh): only
// the weights load before pdl_wait(); the slot, the history (a back-to-back launch updates it) and
// the qkvz column after it.
__global__ void conv_decode_kernel(const __nv_bfloat16* __restrict__ x,
                                   const float* __restrict__ weight,
                                   __nv_bfloat16* __restrict__ conv_states, std::int32_t layer,
                                   const GdnStepControl* __restrict__ control,
                                   __nv_bfloat16* __restrict__ out) {
    pdl_trigger();
    const std::int64_t c = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (c >= kConvC) { return; }
    const float4 w          = conv_weight(weight, c);
    pdl_wait();
    __nv_bfloat16* history  = conv_states + conv_history_offset(pdl_coherent(control)->source_slot, layer);
    const __nv_bfloat16 s0  = history[c];
    const __nv_bfloat16 s1  = history[kConvC + c];
    const __nv_bfloat16 s2  = history[2 * kConvC + c];
    const __nv_bfloat16 x0  = pdl_coherent(x)[c];
    out[c]                  = __float2bfloat16_rn(conv4_silu(w, s0, s1, s2, x0));
    history[c]              = s1;
    history[kConvC + c]     = s2;
    history[2 * kConvC + c] = x0;
}

// T <= 16 verify columns from the selected slot's history; no state write. Each CTA owns one
// 32-channel tile for every column (threadIdx.y = column). Column t's raw input is recorded.
// Programmatic dependent: only the weights load before pdl_wait().
__global__ void conv_record_kernel(const __nv_bfloat16* __restrict__ x,
                                   const float* __restrict__ weight,
                                   const __nv_bfloat16* __restrict__ conv_states,
                                   std::int32_t layer, const GdnStepControl* __restrict__ control,
                                   __nv_bfloat16* __restrict__ conv_record,
                                   __nv_bfloat16* __restrict__ out) {
    __shared__ __nv_bfloat16 history[3][kConvChannelTile];

    pdl_trigger();
    const std::int32_t lane = static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t t    = static_cast<std::int32_t>(threadIdx.y);
    const std::int64_t c    = static_cast<std::int64_t>(blockIdx.x) * kConvChannelTile + lane;
    const float4 w          = conv_weight(weight, c);
    pdl_wait();

    if (t == 0) {
        const __nv_bfloat16* init = pdl_coherent(conv_states) +
                                    conv_history_offset(pdl_coherent(control)->source_slot, layer);
        for (std::int32_t s = 0; s < 3; ++s) { history[s][lane] = init[s * kConvC + c]; }
    }
    __syncthreads();
    const __nv_bfloat16* xs = pdl_coherent(x);

    const auto tap = [&](std::int32_t pos) {
        return pos < 0 ? history[pos + 3][lane] : xs[static_cast<std::int64_t>(pos) * kXLd + c];
    };
    const __nv_bfloat16 x3 = xs[static_cast<std::int64_t>(t) * kXLd + c];
    out[static_cast<std::int64_t>(t) * kConvC + c] =
        __float2bfloat16_rn(conv4_silu(w, tap(t - 3), tap(t - 2), tap(t - 1), x3));
    conv_record[(static_cast<std::int64_t>(layer) * gdn_dims::kMaxRecordWidth + t) * kConvC + c] = x3;
}

// Prefill: one thread per (token, channel pair).
__global__ void conv_prefill_kernel(const __nv_bfloat16* __restrict__ x,
                                    const float* __restrict__ weight,
                                    const __nv_bfloat16* __restrict__ state_in,
                                    __nv_bfloat16* __restrict__ out, std::int32_t T) {
    constexpr std::int64_t kPairs = kConvC / 2;
    const std::int64_t pair_blocks = div_up(kPairs, static_cast<std::int64_t>(blockDim.x));
    const std::int64_t block       = blockIdx.x;
    const std::int32_t t           = static_cast<std::int32_t>(block / pair_blocks);
    const std::int64_t p =
        (block - static_cast<std::int64_t>(t) * pair_blocks) * blockDim.x + threadIdx.x;
    if (t >= T || p >= kPairs) { return; }

    const auto tap = [&](std::int32_t pos) {
        return pos < 0 ? load_vec<__nv_bfloat162>(state_in + (pos + 3) * kConvC + 2 * p)
                       : load_vec<__nv_bfloat162>(x + static_cast<std::int64_t>(pos) * kXLd + 2 * p);
    };
    const __nv_bfloat162 x0 = tap(t - 3);
    const __nv_bfloat162 x1 = tap(t - 2);
    const __nv_bfloat162 x2 = tap(t - 1);
    const __nv_bfloat162 x3 = tap(t);
    const float lo = conv4_silu(conv_weight(weight, 2 * p), __low2bfloat16(x0), __low2bfloat16(x1),
                                __low2bfloat16(x2), __low2bfloat16(x3));
    const float hi =
        conv4_silu(conv_weight(weight, 2 * p + 1), __high2bfloat16(x0), __high2bfloat16(x1),
                   __high2bfloat16(x2), __high2bfloat16(x3));
    store_vec(out + static_cast<std::int64_t>(t) * kConvC + 2 * p, __floats2bfloat162_rn(lo, hi));
}

// Trailing history after T prefill tokens. Runs after conv_prefill_kernel (stream order), and the
// initial values are loaded before any store, so state_out may alias state_in.
__global__ void conv_prefill_state_kernel(const __nv_bfloat16* x, const __nv_bfloat16* state_in,
                                          __nv_bfloat16* state_out, std::int32_t T) {
    const std::int64_t c = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    if (c >= kConvC) { return; }
    const __nv_bfloat16 old[3] = {state_in[c], state_in[kConvC + c], state_in[2 * kConvC + c]};
    for (std::int32_t s = 0; s < 3; ++s) {
        const std::int32_t pos = T - 3 + s;
        state_out[s * kConvC + c] =
            pos < 0 ? old[pos + 3] : x[static_cast<std::int64_t>(pos) * kXLd + c];
    }
}

} // namespace fngine::kernels::gdn

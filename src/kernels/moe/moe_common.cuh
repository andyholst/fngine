#pragma once

// Device helpers shared by the decode and prefill MoE kernels.
#include "core/model_dims.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace fngine::kernels::moe_detail {

inline constexpr int kExperts = dims::kExperts;
inline constexpr int kTopK = dims::kTopK;
inline constexpr int kPerLane = kExperts / 32;

// Order-preserving map of a float to uint32 (larger float -> larger key).
__device__ __forceinline__ uint32_t float_order_key(float f) {
  const uint32_t u = __float_as_uint(f);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Lane `l` of a warp loads logits[l + 32*i], i < 16 (one token's 512 router logits).
__device__ __forceinline__ void warp_load_logits(const float* logits, float (&v)[kPerLane]) {
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) v[i] = logits[lane + 32 * i];
}

// Top-10 of one token's 512 router logits (loaded by warp_load_logits) by one full warp: value
// descending, ties to the lower expert id; weights renormalized over the 10 (softmax restricted to
// the selection, FP32). Every lane returns the full result in ids[] / weights[]. Taken entries are
// tracked in a per-lane bit mask, so the keys stay in registers (no dynamic indexing).
__device__ __forceinline__ void warp_topk10(const float (&v)[kPerLane], int ids[kTopK],
                                            float weights[kTopK]) {
  const int lane = threadIdx.x & 31;
  uint32_t key[kPerLane];
#pragma unroll
  for (int i = 0; i < kPerLane; ++i) {
    float x = v[i];
    if (x != x) x = -INFINITY;  // NaN never wins
    if (x == 0.0f) x = 0.0f;    // -0 ties with +0
    key[i] = float_order_key(x);
  }
  uint32_t taken = 0;
  float vals[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    // Lane-local argmax as a 4-level tree; a higher-id entry wins only when strictly greater, so
    // equal keys resolve to the lowest id (ids increase with i).
    uint32_t tk[kPerLane];
    int ti[kPerLane];
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      tk[i] = (taken >> i) & 1u ? 0u : key[i];  // taken: below every real key
      ti[i] = i;
    }
#pragma unroll
    for (int w = 1; w < kPerLane; w <<= 1) {
#pragma unroll
      for (int i = 0; i < kPerLane; i += 2 * w) {
        const bool up = tk[i + w] > tk[i];
        tk[i] = up ? tk[i + w] : tk[i];
        ti[i] = up ? ti[i + w] : ti[i];
      }
    }
    const uint32_t best_key = tk[0];
    const uint32_t max_key = __reduce_max_sync(0xffffffffu, best_key);
    const uint32_t cand = best_key == max_key ? uint32_t(lane + 32 * ti[0]) : 0xffffffffu;
    const int win = int(__reduce_min_sync(0xffffffffu, cand));
    if ((win & 31) == lane) taken |= 1u << (win >> 5);
    ids[k] = win;
    const uint32_t u = (max_key & 0x80000000u) ? (max_key & 0x7fffffffu) : ~max_key;
    vals[k] = __uint_as_float(u);
  }
  float sum = 0.f;
  const float top = vals[0];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    vals[k] = expf(vals[k] - top);
    sum += vals[k];
  }
  const float inv = 1.0f / sum;
#pragma unroll
  for (int k = 0; k < kTopK; ++k) weights[k] = vals[k] * inv;
}

// Exclusive prefix sum over blockDim.x (multiple of 32, <= 1024) values; returns the exclusive
// prefix of this thread and writes the block total to *total. `scratch` holds 32 ints.
__device__ __forceinline__ int block_exclusive_scan(int v, int* scratch, int* total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
  int incl = v;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const int n = __shfl_up_sync(0xffffffffu, incl, o);
    if (lane >= o) incl += n;
  }
  if (lane == 31) scratch[warp] = incl;
  __syncthreads();
  if (warp == 0) {
    int s = lane < warps ? scratch[lane] : 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int n = __shfl_up_sync(0xffffffffu, s, o);
      if (lane >= o) s += n;
    }
    scratch[lane] = s;  // inclusive warp totals
  }
  __syncthreads();
  const int base = warp > 0 ? scratch[warp - 1] : 0;
  *total = scratch[warps - 1];
  __syncthreads();
  return base + incl - v;
}

// Sum of `v` over the whole block (multiple of 32 threads, <= 1024); every thread receives it.
// `scratch` holds 32 floats and must be shared memory owned by the caller.
__device__ __forceinline__ float block_reduce_sum(float v, float* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  float total = warp == 0 ? (lane < warps ? scratch[lane] : 0.0f) : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) total += __shfl_down_sync(0xffffffffu, total, o);
  }
  if (warp == 0 && lane == 0) scratch[0] = total;
  __syncthreads();
  const float out = scratch[0];
  __syncthreads();  // scratch may be reused by the next call
  return out;
}

}  // namespace fngine::kernels::moe_detail

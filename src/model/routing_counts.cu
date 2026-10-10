#include "model/routing_counts.h"

#include "core/model_dims.h"

#include <cmath>

namespace fngine::model {
namespace {

__global__ void set_entries_kernel(int32_t* table, SlotWrites writes) {
  const int i = int(threadIdx.x);
  if (i < writes.n) table[writes.index[i]] = writes.value[i];
}

// (value, id) pairs: `better` prefers a valid id, then the value order, then the lower id.
struct Pick {
  float v;
  int32_t e;
};

template <bool Max>
__device__ __forceinline__ Pick better(Pick x, Pick y) {
  if (x.e < 0) return y;
  if (y.e < 0) return x;
  if (Max ? y.v > x.v : y.v < x.v) return y;
  if (y.v == x.v && y.e < x.e) return y;
  return x;
}

template <bool Max>
__device__ __forceinline__ Pick warp_pick(Pick p) {
  for (int off = 16; off; off >>= 1) {
    const Pick o{__shfl_xor_sync(0xffffffffu, p.v, off), __shfl_xor_sync(0xffffffffu, p.e, off)};
    p = better<Max>(p, o);
  }
  return p;
}

constexpr int kThreads = dims::kExperts;  // one thread per expert
constexpr int kWarps = kThreads / 32;

__global__ void __launch_bounds__(kThreads) swap_candidates_kernel(const uint32_t* counts, uint32_t* last,
                                                                  float* ema, const int32_t* slot_of,
                                                                  float keep, SwapCandidate* out) {
  const int l = int(blockIdx.x), e = int(threadIdx.x);
  const int i = l * dims::kExperts + e;
  const uint32_t c = counts[i];
  const float v = ema[i] * keep + float(c - last[i]);
  last[i] = c;
  ema[i] = v;
  const bool gpu = slot_of[i] >= 0;
  Pick hot = warp_pick<true>({v, gpu ? -1 : e});
  Pick cold = warp_pick<false>({v, gpu ? e : -1});
  __shared__ Pick s_hot[kWarps], s_cold[kWarps];
  const int warp = e / 32, lane = e % 32;
  if (lane == 0) {
    s_hot[warp] = hot;
    s_cold[warp] = cold;
  }
  __syncthreads();
  if (warp != 0) return;
  hot = warp_pick<true>(lane < kWarps ? s_hot[lane] : Pick{0.0f, -1});
  cold = warp_pick<false>(lane < kWarps ? s_cold[lane] : Pick{0.0f, -1});
  if (lane == 0) out[l] = {hot.e, cold.e, hot.v, cold.v};
}

}  // namespace

cudaError_t set_slot_entries(int32_t* table, const SlotWrites& writes, cudaStream_t s) {
  if (!table || writes.n < 0 || writes.n > SlotWrites::kMax) return cudaErrorInvalidValue;
  if (writes.n == 0) return cudaSuccess;
  set_entries_kernel<<<1, SlotWrites::kMax, 0, s>>>(table, writes);
  return cudaGetLastError();
}

cudaError_t swap_candidates(const uint32_t* counts, uint32_t* last, float* ema, const int32_t* slot_of,
                            float keep, SwapCandidate* out, cudaStream_t s) {
  if (!counts || !last || !ema || !slot_of || !out || !std::isfinite(keep) || keep < 0.0f)
    return cudaErrorInvalidValue;
  swap_candidates_kernel<<<dims::kLayers, kThreads, 0, s>>>(counts, last, ema, slot_of, keep, out);
  return cudaGetLastError();
}

cudaError_t preload_swap_kernels() {
  cudaFuncAttributes attr{};
  if (const cudaError_t e = cudaFuncGetAttributes(&attr, set_entries_kernel); e != cudaSuccess) return e;
  return cudaFuncGetAttributes(&attr, swap_candidates_kernel);
}

}  // namespace fngine::model

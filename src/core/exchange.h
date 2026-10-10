#pragma once

// Host-mapped GPU<->CPU exchange used inside captured CUDA graphs.
//
// Protocol (one "invocation" = one decode/verify step or one prefill chunk, identified by a
// nonzero uint32 sequence number `seq` that the host writes before launching):
//   1. The GPU router kernel of MoE layer l writes the cold (token, expert, weight) lists and the
//      BF16 MoE input into the layer's publish block; every thread that wrote host memory executes
//      __threadfence_system() before a CTA barrier, after which one thread stores
//      `control.ready[l] = seq` (st.relaxed.sys: the data is already visible system-wide).
//   2. CPU expert workers acquire-spin on `ready[l] == seq`, compute the cold experts, write FP32
//      partial sums into the result block, then release-store `done[l] = seq`.
//   3. The GPU combine kernel waits for `done[l] == seq`: one thread per launch acquire-spins on it
//      (ld.acquire.sys, __nanosleep(64), 10 s clock64 timeout -> sets `control.error`) and relays
//      the outcome to the other CTAs through device memory (exchange_wait_sys_relayed); all CTAs
//      then read the result with system-coherent loads and add it to the hot- and shared-expert sums.
// Decode mode (per-layer publish/result blocks): if a layer has no cold pairs the GPU skips the
// wait; the router still publishes n_cold = 0 and releases ready[l], and the CPU advances without
// writing done[l].
// Prefill mode (one shared PrefillExchangeView for all layers): the CPU release-stores done[l]
// for every layer, including n_cold == 0, and the GPU always waits on done[l] when a CPU tier
// exists, so the GPU never overwrites the shared region while workers still read layer l.
// The n-gram rows for the PLE layer use the same pattern in the opposite direction: the host
// stages raw rows, then release-stores `ple_ready = seq`; the GPU acquire-spins before reading.
// The GPU signals completion of a whole step with `step_done = seq`.
#include "core/model_dims.h"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

#include <atomic>

namespace fngine {

struct alignas(64) SeqCell {
  uint32_t value;
  uint32_t pad_[15];
};
static_assert(sizeof(SeqCell) == 64);

inline constexpr int32_t kExchangeLayers = dims::kLayers;  // MTP experts never go to the CPU.
inline constexpr int32_t kDecodeExchangeTokens = dims::kMaxStepTokens;

// Error codes in ExchangeControl::error (low 16 bits = layer + 1 where applicable).
inline constexpr uint32_t kExchangeErrorMoeTimeout = 1u << 16;
inline constexpr uint32_t kExchangeErrorPleTimeout = 2u << 16;

struct alignas(64) ExchangeControl {
  SeqCell ready[kExchangeLayers];
  SeqCell done[kExchangeLayers];
  SeqCell ple_ready;
  SeqCell step_done;
  SeqCell error;
  SeqCell abort;
};

// Decode/verify exchange for one layer (T <= 16), indexed by step token.
struct alignas(64) MoeDecodePublish {
  int32_t tokens;  // T of this step
  int32_t n_cold;  // total cold (token, expert) pairs
  int32_t pad_[14];
  int32_t cold_count[kDecodeExchangeTokens];
  int32_t cold_expert[kDecodeExchangeTokens][dims::kTopK];
  float cold_weight[kDecodeExchangeTokens][dims::kTopK];
  __nv_bfloat16 x[kDecodeExchangeTokens][dims::kHidden];
};

struct alignas(64) MoeDecodeResult {
  float y[kDecodeExchangeTokens][dims::kHidden];
};

// Whole host-mapped decode exchange: control block plus per-layer publish/result blocks.
struct alignas(4096) DecodeExchange {
  ExchangeControl control;
  MoeDecodePublish publish[kExchangeLayers];
  MoeDecodeResult result[kExchangeLayers];
};

// Prefill exchange: one shared region sized for the prefill chunk (layers are strictly
// sequential, so one region serves all layers). Only tokens with at least one cold pair are
// published ("compacted" rows); `row_token[i]` is the chunk token index of compacted row i.
struct PrefillExchangeView {
  int32_t capacity = 0;              // max tokens per chunk
  int32_t* header = nullptr;         // [0]=T, [1]=n_rows (compacted), [2]=n_cold, [3..15] pad
  int32_t* row_token = nullptr;      // [capacity]
  int32_t* cold_count = nullptr;     // [capacity]
  int32_t* cold_expert = nullptr;    // [capacity][10]
  float* cold_weight = nullptr;      // [capacity][10]
  __nv_bfloat16* x = nullptr;        // [capacity][2560]
  float* y = nullptr;                // [capacity][2560]
};

// Bytes of the host-mapped region backing a PrefillExchangeView of `capacity` tokens.
std::size_t prefill_exchange_bytes(int32_t capacity);
// Carve a view out of `base` (64-byte aligned, at least prefill_exchange_bytes(capacity)).
PrefillExchangeView make_prefill_exchange_view(void* base, int32_t capacity);

// ---------------------------------------------------------------- system-scope helpers
#if defined(__CUDACC__)
__device__ __forceinline__ void exchange_store_release_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
// Flag store for data that every writer already fenced with __threadfence_system() (ordered after
// those fences by a CTA barrier): no release fence, so no second PCIe flush.
__device__ __forceinline__ void exchange_store_relaxed_sys(uint32_t* p, uint32_t v) {
  asm volatile("st.relaxed.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ uint32_t exchange_load_acquire_sys(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ float exchange_load_relaxed_sys(const float* p) {
  float v;
  asm volatile("ld.relaxed.sys.global.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ float4 exchange_load_relaxed_sys(const float4* p) {
  float4 v;
  asm volatile("ld.relaxed.sys.global.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
// Spin until *p == expected or the timeout (clock64 cycles) elapses. Returns false on timeout.
__device__ __forceinline__ bool exchange_wait_sys(const uint32_t* p, uint32_t expected,
                                                  long long timeout_cycles) {
  const long long start = clock64();
  while (exchange_load_acquire_sys(p) != expected) {
    __nanosleep(64);
    if (clock64() - start > timeout_cycles) return false;
  }
  return true;
}
// Grid-wide wait with a single host poller. Concurrent GPU reads of one host cache line serialize
// (about 0.8 us per concurrent reader on this platform) and hold off the CPU's store to that line:
// with every combine CTA polling done[l], the store reached the GPU 25-90 us late. So only the
// `poller` thread (one per grid) spins on *p; it then publishes the outcome to `relay` (device
// memory, {seq, ok}, zero-initialized) with a gpu-scope release, which the other CTAs' threads
// acquire. The chain host release -> acquire.sys -> release.gpu -> acquire.gpu makes everything
// the host wrote before *p visible to every waiter. Returns false on timeout. `expected` must be
// nonzero and unique per wait (a step sequence number).
__device__ __forceinline__ bool exchange_wait_sys_relayed(const uint32_t* p, uint32_t expected,
                                                          bool poller, uint32_t* relay,
                                                          long long timeout_cycles) {
  if (poller) {
    const bool ok = exchange_wait_sys(p, expected, timeout_cycles);
    asm volatile("st.relaxed.gpu.global.u32 [%0], %1;" ::"l"(relay + 1), "r"(ok ? 1u : 0u) : "memory");
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(relay), "r"(expected) : "memory");
    return ok;
  }
  const long long start = clock64();
  for (;;) {
    uint32_t v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(relay) : "memory");
    if (v == expected) break;
    __nanosleep(64);
    if (clock64() - start > timeout_cycles) return false;
  }
  uint32_t ok;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(ok) : "l"(relay + 1) : "memory");
  return ok != 0;
}
// ~10 s at the RTX 5090 boost clock.
inline constexpr long long kExchangeTimeoutCycles = 10ll * 2'500'000'000ll;
#endif

// Host-side acquire/release on exchange cells (CPU workers and the engine thread).
inline uint32_t host_load_acquire(const uint32_t& v) {
  return std::atomic_ref<uint32_t>(const_cast<uint32_t&>(v)).load(std::memory_order_acquire);
}
inline void host_store_release(uint32_t& v, uint32_t x) {
  std::atomic_ref<uint32_t>(v).store(x, std::memory_order_release);
}

}  // namespace fngine
